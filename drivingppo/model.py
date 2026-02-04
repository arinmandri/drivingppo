from typing import Callable, Literal
import time

from .world import World
from .environment import WorldEnv
from .common import (
    LOOKAHEAD_POINTS,
    EACH_POINT_INFO_SIZE,
    OBSERVATION_IND_SPD,
    OBSERVATION_IND_WPOINT_0,
    OBSERVATION_IND_WPOINT_1,
    OBSERVATION_IND_WPOINT_E,
    OBSERVATION_DIM,
)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# 훈련 결과 저장
LOG_DIR = f"./ppo_tensorboard_logs/"
CHECKPOINT_DIR = './ppo_world_checkpoints/'


class NoFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        super(NoFeatureExtractor, self).__init__(observation_space, features_dim=OBSERVATION_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations



class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        flatten_output_dim = LOOKAHEAD_POINTS * hidden_dim

        total_feature_dim = 1 + flatten_output_dim

        super(MLPFeatureExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        input_dim = LOOKAHEAD_POINTS * EACH_POINT_INFO_SIZE

        self.layer0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, flatten_output_dim),
            nn.ReLU(),
            nn.Linear(flatten_output_dim, flatten_output_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        output0 = self.layer0(path_data)

        return torch.cat((speed, output0), dim=1)


class RNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        flatten_output_dim = LOOKAHEAD_POINTS * hidden_dim
        total_feature_dim = 1 + flatten_output_dim

        super(RNNFeatureExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        self.layer0 = nn.RNN(
            input_size=EACH_POINT_INFO_SIZE,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity='relu'
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        reshaped_path = path_data.reshape(-1, LOOKAHEAD_POINTS, EACH_POINT_INFO_SIZE)  # [Batch, 길이×채널] -> [Batch, 길이(시간), 채널]
        output, _ = self.layer0(reshaped_path)

        output0 = torch.flatten(output, start_dim=1)

        return torch.cat((speed, output0), dim=1)


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        flatten_output_dim = LOOKAHEAD_POINTS * hidden_dim
        total_feature_dim = 1 + flatten_output_dim

        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=EACH_POINT_INFO_SIZE,
                out_channels=hidden_dim,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        reshaped = path_data.reshape(-1, LOOKAHEAD_POINTS, EACH_POINT_INFO_SIZE).permute(0, 2, 1)  # [Batch, 길이×채널] -> [Batch, 채널, 길이]

        feature = self.cnn(reshaped)[:, :, :LOOKAHEAD_POINTS]

        output0 = torch.flatten(feature, start_dim=1)

        return torch.cat((speed, output0), dim=1)


class CascadedPathEncoder(nn.Module):
    def __init__(self, num_points, point_dim, hidden_dim):
        super(CascadedPathEncoder, self).__init__()

        self.num_points = num_points
        self.point_dim  = point_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(point_dim + hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_points)
        ])

    def forward(self, path_data):
        chunks = torch.chunk(path_data, self.num_points, dim=1)
        features = []

        batch_size = path_data.shape[0]
        curr_hidden = torch.zeros(batch_size, self.hidden_dim, device=path_data.device)  # 첫번째 waypoint와 결합될 빈값.

        for i, layer in enumerate(self.layers):
            current_wp = chunks[i]
            combined = torch.cat((curr_hidden, current_wp), dim=1)
            curr_hidden = layer(combined)
            features.append(curr_hidden)

        # 모든 단계의 추론결과 연결 [Batch, num_points × hidden_dim]
        return torch.cat(features, dim=1)


class CascadedPathExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        total_feature_dim = 1 + LOOKAHEAD_POINTS * hidden_dim

        super(CascadedPathExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        # 경로 순차적 연관
        self.layer0 = CascadedPathEncoder(
            num_points=LOOKAHEAD_POINTS,
            point_dim=EACH_POINT_INFO_SIZE,
            hidden_dim=hidden_dim
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed           = observations[:, OBSERVATION_IND_SPD         : OBSERVATION_IND_SPD+1]
        path_data       = observations[:, OBSERVATION_IND_WPOINT_0    : OBSERVATION_IND_WPOINT_E]

        output0 = self.layer0(path_data)

        return torch.cat((speed, output0), dim=1)



def train_start(
        gen_env:Callable[[], WorldEnv],
        steps:int,
        save_path:str|None=None,
        save_freq:int=0,
        tb_log:bool=False,
        run_name:str='DPPO',
        *,
        vec_env:Literal['dummy', 'subp']|VecEnv='dummy',
        lr=3e-4,
        gamma=0.9,
        ent_coef=0.01,
        n_steps=1024,
        batch_size=256,
        seed=42
) -> PPO:
    """
    학습 처음부터
    """

    if vec_env == 'dummy':
        vec_env_cls = DummyVecEnv
    elif vec_env == 'subp':
        vec_env_cls = SubprocVecEnv
    else:
        raise Exception(f'unknown vec_env: {vec_env}')
    vec_env = make_vec_env(gen_env, n_envs=1, vec_env_cls=vec_env_cls, seed=seed)# n_envs: 병렬 환경 수

    policy_kwargs = dict(
        features_extractor_class=CascadedPathExtractor,
        features_extractor_kwargs=dict(),
        net_arch=dict(
            pi=[256, 256], # Actor
            vf=[256, 256, 128]  # Critic
        )
    )

    # PPO 모델
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,

        verbose=1,
        tensorboard_log=LOG_DIR  if tb_log  else None,

        # 학습 하이퍼파라미터
        learning_rate=lr,
        gamma=gamma,           # 미래 보상 할인율
        ent_coef=ent_coef,     # 엔트로피: 장애물 거의 없는 환경 - 약하게
        n_steps=n_steps,       # 데이터 수집 스텝 (버퍼 크기, NUM_ENVS * n_steps = 총 수집 데이터량)
        batch_size=batch_size, # 미니 배치 크기

        device="auto"  # GPU 사용 설정
    )

    # 콜백 - 모델 저장
    checkpoint_callback = None
    if save_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix='check'
        )

    print("=== PPO 학습 시작 ===")

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        progress_bar=True,
    )

    # 최종 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"=== 학습 완료: {CHECKPOINT_DIR+save_path} ===")

    vec_env.close()  # 환경 정리

    return model


def train_resume(
        model:PPO|str,
        gen_env: Callable[[], WorldEnv],
        steps: int,
        save_path:str|None=None,
        save_freq:int=0,
        tb_log:bool=False,
        run_name:str='DPPO',
        *,
        vec_env:Literal['dummy', 'subp']|VecEnv='dummy',
        log_std=None,
        lr=1e-4,
        gamma=0.99,
        ent_coef=0.0,
        seed=42
) -> PPO:
    """
    기존 모델 추가학습
    """

    if vec_env == 'dummy':
        vec_env_cls = DummyVecEnv
    elif vec_env == 'subp':
        vec_env_cls = SubprocVecEnv
    else:
        raise Exception(f'unknown vec_env: {vec_env}')
    vec_env = make_vec_env(gen_env, n_envs=4, vec_env_cls=vec_env_cls, seed=seed)

    # 모델 로드 (학습된 모델의 환경도 함께 로드)
    if type(model) == str:
        print(f"=== 체크포인트 로드: {CHECKPOINT_DIR+model} ===")
        model = PPO.load(
            path=CHECKPOINT_DIR+model,
            env=vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR  if tb_log  else None,

            learning_rate=lr,
            gamma=gamma,       # 미래 보상 할인율
            ent_coef=ent_coef, # 새로운 환경 -> 새로운 시도 위해 엔트로피 높임.
        )
    assert isinstance(model, PPO)

    # 콜백 - 모델 저장
    checkpoint_callback = None
    if save_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix='check'
        )

    if log_std:
        with torch.no_grad():
            model.policy.log_std.fill_(log_std)

    print(f"=== 학습 재개 (현재 스텝: {model.num_timesteps} / 목표: {steps + model.num_timesteps} / 남은: {steps}) ===")

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        progress_bar=True,

        reset_num_timesteps=False # 내부 타임스텝 카운터 초기화 여부
    )

    # 최종 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"=== 학습 완료 ({model.num_timesteps} 스텝): {CHECKPOINT_DIR+save_path} ===")

    vec_env.close()  # 환경 정리

    return model


def run(
        world_generator:Callable[[], World],
        model:PPO|str,
        time_spd=2.0,
        time_step=111,
        step_per_control=3,
        auto_close_at_end=True,
    ):
    """
    모델 시각적 확인용 실행
    """

    env = WorldEnv(
        world_generator=world_generator,
        time_step=time_step,
        step_per_control=1,
        render_mode='debug',
        auto_close_at_end=auto_close_at_end
    )

    if type(model) == str:
        model = PPO.load(CHECKPOINT_DIR+model, env=env)
    assert isinstance(model, PPO)

    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0

    while not terminated and not truncated:

        action, _ = model.predict(obs, deterministic=True)  # 에이전트가 행동 선택
        for _ in range(step_per_control):
            obs, reward, terminated, truncated, info = env.step(action)  # 행동 실행
            episode_reward += reward
            env.render()  # 시각화 호출
            time.sleep(time_step / 1000.0 / time_spd)# 시각화 프레임을 위해 딜레이 추가
            if terminated or truncated: break

    print(f"에피소드 종료. 총 보상: {episode_reward:.2f}")

    env.close()
