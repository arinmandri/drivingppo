from typing import Callable, Literal
import os, time, random
from collections import defaultdict

from .world import World
from .environment import WorldEnv
from .common import (
    WORLD_DT,
    ACTION_REPEAT,
    LOOKAHEAD_POINTS,
    EACH_POINT_INFO_SIZE,
    LIDAR_NUM,
    OBSERVATION_IND_SPD,
    OBSERVATION_IND_WPOINT_0,
    OBSERVATION_IND_WPOINT_1,
    OBSERVATION_IND_WPOINT_E,
    OBSERVATION_DIM_WPOINT,
    OBSERVATION_IND_LIDAR_S,
    OBSERVATION_IND_LIDAR_E,
    OBSERVATION_DIM_LIDAR,
    OBSERVATION_DIM,
)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# 훈련 결과 저장
LOG_DIR = f"./logs/"
CHECKPOINT_DIR = './checks/'


class NoFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        super(NoFeaturesExtractor, self).__init__(observation_space, features_dim=OBSERVATION_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations



class MLPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        flatten_output_dim = LOOKAHEAD_POINTS * hidden_dim

        super(MLPFeaturesExtractor, self).__init__(observation_space, features_dim=1 + flatten_output_dim)

        input_dim = LOOKAHEAD_POINTS * EACH_POINT_INFO_SIZE

        self.layer0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, flatten_output_dim),
            nn.ReLU(),
            nn.Linear(flatten_output_dim, flatten_output_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        output0 = self.layer0(path_data)

        return torch.cat((speed, output0), dim=1)


class RNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        flatten_output_dim = LOOKAHEAD_POINTS * hidden_dim

        super(RNNFeaturesExtractor, self).__init__(observation_space, features_dim=1 + flatten_output_dim)

        self.layer0 = nn.RNN(
            input_size=EACH_POINT_INFO_SIZE,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity='relu'
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        reshaped_path = path_data.reshape(-1, LOOKAHEAD_POINTS, EACH_POINT_INFO_SIZE)  # [Batch, 길이×채널] -> [Batch, 길이(시간), 채널]
        output, _ = self.layer0(reshaped_path)

        output0 = torch.flatten(output, start_dim=1)

        return torch.cat((speed, output0), dim=1)


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        hidden_dim = 16
        flatten_output_dim = LOOKAHEAD_POINTS * hidden_dim

        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim=1 + flatten_output_dim)

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
        speed     = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        reshaped = path_data.reshape(-1, LOOKAHEAD_POINTS, EACH_POINT_INFO_SIZE).permute(0, 2, 1)  # [Batch, 길이×채널] -> [Batch, 채널, 길이]

        feature = self.cnn(reshaped)[:, :, :LOOKAHEAD_POINTS]

        output0 = torch.flatten(feature, start_dim=1)

        return torch.cat((speed, output0), dim=1)


class MyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

        self.cascade_hidden_dim = 16
        feature1_dim = 64
        feature2_dim = 128
        total_feature_dim = 1 + OBSERVATION_DIM_WPOINT + feature1_dim + feature2_dim

        super(MyFeatureExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        # 라이다 CNN
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=6, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=feature1_dim),
            nn.Flatten(),
            nn.Linear(6 * feature1_dim, feature1_dim)
        )

        # 속도 + 첫번째목표 + 라이다
        self.layer2 = nn.Sequential(
            nn.Linear(1 + EACH_POINT_INFO_SIZE + feature1_dim, feature2_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed           = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        wpoint0         = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_1]
        path_data       = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]
        lidar_dis_data  = observations[:, OBSERVATION_IND_LIDAR_S  : OBSERVATION_IND_LIDAR_E]

        output1 = self.layer1(lidar_dis_data.unsqueeze(1))
        output2 = self.layer2(torch.cat((speed, wpoint0, output1), dim=1))

        return torch.cat((speed, path_data, output1, output2), dim=1)


class TensorboardCallback(BaseCallback):
    """
    환경(Env)에서 info['episode_metrics']로 전달한 커스텀 값을
    텐서보드에 기록(Log)하는 콜백
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 벡터 환경(VecEnv)을 고려하여 모든 환경의 info를 확인
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                # WorldEnv.step에서 넣어준 episode_metrics가 있으면 기록
                if 'episode_metrics' in info:
                    for key, value in info['episode_metrics'].items():
                        self.logger.record(key, value)
        return True


def train_start(
        gen_env:Callable[[], WorldEnv],
        steps:int=0,  # 0: 학습 없이 생성만 된 모델 반환.
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
        progress_bar=True,
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
        features_extractor_class=NoFeaturesExtractor,
        features_extractor_kwargs=dict(),
        net_arch=dict(
            pi=[512, 512], # Actor
            vf=[512, 512, 256]  # Critic
        )
    )
    print('POLICY:', policy_kwargs)

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

    if steps <= 0:
        # 학습 없이 모델 바로 반환
        print("=== PPO 모델 생성 ===")
        if save_path:
            model.save(CHECKPOINT_DIR+save_path)
        return model

    # 콜백
    callbacks:list[BaseCallback] = [TensorboardCallback()]  # 요소별 점수
    if save_freq:
        # 모델 저장 콜백
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix='check'
        )
        callbacks.append(checkpoint_callback)

    print("=== PPO 학습 시작 ===")

    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
        progress_bar=progress_bar,
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
        reset_num_timesteps=False,
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
        progress_bar=True,
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

    is_temp_file = False
    if isinstance(model, str):
        model_loading_path = model
    elif isinstance(model, PPO):
        model_loading_path = 'temp_model-' + time.strftime('%y%m%d%H%M%S') + str(random.randint(0, 9999))
        is_temp_file = True
        model.save(CHECKPOINT_DIR + model_loading_path)
    else:
        raise ValueError(f'model should be str or PPO not {type(model)}')

    print(f"=== 체크포인트 로드: {CHECKPOINT_DIR+model_loading_path} ===")
    model = PPO.load(
        path=CHECKPOINT_DIR+model_loading_path,
        env=vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR  if tb_log  else None,

        learning_rate=lr,
        gamma=gamma,
        ent_coef=ent_coef,
    )
    assert isinstance(model, PPO)

    # 콜백
    callbacks:list[BaseCallback] = [TensorboardCallback()]  # 커스텀 매트릭
    if save_freq:
        # 모델 저장 콜백
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix='check'
        )
        callbacks.append(checkpoint_callback)

    if log_std:
        with torch.no_grad():
            model.policy.log_std.fill_(log_std)

    print(f"=== 학습 재개 (현재 스텝: {model.num_timesteps} / 목표: {steps + model.num_timesteps} / 남은: {steps}) ===")

    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        tb_log_name=run_name,
        log_interval=10,
        progress_bar=progress_bar,

        reset_num_timesteps=reset_num_timesteps # 내부 타임스텝 카운터 초기화 여부
    )

    # 최종 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"=== 학습 완료 ({model.num_timesteps} 스텝): {CHECKPOINT_DIR+save_path} ===")

    # 임시 파일 삭제 로직
    if is_temp_file:
        temp_file_path = CHECKPOINT_DIR + model_loading_path + ".zip"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    vec_env.close()

    return model


def run(
        world_generator:Callable[[], World],
        model:PPO|str,
        time_spd=2.0,
        time_step=WORLD_DT,
        step_per_control=ACTION_REPEAT,
        auto_close_at_end=True,
    ):
    """
    모델 시각적 확인용 실행
    """

    env = WorldEnv(
        world_generator=world_generator,
        time_step=time_step,
        action_repeat=1,
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


def evaluate(
        model:PPO|str,
        world_generator:Callable[[], World],
        episode_num=100,
        verbose:bool=True,
        csv_path:str="",
) -> dict:
    import numpy as np
    import pandas as pd

    env = WorldEnv(
        world_generator=world_generator,
        render_mode=None,
    )

    if type(model) == str:
        model = PPO.load(CHECKPOINT_DIR+model, env=env)
    assert isinstance(model, PPO)

    print(f"{episode_num}회 에피소드 평가 시작...")

    all_metrics = defaultdict(list)
    episode_rewards = []
    episode_lengths = []

    for i in range(episode_num):
        i1 = i+1
        checkPeriod = episode_num // 10
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        esteps = 0

        while not (done or truncated):
            # 에피소드 진행
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            esteps += 1

            if done or truncated:
                episode_rewards.append(total_reward)
                episode_lengths.append(esteps)

                if 'episode_metrics' in info:
                    for key, value in info['episode_metrics'].items():#type:ignore
                        all_metrics[key].append(value)

                # 진행 상황 출력
                if i1 % checkPeriod == 0:
                    if verbose: print(f"[{i1}/{episode_num}] 완료 - Reward: {total_reward:.2f}, Steps: {esteps}")

    if verbose: 
        print("\n" + "="*41)
        print(f"평가 결과 ({episode_num} 에피소드 평균)")
        print("="*41)
        print(f"Total Reward  : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Episode Length: {np.mean(episode_lengths):.1f}")

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)

        # 수: 평균, 표준편차
        num_df = df_metrics.select_dtypes(include=[np.number])
        if not num_df.empty and verbose:
            print("-" * 41)
            summary = num_df.describe().loc[['mean', 'std']].T
            summary['mean'] = summary['mean'].map('{:.4f}'.format)  # 지수표기 안 함.
            print(summary)

        # 문자열(범주형): 종류별 비율
        cat_df = df_metrics.select_dtypes(exclude=[np.number])
        if not cat_df.empty and verbose:
            print("-" * 41)
            for col in cat_df.columns:
                print(f"\n* {col}")
                counts = df_metrics[col].value_counts()
                ratios = df_metrics[col].value_counts(normalize=True)

                for idx, val in counts.items():
                    ratio = ratios[idx] * 100  #type:ignore
                    print(f"   - {idx:<10}: {val:3d}회 ({ratio:5.1f}%)")

        # CSV 저장
        if csv_path:
            df_metrics.to_csv(csv_path, index=False)
            if verbose: 
                print(f"\n💾 세부 결과 저장됨: {csv_path}")
            if verbose: print(f"\n세부 결과가 저장: {csv_path}")
    else:
        if verbose: print("\n⚠️ info['episode_metrics']가 발견되지 않음.")

    env.close()

    return all_metrics
