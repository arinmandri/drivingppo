from typing import Callable, Literal
import os, time, random, json
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


class FE__I(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, **noargs):

        super(FE__I, self).__init__(observation_space, features_dim=OBSERVATION_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class FE__VMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, out_dim_per_wp=16):

        total_feature_dim = 1 + LOOKAHEAD_POINTS * out_dim_per_wp  # WSWE와 동일한 최종 출력 차원

        super().__init__(observation_space, features_dim=total_feature_dim)

        path_input_dim = OBSERVATION_DIM - 1
        path_output_dim = LOOKAHEAD_POINTS * out_dim_per_wp

        self.layer1 = nn.Sequential(
            nn.Linear(path_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, path_output_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        path_output = self.layer1(path_data)

        return torch.cat((speed, path_output), dim=1)


class FE__UWWE(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, out_dim_per_wp=16):

        total_feature_dim = 1 + LOOKAHEAD_POINTS * out_dim_per_wp

        super().__init__(observation_space, features_dim=total_feature_dim)

        self.layers = nn.ModuleList()
        for _ in range(LOOKAHEAD_POINTS):
            self.layers.append(nn.Sequential(
                nn.Linear(EACH_POINT_INFO_SIZE, out_dim_per_wp),
                nn.ReLU(),
            ))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        outputs = [speed]
        for i in range(LOOKAHEAD_POINTS):
            wp_data = path_data[:, i * EACH_POINT_INFO_SIZE : (i+1) * EACH_POINT_INFO_SIZE]
            wp_output = self.layers[i](wp_data)
            outputs.append(wp_output)

        return torch.cat(outputs, dim=1)


class FE__WSWE(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, out_dim_per_wp=16):

        total_feature_dim = 1 + LOOKAHEAD_POINTS * out_dim_per_wp

        super().__init__(observation_space, features_dim=total_feature_dim)

        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=out_dim_per_wp,
                kernel_size=EACH_POINT_INFO_SIZE,
                stride=EACH_POINT_INFO_SIZE,
                padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        speed     = observations[:, OBSERVATION_IND_SPD      : OBSERVATION_IND_SPD+1]
        path_data = observations[:, OBSERVATION_IND_WPOINT_0 : OBSERVATION_IND_WPOINT_E]

        path_output = self.layer1(path_data.unsqueeze(1))

        return torch.cat((speed, path_output), dim=1)


class MyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):

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


def linear_schedule(start:float, end:float=0.0) -> Callable[[float], float]:
    """
    학습 진행도에 따라 학습률을 선형으로 감소시키는 스케줄러.
    """
    def func(progress_remaining: float) -> float:
        return (progress_remaining * (start - end)) + end
    return func


def create_model(
        policy_kwargs=dict(
            features_extractor_class=FE__I,
            features_extractor_kwargs=dict(),
            net_arch=dict(
                pi=[512, 512], # Actor
                vf=[512, 512, 256]  # Critic
            )
        ),
        save_path:str|None=None,
        *,
        n_steps=512,
        batch_size=256,
        seed:int|None=None,
) -> PPO:
    """
    모델 생성만
    """
    vec_env = make_vec_env(WorldEnv, n_envs=1, vec_env_cls=DummyVecEnv)# n_envs: 병렬 환경 수

    print('POLICY:', policy_kwargs)

    # PPO 모델
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,       # 데이터 수집 스텝 (버퍼 크기, NUM_ENVS * n_steps = 총 수집 데이터량)
        batch_size=batch_size, # 미니 배치 크기
        seed=seed,
    )

    # 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"모델 저장: {CHECKPOINT_DIR+save_path}")

    vec_env.close()  # 환경 정리

    return model


def train(
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
        lr:float|Callable[[float], float]=1e-4,
        gamma=0.99,
        ent_coef=0.0,
        progress_display:Literal['tqdm', 'simple']|None='simple',
        seed:int|None=None,
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
        print(f"체크포인트 로드: {CHECKPOINT_DIR+model_loading_path}")
    elif isinstance(model, PPO):
        model_loading_path = 'temp_model-' + time.strftime('%y%m%d%H%M%S') + str(random.randint(0, 9999))
        is_temp_file = True
        model.save(CHECKPOINT_DIR + model_loading_path)
    else:
        raise ValueError(f'model should be str or PPO not {type(model)}')

    model = PPO.load(
        path=CHECKPOINT_DIR+model_loading_path,
        env=vec_env,
        verbose=0,
        tensorboard_log=LOG_DIR  if tb_log  else None,

        learning_rate=lr,
        gamma=gamma,
        ent_coef=ent_coef,
    )
    assert isinstance(model, PPO)

    # 콜백
    callbacks:list[BaseCallback] = []
    if tb_log:
        callbacks.append(TensorboardCallback())  # 커스텀 매트릭
    if progress_display == 'simple':
        callbacks.append(PercentageProgressCallback(total_timesteps=steps))
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

    print(f"학습 (현재 스텝: {model.num_timesteps} | 목표: {steps + model.num_timesteps} | 남은: {steps})")

    if steps > 0:
        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            tb_log_name=run_name,
            log_interval=10,
            progress_bar=True  if progress_display == 'tqdm'  else False,

            reset_num_timesteps=reset_num_timesteps # 내부 타임스텝 카운터 초기화 여부
        )

    # 최종 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"학습 완료 ({model.num_timesteps} 스텝): {CHECKPOINT_DIR+save_path}")

    # 임시 파일 삭제
    if is_temp_file:
        temp_file_path = CHECKPOINT_DIR + model_loading_path + ".zip"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    vec_env.close()

    return model


def run(
        world_generator:Callable[[], World]|World,
        model:PPO|str,
        time_spd=2.0,
        time_step=WORLD_DT,
        action_repeat=ACTION_REPEAT,
        briefly=True,
        auto_close_at_end=True,
    ):
    """
    모델 시각적 확인용 실행
    """
    if isinstance(world_generator, Callable):
        wgen = world_generator
    elif isinstance(world_generator, World):
        wgen = lambda: world_generator
    else:
        raise ValueError()

    env = WorldEnv(
        world_generator=wgen,
        time_step=time_step,
        action_repeat=action_repeat  if briefly  else 1,
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
        for _ in range(1  if briefly  else action_repeat):
            obs, reward, terminated, truncated, info = env.step(action)  # 행동 실행
            episode_reward += reward
            env.render()  # 시각화 호출
            time.sleep(time_step * action_repeat / 1000.0 / time_spd)# 시각화 프레임을 위해 딜레이 추가
            if terminated or truncated: break

    print(json.dumps(info, indent=4))
    print(f"에피소드 종료. 총 보상: {episode_reward:.2f}")

    env.close()


def evaluate(
        model:PPO|str,
        world_generator:Callable[[], World],
        episode_num=100,
        csv_path:str="",
        *,
        time_step=WORLD_DT,
        action_repeat=ACTION_REPEAT,
        print_result:bool=True,
        verbose:bool=True,
) -> dict:
    import numpy as np
    import pandas as pd

    env = WorldEnv(
        world_generator=world_generator,
        time_step=time_step,
        action_repeat=action_repeat,
        render_mode=None,
    )

    if type(model) == str:
        model = PPO.load(CHECKPOINT_DIR+model, env=env)
    assert isinstance(model, PPO)

    if verbose: print(f"{episode_num}회 에피소드 평가 시작...")

    all_metrics = defaultdict(list)
    episode_rewards = []

    for i in range(episode_num):
        i1 = i+1
        checkPeriod = episode_num // 10
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            # 에피소드 진행
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                episode_rewards.append(total_reward)
                all_metrics['total_reward'].append(total_reward)

                if 'episode_metrics' in info:
                    for key, value in info['episode_metrics'].items():#type:ignore
                        all_metrics[key].append(value)

                # 진행 상황 출력
                if i1 % checkPeriod == 0:
                    if verbose: print(f"[{i1}/{episode_num}] 완료 - Reward: {total_reward:.2f}")

    # 평가 결과 출력 및 저장
    if print_result:
        print("="*41)
        print(f"평가 결과 ({episode_num} 에피소드 평균)")
        print("-"*41)
        print(f"Total Reward  : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)

        # 수: 평균, 표준편차
        num_df = df_metrics.select_dtypes(include=[np.number])
        if not num_df.empty and print_result:
            print("-" * 41)
            summary = num_df.describe().loc[['mean', 'std']].T
            summary['mean'] = summary['mean'].map('{:.4f}'.format)  # 지수표기 안 함.
            print(summary)

        # 문자열(범주형): 종류별 비율
        cat_df = df_metrics.select_dtypes(exclude=[np.number])
        if not cat_df.empty and print_result:
            print("-" * 41)
            for col in cat_df.columns:
                print(f"* {col}")
                counts = df_metrics[col].value_counts()
                ratios = df_metrics[col].value_counts(normalize=True)

                for idx, val in counts.items():
                    ratio = ratios[idx] * 100  #type:ignore
                    print(f"   - {idx:<10}: {val:3d}회 ({ratio:5.1f}%)")
        if print_result:
            print("="*41)

        # CSV 저장
        if csv_path:
            df_metrics.to_csv(csv_path, index=False)
            if verbose: print(f"\n💾 세부 결과 저장됨: {csv_path}")
    else:
        if verbose: print("\n⚠️ info['episode_metrics']가 발견되지 않음.")

    env.close()

    return all_metrics

class PercentageProgressCallback(BaseCallback):
    """실제 누적 타임스텝(num_timesteps)을 기준으로 전체 학습의 10% 단위 진행도를 출력하는 콜백"""
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = 0
        self.next_target_percent = 10  # 최초 목표
        self.initial_num_timesteps = 0 # 이번 학습 세션의 시작 스텝

        # 남은 시간 계산을 위한 직전 마일스톤(10% 단위)의 시간과 스텝 기록
        self.last_milestone_time = 0
        self.last_milestone_steps = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_milestone_time = self.start_time

        # 학습 시작 시점의 모델 누적 스텝을 기준점으로 저장
        self.initial_num_timesteps = self.model.num_timesteps
        self.last_milestone_steps = 0

    def _on_step(self) -> bool:
        # 이번 learn() 호출에서 순수하게 진행된 스텝 계산
        current_progress = self.num_timesteps - self.initial_num_timesteps

        # 목표 스텝 계산
        target_step = int(self.total_timesteps * (self.next_target_percent / 100.0))

        # 순수 진행 스텝이 목표 스텝 이상 도달 시 출력
        if current_progress >= target_step:
            current_time = time.time()
            elapsed = current_time - self.start_time
            remaining_steps = max(0, self.total_timesteps - current_progress)

            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)

            eta_str = ""
            # 100% 이하일 때만 직전 구간의 속도를 기반으로 남은 시간 계산
            if self.next_target_percent < 100:
                delta_time = current_time - self.last_milestone_time
                delta_steps = current_progress - self.last_milestone_steps

                if delta_steps > 0:
                    time_per_step = delta_time / delta_steps
                    eta_seconds = int(time_per_step * remaining_steps)

                    eta_m, eta_s = divmod(eta_seconds, 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f" | 남은 시간: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

            print(f"[진행도 {self.next_target_percent:3d}%] 경과 시간: {h:02d}:{m:02d}:{s:02d}{eta_str} | "
                  f"{current_progress} / {self.total_timesteps} 스텝 | "
                  f"모델 총 누적: {self.num_timesteps}")

            # 다음 구간 계산을 위해 현재 상태를 마일스톤으로 갱신
            self.last_milestone_time = current_time
            self.last_milestone_steps = current_progress
            self.next_target_percent += 10

        return True

    def _on_training_end(self) -> None:
        print(f'모델 총 누적: {self.num_timesteps}')
        self.logger.dump(step=self.num_timesteps)
