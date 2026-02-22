"""
PPO 모델 설정값들, 훈련 환경과 훈련 함수들, 각종 변환 함수, 유틸 등.
"""
from typing import Callable, Literal
import math
from datetime import datetime

from .world import World, distance_of, angle_of, pi, pi2, rad_to_deg
from .simsim import WorldViewer
from .common import (
    SPD_SCFAC,
    DIS_SCFAC,
    WORLD_DT,
    ACTION_REPEAT,
    LOOKAHEAD_POINTS,
    OBSERVATION_IND_SPD,
    OBSERVATION_IND_WPOINT_0,
    OBSERVATION_IND_WPOINT_1,
    OBSERVATION_IND_WPOINT_2,
    LIDAR_START,
    LIDAR_END,
    LIDAR_NUM,
    LIDAR_RANGE,
    OBSERVATION_IND_LIDAR_S,
    OBSERVATION_IND_LIDAR_E,
    OBSERVATION_DIM,
)
from .env_utils import (
    speed_norm,
    distance_score_near,
    distance_score_far,
    get_path_features_srel as get_path_features,
    # get_path_features_fagnt as get_path_features,
    observation_str,
    action_str,
    MyMetrics,
)

import numpy as np
from numpy import ndarray as Arr
import gymnasium as gym
from gymnasium import spaces

REWARD_METRIC_SIZE = 10



def get_state(world:World):
    """
    World의 현재 상태를 RL 입력 벡터(고정 크기)로 변환
    """
    p = world.player
    s_norm = speed_norm(p.speed)

    # 경로 정보
    path_data = get_path_features(world)

    # 라이다
    lidar_data = [distance_score_near(distance) if h else 0.0
                  for _,_, distance, _,_,_, h in world.lidar_points]

    # 모든 벡터를 합쳐 고정된 크기의 배열로 만든다.
    observation = np.array([s_norm] + path_data + lidar_data, dtype=np.float32)

    return observation


def apply_action(world:World, action:Arr):
    ws, ad = action
    ws = float(ws)
    ad = float(ad)
    world.set_action(ws, ad, False)


class WorldEnv(gym.Env):
    """
    World에서 주행법을 강화학습하기 위한 gym 환경 클래스.
    """

    def __init__(self,
                 world_generator:Callable[[], World]=World,
                 max_time=120_000,
                 time_step=WORLD_DT,
                 action_repeat=ACTION_REPEAT,
                 time_gain_per_waypoint_rate=500,
                 time_gain_limit=20_000,
                 collision_ending=True,
                 render_mode:Literal['window','debug']|None=None,
                 auto_close_at_end=True):

        super().__init__()
        self.closed = False

        self.estep_count = 0
        self.time_step = time_step  # 월드의 1스텝당 흐르는 시간(천분초)
        self.action_repeat = action_repeat  # 조작값 변경은 월드의 n스텝마다 한 번. Tank Challenge에서도 FPS는 30이어도 API 요청은 최소 0.1초마다 한 번으로 설정 가능하다.
        self.max_time = max_time  # 최대 에피소드 길이(천분초)
        self.time_gain_per_waypoint_rate = time_gain_per_waypoint_rate  # 다음 목표점까지 거리 1당 획득 시간(천분초)
        self.time_gain_limit = time_gain_limit  # 남은 제한시간 최대량(천분초)

        self.collision_ending = collision_ending

        # Action: [A_forward, A_steer]
        self.action_space = spaces.Box(  # Forward, Steer
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space 정의 (고정된 크기의 실수 벡터)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32
        )

        self.world_generator = world_generator

        """
        render_mode
        None: 조용히
        'window': 창 띄움
        'debug': 창 + 터미널에 텍스트
        """
        self.render_mode = render_mode
        self.auto_close_at_end = auto_close_at_end
        self.viewer:WorldViewer|None = None
        print(f'WorldEnv render:{self.render_mode}')


    @property
    def observation(self):
        return get_state(self.world)

    @property
    def time_remaining(self):
        return self.time_limit - self.world.t_acc

    def step(self, action):
        """
        행동을 실행하고, 다음 상태, 보상, 종료 여부를 반환
        """
        observation0 = self.observation

        if self.closed:  # 창 닫아서 종료
            return observation0, 0, False, True, {}

        self.estep_count += 1
        if self.render_mode == 'debug':
            print(f'{self.estep_count} step -------------------------- 남은시간 {int((self.time_remaining)/1000)}')
            print(observation_str(observation0))

        ws, ad = action
        ws = float(ws)
        ad = float(ad)

        w = self.world
        p = w.player

        ang_pv  = w.get_relative_angle_to_wpoint()
        cos_pv  = math.cos(ang_pv)
        dis_pv  = w.get_distance_to_wpoint()
        dis_pv1 = w.get_distance_to_wpoint(1)

        # 액션 적용
        apply_action(self.world, action)
        result_collision = False
        result_wpoint = False
        wstep_count_step = 0  # 이번 환경스텝에서 몇 월드스텝 진행? 평소에는 action_repeat만큼인데 도착시에는 잘릴 수 있음.
        for _ in range(self.action_repeat):
            wstep_count_step += 1
            _, result_collision_step, result_wpoint_step = w.step(self.time_step)
            result_collision = True  if result_collision_step  else result_collision
            result_wpoint    = True  if result_wpoint_step     else result_wpoint
            if w.arrived  or result_collision_step: break
        # if self.estep_count == 1:
        #     if result_collision: print(f'💥💥💥💥💥💥💥💥💥 맵 확인 필요: 시작과동시에 충돌 (hint: 목표점 수 {w.path_len})')
        #     if result_wpoint:    print(f'💥💥💥💥💥💥💥💥💥 맵 확인 필요: 시작과동시에 골 (hint: 목표점 수 {w.path_len})')

        self.wstep_count += wstep_count_step
        tfac = wstep_count_step * self.time_step / 1000.0  # 이번 환경스텝에서 흐른 시간 (초)

        info = {'current_time': w.t_acc / 1000.0}

        observation1 = self.observation

        if self.render_mode == 'debug': print(action_str(action))

        terminated = False
        truncated = False
        ending = ''

        s_norm = speed_norm(p.speed)  # 속도점수
        distance = w.get_distance_to_wpoint()
        ang_nx  = w.get_relative_angle_to_wpoint()
        cos_nx  = math.cos(ang_nx)
        dis_nx  = w.get_distance_to_wpoint()

        ld_max_0 = observation0[OBSERVATION_IND_LIDAR_S:OBSERVATION_IND_LIDAR_E].max()  if LIDAR_NUM  else 0.0
        ld_max_1 = observation1[OBSERVATION_IND_LIDAR_S:OBSERVATION_IND_LIDAR_E].max()  if LIDAR_NUM  else 0.0
        ld_max_d = ld_max_1 - ld_max_0

        self.metrics.step(action, w)

        reward_step = [0.0 for _ in range(REWARD_METRIC_SIZE)]

        if p.speed < 0:
            # 후진진행 억제
            self.time_limit += int(s_norm*1000)

        # 충돌
        if result_collision  and self.collision_ending:
            reward_step[2] += -150.0
            ending = 'collision'
            terminated = True

        # 시간 내에 도착 못 함
        elif w.t_acc > self.time_limit  and self.collision_ending:
            reward_step[2] += -150.0
            ending = 'timeover'
            truncated = True

        # 목표점 도달
        elif result_wpoint:
            if p.speed > 0:  # 후진 진행 억제
                reward_step[1] += 50.0
                # 추가시간 획득; 그러나 무한정 쌓이지는 않음.
                self.time_limit += int(distance * self.time_gain_per_waypoint_rate)
                self.time_limit = min(self.time_limit, w.t_acc + self.time_gain_limit)

            dis_pv = dis_pv1

            if self.render_mode == 'debug': print(f'★[{w.waypoint_idx}] {reward_step[1]:.1f} ~ pass {int(round(ang_nx*rad_to_deg))}({cos_nx:.2f})')

            # 최종 목표 도달
            if w.arrived:
                ending = 'arrived'
                if sum(self.metrics.speed_history) <= 0.0:  # 후진진행한 게 틀림없다.
                    reward_step[1] = - 300.0
                terminated = True

        # 전혀 엉뚱한 곳 감
        elif distance > w.far:
            reward_step[2] += 100.0 * s_norm * cos_nx
            if self.render_mode == 'debug': print(f'LOST ({distance:.1f} > {w.far:.1f}) reward: {reward_step[2]:.2f}')
            ending = 'lost'
            truncated = True

        # 획득한 시간은 모자르지 않으나 그냥 이제까지 많이 함.
        elif w.t_acc >= self.max_time:
            ending = 'timeout'
            truncated = True

        # 밀집보상
        reward_time = -5.0

        distance_d = dis_nx - dis_pv
        reward_progress    = - distance_d * 0.07
        if s_norm < 0: reward_progress = min(0.0, reward_progress)
        reward_orientation = cos_nx * 0.2
        reward_action_ws   = 0.0#- ws * s_norm * 4.0  if ws * s_norm > 0  else 0.0  # 브레이크 사용시 비용 없다 침.
        reward_action_ad   = 0.0#- ad * ad * 1.7
        danger             = - ld_max_1 * 0.6
        danger_d           = - ld_max_d * 80.0
        total = reward_time + reward_progress + reward_action_ws + reward_action_ad + danger + danger_d
        if self.render_mode == 'debug': print(f'REWARD: time {reward_time:+5.2f} |  prog {reward_progress/tfac:+5.2f} | ort {reward_orientation:+4.2f} | ws {reward_action_ws:+4.2f} | ad {reward_action_ad:+4.2f} | danger {danger:+5.2f}~{danger_d:+5.2f} --> {total:+6.2f}')

        reward_step[3] += tfac * reward_time
        reward_step[4] += reward_progress
        reward_step[5] += tfac * reward_orientation
        reward_step[6] += tfac * reward_action_ws
        reward_step[7] += tfac * reward_action_ad
        reward_step[8] += tfac * danger
        reward_step[9] += tfac * danger_d


        # 점수 합
        reward_step[0] = sum(reward_step[1:])
        for i in range(REWARD_METRIC_SIZE):
            self.reward_totals[i] += reward_step[i]


        if truncated or terminated:
            icon = \
                '✅' if ending == 'arrived' else \
                '▶️' if ending == 'timeout' else \
                '👻' if ending == 'lost' else \
                '💥' if ending == 'collision' else \
                '⏰' if ending == 'timeover' else '??'
            self.print_log(f'결과{icon} 도착: {w.waypoint_idx:3d}/{w.path_len:3d} | 시간: {int(w.t_acc/1000):3d}/{int(self.time_limit/1000):3d}/{int(self.max_time/1000):3d} 초 ({int(w.t_acc/self.max_time*100):3d}%) | 위치: {int(p.x):4d}, {int(p.z):4d} ({int(p.x/self.world.MAP_W*100):3d}%, {int(p.z/self.world.MAP_H*100):3d}%)')

            tcount = self.wstep_count * self.time_step / 1000.0  # 흐른 시간 (초)

            info['episode_metrics'] = {
                'ending/achvRate': w.waypoint_idx / w.path_len,
                'ending/type':     ending,
                'ending/estep':    self.estep_count,
                'ending/wstep':    self.wstep_count,
                'ending/sec':      tcount,
                # 'rewards/0.total':       self.reward_totals[0]/tcount,
                # 'rewards/1.wPoint':      self.reward_totals[1]/tcount,
                # 'rewards/2.fail':        self.reward_totals[2]/tcount,
                # 'rewards/3.time':        self.reward_totals[3]/tcount,
                # 'rewards/4.progress':    self.reward_totals[4]/tcount,
                # 'rewards/5.orientat':    self.reward_totals[5]/tcount,
                # 'rewards/6.ws':          self.reward_totals[6]/tcount,
                # 'rewards/7.ad':          self.reward_totals[7]/tcount,
                # 'rewards/8.danger':      self.reward_totals[8]/tcount,
                # 'rewards/9.danger_d':    self.reward_totals[9]/tcount,
            } | self.metrics.export()

            self.print_result()

        # Gymnasium 표준 반환
        return observation1, reward_step[0], terminated, truncated, info


    def reset(self, *, seed=None, options=None):
        """
        환경을 초기화하고 초기 상태를 반환
        """
        super().reset(seed=seed)

        w = self.world_generator()
        self.world = w

        self.estep_count = 0
        self.wstep_count = 0
        self.reward_totals = [0.0 for _ in range(REWARD_METRIC_SIZE)]
        self.time_limit = self.time_gain_limit  # 제한시간. 목표점 도달시마다 추가 획득.
        self.metrics = MyMetrics(w)

        observation = self.observation
        info = {}
        return observation, info


    def render(self):
        if self.render_mode == None: return
        if self.closed:
            self.close()
            return

        # 지연 초기화: WorldViewer가 아직 생성되지 않았다면 생성합니다.
        if self.viewer is None:
            self.viewer = WorldViewer(self.world, auto_update=False)
        elif self.viewer.world is not self.world:
            self.viewer.close()
            self.viewer = WorldViewer(self.world, auto_update=False)

        if self.viewer.closed: self.closed = True; return

        self.viewer.update()

    def print_result(self):
        wstep_count = self.estep_count * self.action_repeat
        if wstep_count:
            self.print_log(f'총점 {int(self.reward_totals[0]):5d} '
                           f'| wpoint {  self.reward_totals[1]:6.1f}({ int(self.reward_totals[1]/wstep_count*100)}%) '
                           f'| fail {    self.reward_totals[2]:6.1f}({ int(self.reward_totals[2]/wstep_count*100)}%) '
                           f'| time {    self.reward_totals[3]:+7.2f}({int(self.reward_totals[3]/wstep_count*100)}%) '
                           f'| prog {    self.reward_totals[4]:+7.2f}({int(self.reward_totals[4]/wstep_count*100)}%) '
                           f'| ort {     self.reward_totals[5]:+7.2f}({int(self.reward_totals[5]/wstep_count*100)}%) '
                           f'| ws {      self.reward_totals[6]:+7.2f}({int(self.reward_totals[6]/wstep_count*100)}%) '
                           f'| ad {      self.reward_totals[7]:+7.2f}({int(self.reward_totals[7]/wstep_count*100)}%) '
                           f'| danger {  self.reward_totals[8]:+7.2f}({int(self.reward_totals[8]/wstep_count*100)}%) '
                           f'| danger_d {self.reward_totals[9]:+7.2f}({int(self.reward_totals[9]/wstep_count*100)}%)')

    def print_log(
            self,
            message: str,
    ):
        current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_message = f"{current_time} {message}"

        if self.render_mode == 'debug':
            print(formatted_message, flush=True)


    def close(self):
        self.print_result()
        self.closed = True
        if self.viewer is None: return
        if self.auto_close_at_end:
            self.viewer.close()
            self.viewer = None
        else:
            self.viewer.occupy_mainloop()
            self.viewer = None
        print('WorldEnv closed')
