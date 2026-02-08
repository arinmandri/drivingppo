"""
PPO 모델 설정값들, 훈련 환경과 훈련 함수들, 각종 변환 함수, 유틸 등.
"""
from typing import Callable, Literal
import math
from datetime import datetime

from .world import World, distance_of, angle_of, pi, pi2, rad_to_deg
from .simsim import WorldViewer
from .common import (
    SPD_MAX_STD,
    LOOKAHEAD_POINTS,
    OBSERVATION_IND_SPD,
    OBSERVATION_IND_WPOINT_0,
    OBSERVATION_IND_WPOINT_1,
    OBSERVATION_IND_WPOINT_2,
    OBSERVATION_DIM,
)

import numpy as np
from numpy import ndarray as Arr
import gymnasium as gym
from gymnasium import spaces


def get_state(world:World):
    """
    World의 현재 상태를 RL 입력 벡터(고정 크기)로 변환
    """
    p = world.player
    s_norm = speed_norm(p.speed)

    # 경로 정보
    path_data = get_path_features(world)

    # 모든 벡터를 합쳐 고정된 크기의 배열로 만든다.
    observation = np.array([s_norm] + path_data, dtype=np.float32)

    return observation

def speed_norm(speed):
    return speed / SPD_MAX_STD

def get_path_features(world:World) -> list[float]:
    """
    경로 정보
    바로 앞의 점 몇 개의 거리와 각도.
    """

    path_data = []
    x0 = world.player.x
    z0 = world.player.z
    a0 = world.player.angle_x

    # 각 목표점의 거리, 각도 정보
    for index in range(
            world.waypoint_idx,
            world.waypoint_idx + LOOKAHEAD_POINTS
        ):
        # 이전 목표점 기준
        if index < world.path_len:
            x1, z1 = world.waypoints[index]
            d_from_prev = distance_of(x0, z0, x1, z1)
            a1          = angle_of(x0, z0, x1, z1)
            a_from_prev = a1 - a0
            x0 = x1
            z0 = z1
            a0 = a1
        else:
            d_from_prev = 0.0
            a_from_prev = 0.0

        a_fp_norm = ((a_from_prev + pi) % pi2 - pi) / pi  # 각도(이전 목표점 기준)
        d_near = distance_score_near(d_from_prev)  # 거리 가까운 정도
        d_far  = distance_score_far(d_from_prev)   # 거리 먼 정도

        path_data.extend([a_fp_norm, math.cos(a_fp_norm), d_near, d_far])

    # # 각 목표점의 거리, 각도 정보
    # for _ in range(LOOKAHEAD_POINTS):

    #     # 에이전트 기준
    #     d_from_agnt = world.get_distance_to_wpoint()
    #     a_from_agnt = world.get_relative_angle_to_wpoint()

    #     a_fp_norm = ((a_from_agnt + pi) % pi2 - pi) / pi  # 각도(이전 목표점 기준)
    #     d_near = distance_score_near(d_from_agnt)  # 거리 가까운 정도
    #     d_far  = distance_score_far(d_from_agnt)   # 거리 먼 정도

    #     path_data.extend([a_fp_norm, math.cos(a_fp_norm), d_near, d_far])

    return path_data

def observation_str(observation):
    agent_speed       = observation[OBSERVATION_IND_SPD]
    obs_wpoint_afp_0  = observation[OBSERVATION_IND_WPOINT_0]
    obs_wpoint_dist_0 = observation[OBSERVATION_IND_WPOINT_0 +3]
    obs_wpoint_afp_1  = observation[OBSERVATION_IND_WPOINT_1]
    obs_wpoint_dist_1 = observation[OBSERVATION_IND_WPOINT_1 +3]
    obs_wpoint_afp_2  = observation[OBSERVATION_IND_WPOINT_2]
    obs_wpoint_dist_2 = observation[OBSERVATION_IND_WPOINT_2 +3]
    return f'STATE:  speed {agent_speed:+.2f}({speed_norm(agent_speed):+.2f})'\
           f' | Path'\
           f' [0] a:{obs_wpoint_afp_0*pi*rad_to_deg:+5.2f} d:{obs_wpoint_dist_0:.2f}'\
           f' [1] a:{obs_wpoint_afp_1*pi*rad_to_deg:+5.2f} d:{obs_wpoint_dist_1:.2f}'\
           f' [2] a:{obs_wpoint_afp_2*pi*rad_to_deg:+5.2f} d:{obs_wpoint_dist_2:.2f}'

def _distance_score_near(x:float) -> float:
    d = x + 10.0
    x = 100./d/d
    if x <= 1:
        return x
    else:
        return 1.0

def distance_score_near(x:float) -> float:
    return _distance_score_near(x)

def distance_score_far(x:float) -> float:
    return x / 30.0


def apply_action(world:World, action:Arr):
    """
    행동 벡터 [A_forward, A_steer]를 World의 제어 함수로 변환하여 적용
    """
    ws, ad = action
    world.set_action(ws, ad, False)

def action_str(action):
    return f'ACTION: {action[0]:.2f}  {action[1]:.2f}'



class WorldEnv(gym.Env):
    """
    World에서 주행법을 강화학습하기 위한 gym 환경 클래스.
    """

    def __init__(self,
                 world_generator:Callable[[], World],
                 max_time=120_000,
                 time_step=111,
                 wstep_per_control=3,
                 time_gain_per_waypoint_rate=500,
                 time_gain_limit=20_000,
                 render_mode:Literal['window','debug']|None=None,
                 auto_close_at_end=True):

        super().__init__()
        self.closed = False

        self.time_step = time_step  # 월드의 1스텝당 흐르는 시간(천분초)
        self.wstep_per_control = wstep_per_control  # 조작값 변경은 월드의 n스텝마다 한 번. Tank Challenge에서도 FPS는 30이어도 API 요청은 최소 0.1초마다 한 번으로 설정 가능하다.
        self.max_time = max_time  # 최대 에피소드 길이(천분초)
        self.time_gain_per_waypoint_rate = time_gain_per_waypoint_rate  # 다음 목표점까지 거리 1당 획득 시간(천분초)
        self.time_gain_limit = time_gain_limit  # 남은 제한시간 최대량(천분초)

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

        self.action_history.append(action)  # 매스텝 액션 기록
        ws, ad = action

        w = self.world
        p = w.player

        ang_pv  = w.get_relative_angle_to_wpoint()
        cos_pv  = math.cos(ang_pv)

        # 액션 적용
        apply_action(self.world, action)
        result_collision = False
        result_wpoint = False
        for _ in range(self.wstep_per_control):
            _, result_collision_step, result_wpoint_step = w.step(self.time_step)
            result_collision += result_collision_step
            result_wpoint      += result_wpoint_step
        if self.estep_count == 1:
            if result_collision: print(f'💥💥💥💥💥💥💥💥💥 맵 확인 필요: 시작과동시에 충돌 (hint: 목표점 수 {w.path_len})')
            if result_wpoint:    print(f'💥💥💥💥💥💥💥💥💥 맵 확인 필요: 시작과동시에 골 (hint: 목표점 수 {w.path_len})')

        observation1 = self.observation

        if self.render_mode == 'debug': print(action_str(action))

        terminated = False
        truncated = False
        ending = ''

        s_norm = speed_norm(p.speed)  # 속도점수
        distance = w.get_distance_to_wpoint()
        ang_nx  = w.get_relative_angle_to_wpoint()
        cos_nx  = math.cos(ang_nx)

        self.speed_history.append(p.speed)

        reward_step = [0.0 for _ in range(7)]

        # 목표점 도달
        if result_wpoint:
            if p.speed > 0:  # 후진 진행 억제
                reward_step[1] += 50.0

            # 추가시간 획득; 그러나 무한정 쌓이지는 않음.
            self.time_limit += int(distance * self.time_gain_per_waypoint_rate)
            self.time_limit = min(self.time_limit, w.t_acc + self.time_gain_limit)

            self.prev_d = self.prev_d1

            if self.render_mode == 'debug': print(f'★[{w.waypoint_idx}] {reward_step[1]:.1f} ~ pass {int(round(ang_pv*rad_to_deg))}({cos_pv:.2f})')

            # 최종 목표 도달
            if w.arrived:
                ending = 'arrived'
                terminated = True

        # 전혀 엉뚱한 곳 감
        elif distance > w.far:
            reward_step[2] += 100.0 * p.speed / SPD_MAX_STD * cos_nx
            if self.render_mode == 'debug': print(f'LOST ({distance:.1f} > {w.far:.1f}) reward: {reward_step[2]:.2f}')
            ending = 'lost'
            truncated = True

        # 시간 내에 도착 못 함
        elif w.t_acc >= self.time_limit:
            reward_step[2] += -150.0
            ending = 'timeover'
            truncated = True

        # 획득한 시간은 모자르지 않으나 그냥 이제까지 많이 함.
        elif w.t_acc >= self.max_time:
            ending = 'timeout'
            truncated = True

        if truncated or terminated:
            icon = \
                '✅' if ending == 'arrived' else \
                '▶️' if ending == 'timeout' else \
                '👻' if ending == 'lost' else \
                '⏰' if ending == 'timeover' else '??'
            self.print_log(f'결과{icon} 도착: {w.waypoint_idx:3d}/{w.path_len:3d} | 시간: {int(w.t_acc/1000):3d}/{int(self.time_limit/1000):3d}/{int(self.max_time/1000):3d} 초 ({int(w.t_acc/self.max_time*100):3d}%) | 위치: {int(p.x):4d}, {int(p.z):4d} ({int(p.x/self.world.MAP_W*100):3d}%, {int(p.z/self.world.MAP_H*100):3d}%)')

        else:
            # 진행 보상

            reward_time = -0.1

            distance_d = distance - self.prev_d
            stat_progress     = - distance_d * 0.15  if s_norm > 0 \
                           else - self.wstep_per_control * s_norm * s_norm * 1.5  # 후진 진행 억제
            reward_action_ws  = - ws**2 * 0.7
            reward_action_ad  = - ad**2 * 0.9
            total = reward_time + stat_progress + reward_action_ws + reward_action_ad
            if self.render_mode == 'debug': print(f'REWARD: time {reward_time:+5.2f} |  prog {stat_progress:+5.2f} | ws {reward_action_ws:+4.2f} | ad {reward_action_ad:+4.2f} --> {total:+6.2f}')

            reward_step[2] += self.wstep_per_control * reward_time
            reward_step[3] += stat_progress
            reward_step[5] += self.wstep_per_control * reward_action_ws
            reward_step[6] += self.wstep_per_control * reward_action_ad

        info = {'current_time': w.t_acc / 1000.0}

        # 점수 합
        reward_step[0] = sum(reward_step[1:])
        for i in range(7):
            self.reward_totals[i] += reward_step[i]

        if truncated or terminated:

            # 액션 분산
            if len(self.action_history) > 0:
                action_arr = np.array(self.action_history)
                ws_var = np.var(action_arr[:, 0])
                ad_var = np.var(action_arr[:, 1])
            else:
                ws_var, ad_var = 0.0, 0.0

            if len(self.speed_history) > 0:
                speed_arr = np.array(self.speed_history)
                speed_var = np.var(speed_arr)
                speed_mean = sum(self.speed_history) / len(self.speed_history)
            else:
                speed_var = 0.0
                speed_mean = 0.0

            wstep_count = self.estep_count * self.wstep_per_control

            info['episode_metrics'] = {
                'ending/type': ending,
                'ending/estep': self.estep_count,
                'ending/wstep': self.estep_count * self.wstep_per_control,
                'rewards/0.total':       self.reward_totals[0]/wstep_count,
                'rewards/1.wPoint':      self.reward_totals[1]/wstep_count,
                'rewards/2.time':        self.reward_totals[2]/wstep_count,
                'rewards/3.progress':    self.reward_totals[3]/wstep_count,
                'rewards/5.ws':          self.reward_totals[5]/wstep_count,
                'rewards/6.ad':          self.reward_totals[6]/wstep_count,
                'metrics/ws_var':        ws_var,
                'metrics/ad_var':        ad_var,
                'metrics/speed_mean':    speed_mean,
                'metrics/speed_var':     speed_var,
            }

            self.print_result()

        self.prev_d = distance
        self.prev_d1 = w.get_distance_to_wpoint(1)

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
        self.reward_totals = [0.0 for _ in range(7)]
        self.time_limit = self.time_gain_limit  # 제한시간. 목표점 도달시마다 추가 획득.
        self.action_history = []  # 액션 기록
        self.speed_history = []

        self.prev_d  = w.get_distance_to_wpoint()
        self.prev_d1 = w.get_distance_to_wpoint(1)

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
        wstep_count = self.estep_count * self.wstep_per_control
        if wstep_count:
            self.print_log(f'총점 {int(self.reward_totals[0]):5d} '
                           f'| wpoint {self.reward_totals[1]:6.1f}({ int(self.reward_totals[1]/wstep_count*100)}%) '
                           f'| time {  self.reward_totals[2]:+7.2f}({int(self.reward_totals[2]/wstep_count*100)}%) '
                           f'| prog {  self.reward_totals[3]:+7.2f}({int(self.reward_totals[3]/wstep_count*100)}%) '
                           f'| ws {    self.reward_totals[5]:+7.2f}({int(self.reward_totals[5]/wstep_count*100)}%) '
                           f'| ad {    self.reward_totals[6]:+7.2f}({int(self.reward_totals[6]/wstep_count*100)}%)')

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
