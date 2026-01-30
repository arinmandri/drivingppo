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
    LIDAR_START,
    LIDAR_END,
    LIDAR_NUM,
    LIDAR_RANGE,
    LOOKAHEAD_POINTS,
    OBSERVATION_IND_SPD,
    OBSERVATION_IND_WPOINT_0,
    OBSERVATION_IND_WPOINT_1,
    OBSERVATION_IND_WPOINT_2,
    OBSERVATION_IND_LIDAR_DIS_S,
    OBSERVATION_IND_LIDAR_DIS_E,
    OBSERVATION_DIM,
)

import numpy as np
from numpy import ndarray as Arr
import gymnasium as gym
from gymnasium import spaces

LOG_FILE_PATH="training_log.txt"


def get_state(world:World):
    """
    World의 현재 상태를 RL 입력 벡터(고정 크기)로 변환
    """
    p = world.player
    s_norm = speed_norm(p.speed)

    # 경로 정보
    path_data = get_path_features(world)

    # 라이다 거리가까운점수
    obs_near = [distance_score_near(distance) if h else 0.0
                for _,_, distance, _,_,_, h in world.lidar_points]

    # 모든 벡터를 합쳐 고정된 크기의 배열로 만든다.
    observation = np.array([s_norm] + path_data + obs_near, dtype=np.float32)

    return observation

def speed_norm(speed):
    return min(speed / SPD_MAX_STD, 1.0)  # 가능한 최대 속력은 19쯤이지만 실제로 7이 넘어가는 경우가 거의 없어서 최대 속력 10으로 치고 정규화.

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

        # 에이전트 기준
        # d_from_agnt = world.get_distance_to_wpoint(index)
        # a_from_agnt = world.get_relative_angle_to_wpoint(index)

        a_fp_norm = ((a_from_prev + pi) % pi2 - pi) / pi  # 각도(이전 목표점 기준)
        d_near = distance_score_near(d_from_prev)  # 거리 가까운 정도
        d_far  = distance_score_far(d_from_prev)   # 거리 먼 정도

        path_data.extend([a_fp_norm, math.cos(a_fp_norm), d_near, d_far])

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

distance_score_near_base = _distance_score_near(LIDAR_RANGE)

def distance_score_near(x:float) -> float:
    return max(0, _distance_score_near(x) - distance_score_near_base)

def distance_score_far(distance:float) -> float:
    return math.log(distance + 1.0)/10.0


def apply_action(world:World, action:Arr):
    """
    행동 벡터 [A_forward, A_steer]를 World의 제어 함수로 변환하여 적용
    """

    A_forward, A_steer = action

    # WS
    if A_forward > 0:
        world.moveWS('W', A_forward)
    else:
        world.moveWS('S', -A_forward)

    # AD
    if A_steer > 0: # 양수: 우회전 (D)
        world.moveAD('D', A_steer)
    else: # 음수: 좌회전 (A)
        world.moveAD('A', -A_steer)

def action_str(action):
    return f'ACTION: {action[0]:.2f}  {action[1]:.2f}'



class WorldEnv(gym.Env):
    """
    World에서 주행법을 강화학습하기 위한 gym 환경 클래스.
    """

    time_gain_per_waypoint = 10_000
    time_gain_limit = 20_000

    def __init__(self,
                 world_generator:Callable[[], World],
                 max_time=120_000,
                 time_step=111,
                 step_per_control=3,
                 render_mode:Literal['window','debug']|None=None,
                 auto_close_at_end=True):

        super().__init__()
        self.closed = False

        self.lidar_angles = np.linspace(LIDAR_START, LIDAR_END, LIDAR_NUM)

        self.time_step = time_step
        self.step_per_control = step_per_control  # 조작값 변경은 월드의 n스텝마다 한 번. Tank Challenge에서도 FPS는 30이어도 API 요청은 최소 0.1초마다 한 번으로 설정 가능하다.
        self.max_time = max_time  # 최대 에피소드 타임

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


    def step(self, action):
        """
        행동을 실행하고, 다음 상태, 보상, 종료 여부를 반환
        """
        observation0 = self.observation

        if self.closed:  # 창 닫아서 종료
            return observation0, 0, False, True, {}

        self.step_count += 1
        if self.render_mode == 'debug': print(f'{self.step_count} step --------------------------')
        if self.render_mode == 'debug': print(observation_str(observation0))

        w = self.world
        p = w.player

        ang_pv  = w.get_relative_angle_to_wpoint()
        cos_pv  = math.cos(ang_pv)

        # 액션 적용
        apply_action(self.world, action)
        result_collision = False
        result_wpoint = False
        for _ in range(self.step_per_control):
            _, result_collision_step, result_wpoint_step = w.step(self.time_step)
            result_collision += result_collision_step
            result_wpoint      += result_wpoint_step
        if self.step_count == 1:
            if result_collision: print(f'💥💥💥💥💥💥💥💥💥 맵 확인 필요: 시작과동시에 충돌 (hint: 목표점 수 {w.path_len})')
            if result_wpoint:      print(f'💥💥💥💥💥💥💥💥💥 맵 확인 필요: 시작과동시에 골 (hint: 목표점 수 {w.path_len})')

        observation1 = self.observation

        if self.render_mode == 'debug': print(action_str(action))

        terminated = False
        truncated = False
        ending = ''

        s_norm = speed_norm(p.speed)  # 속도점수
        distance = w.get_distance_to_wpoint() +1
        ang_nx  = w.get_relative_angle_to_wpoint()
        cos_nx  = math.cos(ang_nx)

        obs0 = observation0[OBSERVATION_IND_LIDAR_DIS_S:OBSERVATION_IND_LIDAR_DIS_E].max()
        obs1 = observation1[OBSERVATION_IND_LIDAR_DIS_S:OBSERVATION_IND_LIDAR_DIS_E].max()
        obs_d = obs1 - obs0

        reward_step = [0.0 for _ in range(7)]

        # 충돌
        if result_collision:
            reward_step[2] += -200.0
            ending = '충돌'
            terminated = True

        # 목표점 도달
        elif result_wpoint:
            reward_step[1] += (18.0 * cos_pv) + (12.0 * cos_nx) + (7.5 * s_norm)
            if self.render_mode == 'debug': print(f'★[{w.waypoint_idx}] {reward_step[1]:.1f} ~ pass {int(round(ang_pv*rad_to_deg))}({cos_pv:.2f}) | next_a {int(round(ang_nx*rad_to_deg))}({cos_nx:.2f})')

            # 추가시간 획득; 그러나 무한정 쌓이지는 않음.
            self.time_limit += self.time_gain_per_waypoint
            self.time_limit = min(self.time_limit, w.t_acc + self.time_gain_limit)

            # 최종 목표 도달
            if w.arrived:
                ending = '도착'
                terminated = True

        # 전혀 엉뚱한 곳 감
        elif distance > w.far:
            reward_step[2] += 100.0 * p.speed / SPD_MAX_STD * cos_nx
            if self.render_mode == 'debug': print(f'LOST ({distance:.1f} > {w.far:.1f}) reward: {reward_step[2]:.2f}')
            ending = '길잃음'
            truncated = True

        # 시간 내에 도착 못 함
        elif w.t_acc >= self.time_limit:
            reward_step[2] += -200.0  # 목적지가 코앞인데 벽앞에서 가만히있기를 택하지 않도록 충돌만큼의 벌점. 대신 시간은 넉넉히 줌.
            ending = '시간초과'
            truncated = True

        # 획득한 시간은 모자르지 않으나 그냥 이제까지 많이 함.
        elif w.t_acc >= self.max_time:
            ending = '시간한계'
            truncated = True

        if truncated or terminated:
            icon = \
                '✅' if ending == '도착' else \
                '▶️' if ending == '시간한계' else \
                '💥' if ending == '충돌' else \
                '👻' if ending == '길잃음' else \
                '⏰' if ending == '시간초과' else '??'
            self.print_log(f'결과{icon} 도착: {w.waypoint_idx:3d}/{w.path_len:3d} | 시간: {int(w.t_acc/1000):3d}/{int(self.time_limit/1000):3d}/{int(self.max_time/1000):3d} 초 ({int(w.t_acc/self.max_time*100):3d}%) | 위치: {int(p.x):4d}, {int(p.z):4d} ({int(p.x/self.world.MAP_W*100):3d}%, {int(p.z/self.world.MAP_H*100):3d}%)')

        else:
            # 진행 보상

            reward_time = -0.1

            stat_progress     = + (cos_nx * s_norm) * 0.3  if s_norm > 0 \
                           else - s_norm * s_norm * 1.5  # 후진 진행 억제
            stat_orientation  = + cos_nx * 0.06
            danger            = - obs1 * 0.15
            danger_d          = - obs_d * 8.0
            total = reward_time+stat_progress+stat_orientation+danger+danger_d
            if self.render_mode == 'debug': print(f'REWARD: time {reward_time:+5.2f} |  prog {stat_progress:+5.2f} | ang {stat_orientation:+5.2f} | danger {danger:+5.2f} ~  {danger_d:+5.2f} --> {total:+6.2f}')

            reward_step[2] += self.step_per_control * reward_time
            reward_step[3] += self.step_per_control * stat_progress
            reward_step[4] += self.step_per_control * stat_orientation
            reward_step[5] += self.step_per_control * danger
            reward_step[6] += self.step_per_control * danger_d

        info = {'current_time': w.t_acc / 1000.0}

        # 점수 합
        reward_step[0] = sum(reward_step[1:])
        for i in range(7):
            self.reward_totals[i] += reward_step[i]
        if truncated or terminated:
            self.print_result()

        # Gymnasium 표준 반환
        return observation1, reward_step[0], terminated, truncated, info


    def reset(self, *, seed=None, options=None):
        """
        환경을 초기화하고 초기 상태를 반환
        """
        super().reset(seed=seed)

        self.reset_randomly()

        self.step_count = 0
        self.reward_totals = [0.0 for _ in range(7)]
        self.time_limit = self.time_gain_limit  # 제한시간. 목표점 도달시마다 추가 획득.

        w = self.world
        p = w.player
        self.S_MAX = p.speed_max_w(1)  # 최대속도

        observation = self.observation
        info = {}
        return observation, info

    def reset_randomly(self):
        self.world = self.world_generator()


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
        self.print_log(f'총점 {int(self.reward_totals[0]):5d} | wpoint {self.reward_totals[1]:6.1f} | time {self.reward_totals[2]:+7.2f} | prog {self.reward_totals[3]:+7.2f} | ang {self.reward_totals[4]:+7.2f} | danger {self.reward_totals[5]:+7.2f} ~ {self.reward_totals[6]:+7.2f}')

    def print_log(
            self,
            message: str,
    ):
        current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_message = f"{current_time} {message}"

        if self.render_mode == 'debug':
            print(formatted_message, flush=True)

        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(formatted_message + "\n")
        except Exception as e:
            print(f"!!! 로그 저장 실패: {e}")


    def close(self):
        self.print_result()
        self.closed = True
        if self.viewer is None: return
        if self.auto_close_at_end:
            self.viewer.close()
            self.viewer = None
        else:
            self.viewer.mainloop()
            self.viewer = None
        print('WorldEnv closed')
