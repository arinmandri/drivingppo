from typing import Callable, Literal
import math
from datetime import datetime

from .world import World, distance_of, angle_of, pi, pi2, rad_to_deg
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

import numpy as np
from numpy import ndarray as Arr



def speed_norm(speed):
    """속도 정규화"""
    return speed / SPD_SCFAC

def _distance_score_near(x:float) -> float:
    d = x + 10.0
    x = 100./d/d
    if x <= 1:
        return x
    else:
        return 1.0

distance_score_near_base = _distance_score_near(LIDAR_RANGE)

def distance_score_near(x:float) -> float:
    """거리가까움점수"""
    return max(0, _distance_score_near(x) - distance_score_near_base)

def distance_score_far(x:float) -> float:
    """거리멂점수(비례)"""
    return x / DIS_SCFAC


def get_path_features_srel(world:World) -> list[float]:
    """
    경로 정보
    바로 앞의 점 몇 개의 거리와 각도.
    """

    path_data = []
    x0 = world.player.x
    z0 = world.player.z
    a0 = world.player.angle_x

    # 각 목표점의 거리, 각도 정보
    for index_rel in range(LOOKAHEAD_POINTS):
        # 이전 목표점 기준
        x1, z1 = world.get_curr_wpoint(index_rel)
        d_from_prev = distance_of(x0, z0, x1, z1)
        if d_from_prev == 0:
            a_from_prev = 0
        else:
            a1          = angle_of(x0, z0, x1, z1)
            a_from_prev = a1 - a0
            a0 = a1
        x0 = x1
        z0 = z1

        a_fp_norm = ((a_from_prev + pi) % pi2 - pi) / pi  # 각도(이전 목표점 기준)
        d_near = distance_score_near(d_from_prev)  # 거리 가까운 정도
        d_far  = distance_score_far(d_from_prev)   # 거리 먼 정도

        path_data.extend([a_fp_norm, math.cos(a_from_prev), d_near, d_far])

    return path_data

def get_path_features_fagnt(world:World) -> list[float]:
    """
    경로 정보 - 에이전트 기준
    """
    path_data = []

    for index_rel in range(LOOKAHEAD_POINTS):
        d_from_agnt = world.get_distance_to_wpoint(index_rel)
        a_from_agnt = world.get_relative_angle_to_wpoint(index_rel)
        a_fp_norm = ((a_from_agnt + pi) % pi2 - pi) / pi  # 각도(이전 목표점 기준)
        d_near = distance_score_near(d_from_agnt)  # 거리 가까운 정도
        d_far  = distance_score_far(d_from_agnt)   # 거리 먼 정도
        path_data.extend([a_fp_norm, math.cos(a_from_agnt), d_near, d_far])

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



def action_str(action):
    return f'ACTION: {action[0]:.2f}  {action[1]:.2f}'


def get_path_length_of(world:World) -> float:
    """경로 총 길이: 플레이어~첫wp + 각 wp 사이 거리"""
    pts = [(world.player.x, world.player.z)] + world.waypoints
    result = 0.0
    for i in range(len(pts)-1):
        x0, z0 = pts[i]
        x1, z1 = pts[i+1]
        d = distance_of(x0, z0, x1, z1)
        result += d
    return result


class MyMetrics:
    def __init__(self, world:World):
        self.action_history = []
        self.speed_history = []
        self.trace_length = 0.0
        self.path_len_scfac = get_path_length_of(world)
        self.__x0 = world.player.x
        self.__z0 = world.player.z

    def step(self, action, world:World):
        self.action_history.append(action)
        self.speed_history.append(world.player.speed)

        # 이동거리
        x1 = world.player.x
        z1 = world.player.z
        dd = distance_of(self.__x0, self.__z0, x1, z1)
        self.trace_length += dd
        self.__x0 = world.player.x
        self.__z0 = world.player.z

    def export(self) -> dict[str, float]:
        # 액션 분산
        if len(self.action_history) > 0:
            action_arr = np.array(self.action_history)
            ws_diff_mean = float(np.mean(np.abs(np.diff(action_arr[:, 0]))))  if len(self.action_history) > 1  else 0.0
            ad_sq_mean   = float(np.mean(np.square(action_arr[:, 1])))
        else:
            ws_diff_mean, ad_sq_mean = 0.0, 0.0

        if len(self.speed_history) > 0:
            speed_arr = np.array(self.speed_history)
            speed_var = float(np.var(speed_arr))
            speed_mean = sum(self.speed_history) / len(self.speed_history)
        else:
            speed_var = 0.0
            speed_mean = 0.0

        if self.path_len_scfac > 1e-5:
            normed_path_len = self.trace_length / self.path_len_scfac
        else:
            normed_path_len = 0.0

        return {
            "ws_diff_mean": ws_diff_mean,
            "ad_sq_mean": ad_sq_mean,
            "speed_mean": speed_mean,
            "speed_var": speed_var,
            "normed_path_len": normed_path_len,
        }

