from typing import Any
import csv
from collections.abc import Iterable

from .world import World, Car, angle_of, create_empty_map, pi, pi2, rad_to_deg
from .common import LIDAR_NUM, LIDAR_RANGE, LIDAR_START, LIDAR_END, MAP_W, MAP_H
from .environment import get_state, observation_str, apply_action, action_str

import numpy as np
from numpy import ndarray as Arr
from stable_baselines3 import PPO


DEBUG = False
LOG_FILE_PATH = "./logs/ppo.csv"


# delete existing log file
if DEBUG:
    import os
    csv_filename = LOG_FILE_PATH
    if os.path.exists(csv_filename):
        os.remove(csv_filename)


class MyPpoAdaptor:
    """
    우리 서비스에서 나의 PPO 모델을 이용하도록 잇는다.
    """
    def __init__(
            self,
            model_path:str,
            obstacle_map:Arr|None=None,
            waypoints=[],
        ):

        self.__poop_x = 0.0  # speed 부호 판별용
        self.__poop_z = 0.0

        self.model = PPO.load(model_path)

        if obstacle_map is None:
            obstacle_map = create_empty_map(MAP_W, MAP_H)

        self.world = create_initial_world(
            obstacle_map=obstacle_map,
            waypoints=waypoints,
        )

        self.set_path(waypoints)

    def init(self, config):
        world = self.world
        p = world.player
        """
        Tank Challenge /init 참고
        {
            "blStartX": x,  # Blue Start Position
            "blStartY": y,
            "blStartZ": z,

            "rdStartX": 60,  # Red Start Position
            "rdStartY": 10,
            "rdStartZ": 280,

            "trackingMode": True,
            "detactMode": False,
            "logMode": True,
        }
        """
        p.x = config.get('blStartX', 0)
        p.z = config.get('blStartZ', 0)
        p.speed = 0
        p.angle_x = 0

    def set_path(self, path: Iterable):
        if len(path):#type:ignore
            self.world.waypoints = list(map(tuple, path))
        else:
            self.world.waypoints = []

    def update_obstacle_map(self, obstacle_map: Arr):
        """
        장애물맵을 업데이트한다.
        """
        self.world.obstacle_map = obstacle_map

    def get_action(self, info: dict[str, Any]) -> tuple[bool, float, float]:

        # 내부 World 상태 업데이트
        world = self.world
        world.player.status = info
        self.__adjust_speed_of_pooping_tank_challenge()
        world.step(0.0)

        # 도착했으면 그냥 STOP
        if world.arrived:
            if DEBUG: print('도착')
            return True, 0.0, 0.0

        # 상태값
        observation = get_state(world)
        if DEBUG: print(observation_str(observation), f' / GOAL({world.waypoint_idx}/{len(world.waypoints)}): {world.get_relative_angle_to_wpoint()/pi2*360:.1f}, {world.get_distance_to_wpoint():.1f}')

        # 액션 산출
        action, _states = self.model.predict(observation, deterministic=True)
        ws, ad = float(action[0]), float(action[1])

        if DEBUG:  # 액션, 관찰 로그
            print(action_str(action))
            if not hasattr(self, "_csv_header_written"):
                # 컬럼 한 번만 기록
                obs_cols = [f"obs_{i}" for i in range(len(observation))]
                header = ["ws", "ad"] + obs_cols
                with open(LOG_FILE_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                self._csv_header_written = True

            # 값 기록
            row = [f"{v:+.3f}" for v in [ws, ad] + observation.tolist()]
            with open(LOG_FILE_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            a0 = world.get_relative_angle_to_wpoint()
            d0 = world.get_distance_to_wpoint()
            print(f'GOAL: {d0:.1f} / {a0*pi2/360:.1f} | ACTION: {ws:.1f}, {ad:.1f}')

            apply_action(world, action)  # 확인용

        return False, ws, ad

    @property
    def lost(self) -> bool:
        """
        경로 이탈 여부
        """
        return self.world.lost

    @property
    def arrived(self) -> bool:
        """
        도착 여부
        """
        return self.world.arrived

    def __adjust_speed_of_pooping_tank_challenge(self):
        # Tank Challenge에서는 후진 상태여도 speed가 양수로 들어온다. 전차 위치 변화를 보고 후진인지를 직접 판별하여 후진이면 spped를 음수로 바꿔줘야 함.
        world = self.world

        x0 = self.__poop_x
        z0 = self.__poop_z
        x1 = world.player.x
        z1 = world.player.z

        real_angle = angle_of(x0,z0,x1,z1)
        info_angle = world.player.angle_x
        angle_diff = (info_angle - real_angle + pi) % pi2 - pi

        if abs(angle_diff) > pi/2: # 후진중
            world.player.speed *= -1

        self.__poop_x = x1
        self.__poop_z = z1

def create_initial_world(
        obstacle_map:Arr|None=None,
        waypoints:list[tuple[float, float]]=[],
        blStartX:float=0,
        blStartY:float=0,
        blStartZ:float=0,
) -> World:
    """
    원본 시뮬레이터 /init 참고
    """

    w = World(
        wh=(MAP_W, MAP_H),
        player=Car({
            "playerPos":{
                "x": blStartX,
                "y": blStartY,
                "z": blStartZ
            },
            "playerSpeed": 0,
            "playerBodyX": 0,
            "playerBodyY": 0,
            "playerBodyZ": 0,
        }),
        obstacle_map=obstacle_map,
        waypoints=waypoints,
        config={
            'lidar_raynum': LIDAR_NUM,
            'lidar_range': LIDAR_RANGE,
            'angle_start': LIDAR_START,
            'angle_end': LIDAR_END,
            'near': 6.0,
            'far': 30.0,  # smooth_los_distance(경로 단순화시 노드 사이 거리 최대값) 값보다 커야 함.
        }
    )
    return w