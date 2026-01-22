from typing import Any
import csv
from collections.abc import Iterable

from .common import MAP_W, MAP_H, LIDAR_NUM, LIDAR_RANGE, LIDAR_START, LIDAR_END
from .world import World, Car, create_empty_map, pi2
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
            goal_points=[],
            debugMode=False,
        ):

        self.debugMode = debugMode

        self.model = PPO.load(model_path)

        if obstacle_map is None:
            obstacle_map = create_empty_map(MAP_W, MAP_H)

        self.world = create_initial_world(
            obstacle_map=obstacle_map,
            goal_points=goal_points,
            # TODO x, z 좌표도 초기화?
        )

        self.set_path(goal_points)

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
        if path:
            self.world.goal_points = list(map(tuple, path))
        else:
            self.world.goal_points = []

    def update_obstacle_map(self, obstacle_map: Arr):
        """
        장애물맵을 업데이트한다.
        """
        pass # TODO

    def get_action(self, info: dict[str, Any]) -> tuple[bool, float, float]:

        # 내부 World 상태 업데이트
        world = self.world
        world.player.status = info
        world.step(0.0)

        # 상태값
        observation = get_state(world)
        if self.debugMode: print(observation_str(observation), f' / GOAL({world.current_goal_idx}/{len(world.goal_points)}): {world.get_relative_angle_to_goal()/pi2*360:.1f}, {world.get_distance_to_goal():.1f}')

        # 도착했으면 그냥 STOP
        a0 = world.get_relative_angle_to_goal()
        d0 = world.get_distance_to_goal()
        if world.arrived:
            if DEBUG: print('도착', d0)
            return True, 0.0, 0.0

        # 액션 산출
        action, _states = self.model.predict(observation, deterministic=True)
        if self.debugMode: print(action_str(action))
        ws, ad = float(action[0]), float(action[1])
        if DEBUG:  # 액션, 관찰 로그
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
            if DEBUG: print(f'GOAL: {d0:.1f} / {a0*pi2/360:.1f} | ACTION: {ws:.1f}, {ad:.1f}')

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


def create_initial_world(
        obstacle_map:Arr|None=None,
        goal_points:list[tuple[float, float]]=[],
        blStartX:float=0,
        blStartY:float=0,
        blStartZ:float=0,
) -> World:
    """
    Tank Challenge /init 참고
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
        goal_points=goal_points,
        config={
            'lidar_raynum': LIDAR_NUM,
            'lidar_range': LIDAR_RANGE,
            'angle_start': LIDAR_START,
            'angle_end': LIDAR_END,
            'near': 6.0,
            'far': 12.0,
            'skip_past_waypoints': True,
            'skip_waypoints_num': 10,
        }
    )
    return w