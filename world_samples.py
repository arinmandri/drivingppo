"""
훈련용 world 랜덤 생성
"""
from typing import Literal, Callable

import math, random
import numpy as np
from numpy import ndarray as Arr
from random import randint

from drivingppo.world import World, Car, OBSTACLE_VALUE, create_empty_map, angle_of, distance_of, pi, pi2, rad_to_deg
from drivingppo.environment import SPD_MAX_STD

MAP_W = 150
MAP_H = 150

NEAR = 3.0

W_CONFIG = {
    'map_border': False,
    'near': NEAR,
    'far': 999.9,
}
CAR_NEAR = math.sqrt(Car.w**2 + Car.h**2) / 2  # 장애물 피하기 기능을 학습한다곤 해도 목적지와 장애물이 이 이상 가깝지는 말자.  # 에이전트 대각선길이의 반  (1.5, 3)-->1.68

"""
생초보
바로 눈앞에 목적지 떠먹여줌.

기초 주행 학습
웬만하면 시간초과 벌점을 받지 않도록 코스를 짧게 시간을 매우 넉넉히 주자.

장애물 추가
이제 그럭저럭 제한시간 안에 들어올만하게 코스 길이를 조절할 것.
벽 앞에서 가만히 있기를 택하지 않도록; 시간초과 벌점을 충돌만큼 부여.
"""

def gen_0(): return generate_random_world_plain(map_h= 50, map_w= 50, num=1,  wpoint_dist_min=6,  wpoint_dist_max=12, ang_init='half', ang_lim=pi*0.5, spd_init=0)
def gen_1(): return generate_random_world_plain(map_h=150, map_w=150, num=4,  wpoint_dist_min=8,  wpoint_dist_max=20, ang_init='rand', ang_lim=pi*1.0, spd_init='rand')
def gen_2(): return generate_random_world_plain(map_h=300, map_w=300, num=30, wpoint_dist_min=8,  wpoint_dist_max=45, ang_init='rand', ang_lim=pi*1.0, spd_init='rand')

def generate_random_world_plain(
        map_w=MAP_W,
        map_h=MAP_H,
        num=15,
        wpoint_dist_min=NEAR*2,
        wpoint_dist_max=20,
        ang_init:float|Literal['p', 'half', 'rand', 'inv']='p',
        ang_lim=pi*0.5,
        pos_init:Literal['corner', 'center']='center',
        spd_init:float|Literal['rand']=0,
        seed=None,
    ):

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    if pos_init == 'center':
        px = map_w/2
        pz = map_h/2
        pangle_x = np.random.uniform(0, pi2)
        init_ang = np.random.uniform(0, pi2)  if ang_init == 'rand'  else pangle_x-pi/2 + np.random.uniform(0, pi)  if ang_init == 'half'  else pangle_x  if ang_init == 'p'  else pangle_x+pi  if ang_init == 'inv'  else ang_init
    elif pos_init == 'corner':
        px = map_w/10
        pz = map_h/10
        pangle_x = np.random.uniform(0, pi2)
        init_ang = np.random.uniform(0, pi/4)  if ang_init == 'rand'  or ang_init == 'half'  else pangle_x  if ang_init == 'p'  else pangle_x+pi  if ang_init == 'inv'  else ang_init

    pspeed = np.random.uniform(-SPD_MAX_STD, SPD_MAX_STD)*0.5  if spd_init == 'rand'  else spd_init


    # 목표점 생성
    waypoints = generate_random_waypoints(num,
                                              map_w, map_h,
                                              px, pz,
                                              init_ang=init_ang,
                                              angle_change_limit=ang_lim,
                                              min_dist=wpoint_dist_min,
                                              max_dist=wpoint_dist_max)

    # 맵 생성
    obstacle_map = create_empty_map(map_w, map_h)

    w = World(
        wh=(map_w, map_h),
        player=Car({
            'playerPos': {'x': px, 'z': pz},
            'playerBodyX': pangle_x*rad_to_deg,
            'playerSpeed': pspeed,
        }),
        obstacle_map=obstacle_map,
        waypoints=waypoints,
        config=W_CONFIG|{
            'far': wpoint_dist_max + NEAR + 12.0,
        }
    )
    return w


def generate_random_waypoints(
        num,
        map_w, map_h,
        init_x, init_z,
        init_ang,
        angle_change_limit=pi/2,
        min_dist=2.0,
        max_dist=Car.SPEED_MAX_W,
        margin:int=5,
        seed=None,
    ) -> list[tuple[float, float]]:

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    if min_dist > max_dist: raise Exception('min_dist >= max_dist')

    waypoints = []

    last_x = init_x
    last_z = init_z
    angle  = init_ang

    for i in range(num):

        # 랜덤 거리
        distance = np.random.uniform(min_dist, max_dist)

        # 새 좌표 계산
        x = last_x + math.sin(angle) * distance
        z = last_z + math.cos(angle) * distance

        # 맵 경계 제한 (맵 밖으로 나가지 않게 클램핑)
        if   x <  margin:       x = -x + margin*2
        elif x >= map_w-margin: x = -x + (map_w - margin)*2
        if   z <  margin:       z = -z + margin*2
        elif z >= map_h-margin: z = -z + (map_h - margin)*2

        # 좌표 추가
        waypoints.append((x, z))

        # 랜덤 각도: 이전 각도에서 일정 이내로만 변화를 제한
        angle_d = np.clip(np.random.normal(0, angle_change_limit/pi, 1), -angle_change_limit, angle_change_limit)
        angle = angle_of(last_x, last_z, x, z) + angle_d
        angle = angle % pi2

        last_x = x
        last_z = z

    return waypoints
