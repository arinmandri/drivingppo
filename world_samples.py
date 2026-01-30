"""
훈련용 world 랜덤 생성
"""
from typing import Literal, Callable

import math, random
import numpy as np
from numpy import ndarray as Arr
from random import randint

from drivingppo.world import World, Car, OBSTACLE_VALUE, create_empty_map, angle_of, distance_of, pi, pi2, rad_to_deg
from drivingppo.common import LIDAR_NUM, LIDAR_RANGE, LIDAR_START, LIDAR_END
from drivingppo.environment import SPD_MAX_STD

MAP_W = 150
MAP_H = 150


W_CONFIG = {
    'lidar_raynum': LIDAR_NUM,
    'lidar_range':  LIDAR_RANGE,
    'angle_start':  LIDAR_START,
    'angle_end':    LIDAR_END,
    'near': 3.5,
    'far': 35.0,
}
CAR_NEAR = 1.5  # 장애물 피하기 기능을 학습한다곤 해도 목적지와 장애물이 이 이상 가깝지는 말자.

"""
생초보
바로 눈앞에 목적지 떠먹여줌.

기초 주행 학습
웬만하면 시간초과 벌점을 받지 않도록 코스를 짧게 시간을 매우 넉넉히 주자.

장애물 추가
이제 그럭저럭 제한시간 안에 들어올만하게 코스 길이를 조절할 것.
벽 앞에서 가만히 있기를 택하지 않도록; 시간초과 벌점을 충돌만큼 부여.
"""


def gen_0():  return generate_random_world_plain(map_h=100 , map_w=100 , num=1, wpoint_dist_min=10,  wpoint_dist_max=10,  ang_init='half', ang_lim=0, spd_init=0)

def gen_11(): return generate_random_world_plain(map_h=300, map_w=300, num=10, wpoint_dist_min=5,  wpoint_dist_max=15, ang_init='p',    ang_lim=pi*0.6,  spd_init=0)
def gen_12(): return generate_random_world_plain(map_h=300, map_w=300, num=7,  wpoint_dist_min=4,  wpoint_dist_max=15, ang_init='half', ang_lim=pi*0.2, spd_init='rand')
def gen_env_naive():
    choice = randint(0, 4)
    if choice < 2:
        return gen_11()
    if choice < 4:
        return gen_12()
    else:
        return gen_0()

def gen_21(): return generate_random_world_plain(map_h=300, map_w=300, num=13, wpoint_dist_min=4,  wpoint_dist_max=25, ang_init='rand', ang_lim=pi*0.9, spd_init=0.0)
def gen_env_plain():
    if randint(0, 1):
            return gen_21()
    else:
        choice = randint(0, 4)
        if choice < 4:
            return gen_env_naive()
        else:
            return generate_world_square(randint(30, 50), randint(30, 50), padding=5, num=4)

def gen_env_obs():
    if randint(0, 2):
        choice = randint(0, 5)
        if choice < 1:
            return generate_random_world_obs_matrix(num=11, obs_dist=randint(10, 18))
        if choice < 2:
            return generate_random_world_narrow(num=9, hollow_radius=randint(10, 15))
        if choice < 3:
            return generate_random_world_obs_between(num=6)
        if choice < 4:
            return generate_world_square(randint(30, 50), randint(30, 50), num=4)
        else:
            return gen_env_naive()
    else:
        return gen_env_plain()

def gen_inv(): return generate_random_world_plain(map_h=150, map_w=150, num=5,  wpoint_dist_min=9,  wpoint_dist_max=10, ang_init='inv',  ang_lim=0.0,     spd_init=0)


def generate_random_world_plain(
        map_w=MAP_W,
        map_h=MAP_H,
        num=15,
        wpoint_dist_min=2,
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
            'lidar_real': False,
            'far': wpoint_dist_max * 1.8,
        }
    )
    return w



def generate_random_world_obs_matrix(
        map_w=MAP_W,
        map_h=MAP_H,
        num=16,
        wpoint_dist_min=8,
        wpoint_dist_max=20,
        obs_dist=10,
        seed=None,
    ) -> World:

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    px = (int(map_w/2/obs_dist) + 0.5)*obs_dist
    pz = (int(map_h/2/obs_dist) + 0.5)*obs_dist
    pspeed = 0
    pangle_x = np.random.uniform(0, pi2)

    # 목표점 생성
    waypoints = generate_random_waypoints(num,
                                              map_w, map_h,
                                              px, pz,
                                              pangle_x,
                                              angle_change_limit=pi*0.5,
                                              min_dist=wpoint_dist_min,
                                              max_dist=wpoint_dist_max)

    # 맵 생성
    obstacle_map = create_empty_map(map_w, map_h)
    obstacle_map[::obs_dist, ::obs_dist] = 1

    empty_around_waypoints(obstacle_map, waypoints)

    w = World(
        wh=(map_w, map_h),
        player=Car({
            'playerPos': {'x': px, 'z': pz},
            'playerBodyX': pangle_x*rad_to_deg,
            'playerSpeed': pspeed,
        }),
        obstacle_map=obstacle_map,
        waypoints=waypoints,
        config=W_CONFIG
    )
    return w



def generate_random_world_narrow(
        map_w=MAP_W,
        map_h=MAP_H,
        num=15,
        hollow_radius=4,
        ang_lim=pi*0.3,
        seed=None,
    ) -> World:

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    px = map_w/2
    pz = map_h/2
    pspeed = 0
    pangle_x = np.random.uniform(0, pi2)

    min_dist = hollow_radius
    max_dist = hollow_radius * 2.0 - 1.5

    # 목표점 생성
    waypoints = generate_random_waypoints(num,
                                              map_w, map_h,
                                              px, pz,
                                              init_ang=np.random.uniform(0, pi2),
                                              angle_change_limit=ang_lim,
                                              min_dist=min_dist,
                                              max_dist=max_dist)

    # 맵 생성
    obstacle_map = create_map_narrow(map_w, map_h, waypoints + [(px, pz)], hollow_radius=hollow_radius)

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
            'far': max_dist + 10.0
        }
    )
    return w



def generate_random_world_obs_between(
        map_w=MAP_W,
        map_h=MAP_H,
        num=6,
        min_dist=16,
        max_dist=20,
        seed=None,
    ) -> World:

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    px = (randint(0, 1)*0.8 + 0.1) * map_w
    pz = (randint(0, 1)*0.8 + 0.1) * map_h
    pspeed = 0
    pangle_x = angle_of(px, pz, map_w/2, map_h/2)  # 맵 중앙을 보도록

    # 목표점 생성
    waypoints = generate_random_waypoints(num,
                                              map_w, map_h,
                                              px, pz,
                                              pangle_x,
                                              angle_change_limit=pi*0.3,
                                              min_dist=min_dist,
                                              max_dist=max_dist)

    # 맵 생성
    obstacle_map = create_empty_map(map_w, map_h)
    add_obstacles_between_wpoints(obstacle_map, waypoints)

    empty_around_waypoints(obstacle_map, waypoints, 5)

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
            'far': max_dist + 10
        }
    )
    return w



def add_obstacles_between_wpoints(obstacle_map, waypoints):
    """
    목표점 사이마다 장애물을 추가
    """

    for i in range(len(waypoints)-1):
        gx0, gz0 = waypoints[i]
        gx1, gz1 = waypoints[i+1]

        distance = distance_of(gx0, gz0, gx1, gz1)
        obs_size = int(distance / 3) - 2
        if obs_size <= 0:
            continue
        obs_size = randint(int(obs_size/2), obs_size)

        obs_pos_x = int( (gx0 + gx1)/2 + randint(0, obs_size) - obs_size/2 )
        obs_pos_z = int( (gz0 + gz1)/2 + randint(0, obs_size) - obs_size/2 )

        add_obstacle(obstacle_map,
            obs_pos_x-int(obs_size/2-0.5),
            obs_pos_z-int(obs_size/2-0.5),
            obs_pos_x+int(obs_size/2),
            obs_pos_z+int(obs_size/2))


def create_map_narrow(w, h, waypoints, hollow_radius):
    obs_radius = hollow_radius + 1

    obstacle_map = create_empty_map(w, h)

    fill_around_waypoints(obstacle_map, waypoints, obs_radius)
    empty_around_waypoints(obstacle_map, waypoints, hollow_radius)

    return obstacle_map



def fill_around_waypoints(map:Arr, points, r=CAR_NEAR):
    h, w = map.shape
    r_2 = r**2

    # 둘레 채움
    for px, py in points:
        px, py = int(px), int(py)

        y_min = int(round(max(0, min(h, py - r))))
        y_max = int(round(max(0, min(h, py + r + 1))))
        x_min = int(round(max(0, min(w, px - r))))
        x_max = int(round(max(0, min(w, px + r + 1))))

        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        mask_obs = (x_grid - px)**2 + (y_grid - py)**2 <= r_2

        map[y_min:y_max, x_min:x_max][mask_obs] = OBSTACLE_VALUE

def empty_around_waypoints(obstacle_map:Arr, points, r=CAR_NEAR):
    h, w = obstacle_map.shape
    r_2 = r**2

    # 내부 비움
    for px, py in points:
        px, py = int(px), int(py)

        y_min = int(round(max(0, min(h, py - r))))
        y_max = int(round(max(0, min(h, py + r + 1))))
        x_min = int(round(max(0, min(w, px - r))))
        x_max = int(round(max(0, min(w, px + r + 1))))

        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        mask_hollow = (x_grid - px)**2 + (y_grid - py)**2 <= r_2

        obstacle_map[y_min:y_max, x_min:x_max][mask_hollow] = 0

def generate_random_waypoints(
        num,
        map_w, map_h,
        init_x, init_z,
        init_ang,
        angle_change_limit=pi/2,
        min_dist=2,
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
        if i == 0:
            distance = max_dist
        else:
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


def add_obstacle(obstacle_map, x0, z0, x1, z1):
    map_h, map_w = obstacle_map.shape

    for x in range(x0, x1+1):
        for z in range(z0, z1+1):
            if 0 <= z < map_h and 0 <= x < map_w:
                obstacle_map[z][x] = OBSTACLE_VALUE



def add_obstacles_near_wpoints(
        obstacle_map,
        p_xz,
        waypoints,
        seed=None,
    ):
    """
    각 목표점 근처에 장애물을 하나씩 랜덤하게 추가하며,
    어떤 목표점과도 같은 위치에 배치되지 않도록 합니다.

    Args:
        obstacle_map (np.array): 맵의 2차원 배열
        waypoints (list): [(x, z), ...] 형태의 목표점 리스트 (실수 좌표)
    """

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    map_h, map_w = obstacle_map.shape
    SEARCH_RANGE = 15 # 목표점 주변 탐색 반경

    for gx, gz in waypoints:
        placed = False
        attempts = 0
        MAX_ATTEMPTS = 20

        while not placed and attempts < MAX_ATTEMPTS:
            attempts += 1

            # 장애물 중심 좌표 (정수형)
            obs_x = int(gx + random.uniform(-SEARCH_RANGE, SEARCH_RANGE))
            obs_z = int(gz + random.uniform(-SEARCH_RANGE, SEARCH_RANGE))
            obs_size = randint(1, 3)

            # 맵 경계 확인
            if (obs_x < 0 or obs_x >= map_w
            or  obs_z < 0 or obs_z >= map_h):
                continue

            margin_sq = np.square(3 + obs_size)

            # 플레이어 시작점, 목표점과의 충돌 검사
            d_sq = np.square(obs_x-p_xz[0]) + np.square(obs_z-p_xz[1])
            if d_sq < margin_sq*3: break
            break_twice = False
            for gx2, gz2 in waypoints:
                d_sq = np.square(obs_x-gx2) + np.square(obs_z-gz2)
                if d_sq < margin_sq:
                    break_twice = True
                    break
            if break_twice: break

            # 장애물 배치 확정 및 맵에 반영
            placed = True

            # 장애물 뻥튀기
            add_obstacle(
                obstacle_map,
                obs_x-obs_size,
                obs_z-obs_size,
                obs_x+obs_size,
                obs_z+obs_size)

    return obstacle_map



def generate_world_simpleLine(dist=50, h=4, end_margin=5, mid_num=3):
    w = dist + 30
    z = h/2
    return World(
        wh=(w, h),
        player=Car({
            'playerPos': {'x': 25, 'z': z},
            'playerBodyX': 90.0,
            'playerSpeed': 0.0,
        }),
        obstacle_map=create_empty_map(w, h),
        waypoints = [(25 + dist/2 - mid_num/2 + i, z) for i in range(mid_num)] + [(w-end_margin, z)],
        config=W_CONFIG|{
            'far': 999
        }
    )



def generate_world_square(
        w=50,
        h=30,
        num=8,
        padding=5,
        seed=None,
    ) -> World:

    if seed:
        np.random.seed(seed)
        random.seed(seed)


    x1 = int(round(w*0.1 +padding))
    x2 = int(round(w*0.9 -padding))
    z1 = int(round(h*0.1 +padding))
    z2 = int(round(h*0.9 -padding))

    obstacle_map = create_empty_map(w, h)
    # 가운데 점?
    obsType = randint(0, 2)
    if obsType < 1:
        obstacle_map[h//2, w//2] = OBSTACLE_VALUE
    elif obsType < 2:
        obsx1 = x1 + 10
        obsx2 = x2 - 10
        obsz1 = z1 + 10
        obsz2 = z2 - 10
        if obsx1 < obsx2  and obsz1 < obsz2:
            obstacle_map[obsz1:obsz2, obsx1:obsx2] = OBSTACLE_VALUE

    if randint(0,1) == 1:
        world = World(
            wh=(w, h),
            player=Car({
                'playerPos': {'x': x1, 'z': z1},
                'playerBodyX': 0.0,
                'playerSpeed': 0.0,
            }),
            obstacle_map=obstacle_map,
            waypoints = [ # 시계방향
                (x1, z2)  if i%4==0  else
                (x2, z2)  if i%4==1  else
                (x2, z1)  if i%4==2  else
                (x1, z1)  for i in range(num)
            ],
            config=W_CONFIG|{
                'far': 999
            }
        )
    else:
        world = World(
            wh=(w, h),
            player=Car({
                'playerPos': {'x': x1, 'z': z1},
                'playerBodyX': 90.0,
                'playerSpeed': 0.0,
            }),
            obstacle_map=obstacle_map,
            waypoints = [ # 반시계방향
                (x2, z1)  if i%4==0  else
                (x2, z2)  if i%4==1  else
                (x1, z2)  if i%4==2  else
                (x1, z1)  for i in range(num)
            ],
            config=W_CONFIG|{
                'far': 999
            }
        )
    return world
