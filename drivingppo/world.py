"""
World는 이 패키지에서 중심이 되는 데이터를 다루는 클래스이다.
Tank Challenge의 물리법칙을 모방한 가상의 세계이다.
이 모듈은 이 패키지의 common 외 다른 모듈에 의존하지 않는다.
"""
from typing import Callable
import math
import numpy as np
from numpy import ndarray as arr

from .common import MAP_W as MAP_DEFAULT_W, MAP_H as MAP_DEFAULT_H, SPD_MAX_STD

idseq = 0

pi = np.pi
pi2 = pi*2
rad_to_deg = 360/pi2
deg_to_rad = pi2/360


OBSTACLE_VALUE = 1


'''
x축은 왼쪽0에서 오른쪽
z축은 아래0에서 위쪽
x방향은 위0에서 시계방향
'''

def angle_of(x0, z0, x1, z1):
    """
    두 점의 절대 각도
    """
    dx = x1 - x0
    dz = z1 - z0

    absolute_angle = pi/2 - math.atan2(dz, dx)

    return absolute_angle

def distance_of(x0:float, z0:float, x1:float, z1:float):
    return math.hypot(x0-x1, z0-z1)

def rotate(x, z, ang):
    rx =   x * math.cos(ang) + z * math.sin(ang)
    rz = - x * math.sin(ang) + z * math.cos(ang)
    return rx, rz

def linspace(a, b, num):
    step = (b-a)/num
    r = [a + i * step for i in range(num)]
    r.append(b)
    return r


# 장애물맵 불러오기 0or1 [w][h]
def load_obstacle_map(map_file: str) -> tuple[arr, int, int]:
    map_data = []

    try:
        with open(map_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

            if not lines:
                raise ValueError("맵 파일이 비어있음.")

            h = len(lines)
            w = len(lines[0])

            for i, line in enumerate(lines):
                try:
                    row = [int(c) for c in line]
                except ValueError:
                    raise ValueError(f"{i+1}번째 줄에 숫자(0 또는 1)가 아닌 문자가 포함됨.")

                if len(row) != w:
                    raise ValueError(
                        f"{i+1}번째 줄의 열(Column) 개수 불일치: 기대값 {w}, 실제 {len(row)}"
                    )

                if any(c not in (0, 1) for c in row):
                    raise ValueError(f"{i+1}번째 줄에 0 또는 1이 아닌 값이 포함됨.")

                map_data.append(row)

        print(f"INFO: 장애물 맵({w}*{h}) 불러옴.")
        return np.array(map_data), w, h

    except FileNotFoundError as e:
        print(f"ERROR: 맵 파일({map_file})이 없음.", e)
        raise e

def create_empty_map(w, h) -> arr:
    # 맵 0으로 초기화
    return np.zeros((h, w), dtype=int)




class Car:
    """
    차 한 대.
    속성: 크기, 위치, 방향, 속도 등.
    """
    SPEED_MAX_W:float = 10.0
    SPEED_MAX_S:float = -5.0
    w = 1.5
    h = 3

    engine_power = 0.001 *  2.3
    brake_power  = 0.001 *  2.7
    friction     = 0.001 *  0.3
    stop         = 0.001 * 50
    turn_speed   = 0.001 * 40 * deg_to_rad
    drag_coeff   = 0.001 * 0.02  # 공기저항 계수: 속도의 제곱에 비례하여 저항 생성

    def __init__(self, init_status={}):

        self.status = init_status

    @property
    def status(self):
        # Tank Challenge의 /info 와 통일
        return {
            "playerPos":{
                "x": self.x,
                "y": self.y,
                "z": self.z
            },
            "playerSpeed": self.speed,
            "playerBodyX": self.angle_x * rad_to_deg,
            "playerBodyY": 0,
            "playerBodyZ": 0,
            "playerHealth": 0,
            "playerTurretX": 0,
            "playerTurretY": 0,
        }

    @status.setter
    def status(self, init_status):
        # Tank Challenge의 /info 와 통일
        # 기본값 0
        pos   = init_status.get('playerPos', {})
        self.x:float       = pos.get('x', 0)
        self.y:float       = pos.get('y', 0)
        self.z:float       = pos.get('z', 0)
        self.angle_x:float = init_status.get('playerBodyX', 0) * deg_to_rad
        self.speed:float   = init_status.get('playerSpeed', 0)

    def get_corners(self) -> list[tuple[float, float]]:
        """
        네 꼭지점 위치 (그리기용)
        """
        corners = []
        for dx, dz in [(-self.w/2, self.h/2), (self.w/2, self.h/2), (self.w/2, -self.h/2), (-self.w/2, -self.h/2)]:
            rx, rz = rotate(dx, dz, self.angle_x)
            corners.append((self.x + rx, self.z + rz))
        return corners

    def get_points_to_check_collision(self):
        """
        충돌판정에 쓰이는 점
        """
        corners = self.get_corners()
        points = []
        num = 5
        for i in range(len(corners)):
            # 변 따라 점 몇 개씩
            cx0, cz0 = corners[i]
            cx1, cz1 = corners[(i+1)%len(corners)]
            cxs = linspace(cx0, cx1, num)
            czs = linspace(cz0, cz1, num)
            points.extend([(cxs[k], czs[k]) for k in range(num+1)])
        points.append((self.x, self.z))
        return points


    def apply_drag_and_rolling(self, dt):
        '''현재 속도에 따른 자연적인 저항력 적용'''
        drag = self.drag_coeff * (self.speed ** 2)
        rolling = self.friction

        if self.speed > 0:
            self.speed -= (drag + rolling) * dt
            if self.speed < 0:  self.speed = 0
        if self.speed < 0:
            self.speed += (drag + rolling) * dt
            if self.speed > 0:  self.speed = 0

    def control_w(self, weight, dt):
        """
        전진 가속
        """
        if weight < 0: return

        # 마찰 적용
        self.apply_drag_and_rolling(dt)

        # 후진 중: 브레이크
        if self.speed < 0:
            f = self.brake_power * weight
            self.speed += f * dt
            if self.speed > 0: self.speed = 0

        # 전진/정지 중: 가속
        elif self.speed >= 0:
            f = self.engine_power * weight
            self.speed += f * dt

    def control_s(self, weight, dt):
        """
        후진 가속
        """
        if weight < 0: return

        self.apply_drag_and_rolling(dt)

        # 전진 중: 브레이크
        if self.speed > 0:
            f = self.brake_power * weight
            self.speed -= f * dt
            if self.speed < 0: self.speed = 0

        # 후진/정지 중: 가속
        elif self.speed <= 0:
            f = self.engine_power * weight
            self.speed -= f * dt

    def control_stop(self, dt):
        """
        브레이크 동작
        """
        self.apply_drag_and_rolling(dt)

        if self.speed > 0:
            self.speed -= self.stop * dt
            if self.speed < 0: self.speed = 0
        elif self.speed < 0:
            self.speed += self.stop * dt
            if self.speed > 0: self.speed = 0

    def control_ad(self, weight, dt):
        """
        선회 동작
        """
        self.angle_x = (self.angle_x + self.turn_speed * weight * dt) % pi2

    def step(self, dt) -> bool:
        """
        속도와 각도에 따라 위치를 이동한다.
        return 움직임 여부
        """
        if self.speed == 0: return False

        move_step = self.speed * dt / 1000  # 이동거리 = 속도 * 시간

        self.x += math.sin(self.angle_x) * move_step
        self.z += math.cos(self.angle_x) * move_step

        return True

    def clone(self):
        return Car(self.status)



class World:
    """
    맵.
    차.
    충돌판정.
    순서있는 목표점 목록.
    XXX 코스트맵? 높이나 경사?
    XXX 장애물 새로 발견하여 장애물맵 업데이트?
    """
    def __init__(self,
                 wh:tuple[int, int]|None=None,
                 player:Car|None=None,
                 obstacle_map:arr|None=None,
                 waypoints:list[tuple[float, float]]=[],
                 config={}
        ):
        # 디버깅용 고유번호
        global idseq
        idseq += 1
        self.id = idseq


        self.init_config = config

        self.near:float = config.get('near',  2.5)  # 목표점과의 거리가 이보다 작으면 도착 판정
        self.far:float  = config.get('far',  12.0)  # 목표점과의 거리가 이보다 크면 길잃음 판정
        self.map_border = config.get('map_border', True)  # 맵 경계와 부딪힌다고 판정
        self.skip_past_waypoints:bool = config.get('skip_past_waypoints', False)  # 현재로부터 가장 가까운 waypoint로 건너뜀.
        self.skip_waypoints_num:int   = config.get('skip_waypoints_num', 10)   # skip_past_waypoints에서 최대 몇 개를 건너뛸지

        self.t_acc = 0  # 에피소드 경과시간(XXX 현재 스스로 업데이트하지 않고 WorldController에서 받아오는데... 현재는 info 내보낼 때만 씀. 경과시간 관리를 WorldController 말고 여기서 하는 게 맞는지 고민중.)

        # 맵
        # wh, obstacle_map 둘 중 하나를 지정하면 나머지는 자동, 둘 다 없으면 기본
        if wh is None:
            if obstacle_map is None:
                self.MAP_W, self.MAP_H = MAP_DEFAULT_W, MAP_DEFAULT_H
            else:
                self.MAP_H, self.MAP_W = obstacle_map.shape
        else:
            self.MAP_W = wh[0]
            self.MAP_H = wh[1]

        if obstacle_map is None:
            self.obstacle_map = create_empty_map(self.MAP_W, self.MAP_H)
        else:
            if (self.MAP_H, self.MAP_W) != obstacle_map.shape:
                raise ValueError(f'World의 wh {(self.MAP_H, self.MAP_W)}, obstacle_map.shape {obstacle_map.shape} 크기 불일치')
            self.obstacle_map = obstacle_map

        # 플레이어
        if player is None:
            player = Car({
                'pos': {'x': self.MAP_W/2, 'z': self.MAP_H/2}
            })
        self.player = player
        self.player_collision = False
        self.trace = [(self.player.x, self.player.z, 0.0)]
        self.trace_count = 0
        self.trace_max = 800

        # 목표점
        self.__waypoints:list[tuple[float, float]] = waypoints
        self.__waypoint_idx:int = 0

        self.ws:float  = 0.0      # -1.0 (후진) ~ 1.0 (전진)
        self.ad:float  = 0.0      # -1.0 (좌회전) ~ 1.0 (우회전)
        self.stop:bool = False   # True면 정지(브레이크)

    def __str__(self):
        return f"World-{self.id}({self.size})"

    def __repr__(self):
        return f"World-{self.id}({self.size})"

    @property
    def size(self):
        return self.MAP_W, self.MAP_H

    @property
    def waypoints(self):
        return self.__waypoints

    @property
    def path_len(self):
        return len(self.__waypoints)

    @property
    def arrived(self) -> bool:
        return self.waypoint_idx >= len(self.__waypoints)

    @property
    def lost(self) -> bool:
        return self.get_distance_to_wpoint() > self.far

    @waypoints.setter
    def waypoints(self, waypoints:list[tuple[float, float]]):
        self.__waypoints = waypoints
        self.__waypoint_idx = 0

    @property
    def waypoint_idx(self):
        return self.__waypoint_idx

    @waypoint_idx.setter
    def waypoint_idx(self, i:int):
        if i < 0: i = 0
        if i >= len(self.__waypoints): i = len(self.__waypoints)-1
        self.__waypoint_idx = i

    def next_wpoint(self):
        if self.__waypoint_idx < self.path_len:
            self.__waypoint_idx += 1
        return self.__waypoint_idx


    # 충돌 판정
    def check_collision(self) -> bool:
        result = self.check_collision_player()
        self.player_collision = result
        return result

    def check_collision_player(self) -> bool:
        # 맵밖에 나간 경우도 충돌 취급 및 못나가게 제한
        if self.map_border:
            outofmap = False
            if self.player.x < 0:          self.player.x = 0;          outofmap = True
            if self.player.z < 0:          self.player.z = 0;          outofmap = True
            if self.player.x > self.MAP_W: self.player.x = self.MAP_W; outofmap = True
            if self.player.z > self.MAP_H: self.player.z = self.MAP_H; outofmap = True
            if outofmap: return True

        # 플레이어 충돌 판정
        player_points = self.player.get_points_to_check_collision()
        for p in player_points:
            x = p[0]
            z = p[1]
            if x >= 0 and x < self.MAP_W and z >= 0 and z < self.MAP_H:  # 맵 안의 점은 장애물맵으로 판정
                if self.obstacle_map[int(z)][int(x)] == OBSTACLE_VALUE:
                    return True
            else:
                if self.map_border: return True
        return False

    def set_action(self, ws:float, ad:float, stop:bool=False):
        self.ws = max(min(ws, 1.0), -1.0)
        self.ad = max(min(ad, 1.0), -1.0)
        self.stop = stop

    def step(self, dt, callback:Callable|None=None) -> tuple[bool, bool, bool]:
        """
        Return: 플레이어 움직임?, 충돌?, 목표점도달?
        """
        if dt < 0: dt=0
        self.t_acc += dt

        self.control(dt)

        p = self.player
        result_p = p.step(dt)

        if result_p: # 플레이어 움직임
            self.trace_count += 1
            if self.trace_count >= 3:
                self.trace_count = 0
                self.trace.append((p.x, p.z, self.ws))
            if len(self.trace) > self.trace_max: # 기록이 길면 오래된거 버림
                self.trace = self.trace[-self.trace_max:]

        result_collision = self.check_collision()

        # 목표점 도달 판정
        result_wpoint = False
        while True:
            if self.arrived:
                break
            distance = self.get_distance_to_wpoint()
            angle    = self.get_relative_angle_to_wpoint()
            if distance < 1.0 \
            or distance < self.near and  math.cos(angle) < 0:  # 일정 거리 이내에서 멀어지는 방향이 되는 순간
                result_wpoint = True
                self.next_wpoint()
            else:
                break

        if self.skip_past_waypoints:
            # 가야했던 경유지를 지나친 경우 생략. (최대 skip_waypoints_num개)
            nearest_idx = self.__waypoint_idx
            nearest_dist = self.get_distance_to_wpoint(0)
            for i in range(1, min(self.skip_waypoints_num, len(self.__waypoints) - self.__waypoint_idx)):
                dist = self.get_distance_to_wpoint(i)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = self.__waypoint_idx + i
            self.__waypoint_idx = nearest_idx

        if callback: callback(self)

        return result_p, result_collision, result_wpoint


    def control(self, dt):
        """
        조작상태에 따라 동작
        """
        if self.stop:
            self.player.control_stop(dt)
        else:
            if self.ws > 0:
                self.player.control_w(self.ws, dt)
            elif self.ws < 0:
                self.player.control_s(-self.ws, dt)
            else:
                self.player.apply_drag_and_rolling(dt)

        self.player.control_ad(self.ad, dt)

    def get_curr_wpoint(self, index_rel:int=0) -> tuple[float, float]:
        index = self.__ind_rel_to_abs(index_rel)
        return self.__waypoints[index]

    def get_distance_to_wpoint(self, index_rel:int=0):
        """
        현재 플레이어 위치에서 현재 목표점까지 거리.
        index: 몇 번째 목표점?(현재목표점 기준 즉 0이면 현재목표점) 범위 벗어나면 마지막 목표점의 것으로.
        목표점이 없으면 마지막 목표점 반복.
        """
        if not self.__waypoints: return 0

        index = self.__ind_rel_to_abs(index_rel)

        tx, tz = self.__waypoints[index]
        px, pz = self.player.x, self.player.z

        return math.hypot(tx-px, tz-pz)

    def _get_absolute_angle_to_wpoint(self, index:int):
        """
        현재 플레이어 위치에서 현재 목표점을 바라보는 방향의 절대 각도
        """
        tx, tz = self.__waypoints[index]
        px, pz = self.player.x, self.player.z
        return angle_of(px, pz, tx, tz)

    def get_relative_angle_to_wpoint(self, index_rel:int=0):
        """
        현재 플레이어 위치에서 현재 목표점을 바라보는 방향의 상대 각도 -pi~pi
        index_rel: 몇 번째 목표점?(현재목표점 기준 즉 0이면 현재목표점) 범위 벗어나면 마지막 목표점의 것으로.
        목표점이 없으면 플레이어 위치를 목표점으로 취급하여 0.
        """
        if not self.__waypoints: return 0

        index = self.__ind_rel_to_abs(index_rel)

        abs_ang = self._get_absolute_angle_to_wpoint(index)
        rel_ang = abs_ang - self.player.angle_x
        rel_ang = (rel_ang + pi) % pi2 - pi

        return rel_ang

    def __ind_rel_to_abs(self, index_rel:int):
        index = self.__waypoint_idx + index_rel
        if index < 0:
            return 0
        if index >= len(self.__waypoints):
            return len(self.__waypoints) - 1
        return index



    @property
    def info(self):
        # Tank Challenge의 /info 와 통일
        player_status = self.player.status
        etc = {
            'time': self.t_acc / 1000,
        }
        return player_status | etc


    def clone(self):
        o = World(
            (self.MAP_W, self.MAP_H),
            self.player.clone(),
            self.obstacle_map.copy(),
            self.__waypoints[:],
            config=self.init_config
        )
        o.t_acc = self.t_acc
        return o

