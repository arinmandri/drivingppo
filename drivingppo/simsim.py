"""
Tank Challenge의 그래픽 및 API 호출 기능을 모방한다.
"""
import time, math

import numpy as np
from numpy import ndarray as arr
import tkinter as tk
import threading, requests

from .common import MAP_W, MAP_H
from .world import World, Car, pi, pi2, deg_to_rad, rad_to_deg, distance_of



def get_curt():  # 현재시각 (천분초)
    return int(time.time() * 1000)


def generate_lut(color_start, color_end, steps, steepness=1.5):
    """
    두 색상 사이의 그라데이션 색상 코드를 미리 계산하여 리스트로 반환
    steepness: 곡선의 가파름 정도 (클수록 0 근처에서 변화가 심함)
    """
    start_rgb = tuple(int(color_start[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(color_end[i:i+2], 16) for i in (1, 3, 5))

    lut = []
    # 분모는 상수이므로 루프 밖에서 미리 계산 (최적화)
    denom = math.log(1 + steepness)

    r,g,b=0,0,0

    for i in range(steps + 1):
        # 선형 진행률 (0.0 ~ 1.0)
        linear_t = i / steps

        # 값이 작을 때 빠르게 바뀌도록 로그 함수 적용
        t = math.log(1 + steepness * linear_t) / denom

        # 변환된 t로 색상 보간
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
        lut.append(f'#{r:02x}{g:02x}{b:02x}')

    lut.append(f'#{r:02x}{g:02x}{b:02x}')  # 이거 이용시에 인덱스를 클리핑하는 대신 그냥 여기에 방패를 추가하면 되잖아?

    return lut


def api_get(url):
    try:
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            return response.json()
    except (requests.exceptions.ConnectionError,
            requests.exceptions.ConnectTimeout):
        # 서버 연결 안 됨: 무시
        print('서버X')
        return None
    except Exception as e:
        print(f"API Error ({url}): ({type(e)}): {e}")
    return None

def api_post(url, data):
    try:
        response =  requests.post(url, json=data, timeout=1)
        if response.status_code == 200:
            return response.json()
    except (requests.exceptions.ConnectionError,
            requests.exceptions.ConnectTimeout):
        # 서버 연결 안 됨: 무시
        print('서버X')
        return None
    except Exception as e:
        print(f"API Error ({url}): ({type(e)}): {e}")
    return None



class FieldCanvasPainter:
    """
    WorldViewer에서 직접 그리지 않고
    그릴거랑 그에 필요한 점좌표들을 따로 모아두기: add_* 메서드들
    모은 점좌표들 한 번에 계산.(넘파이 이용)
    모아둔 그릴것들 하나씩 뽑으면서 변환된 점좌표들 뽑아 그림.
    """
    def __init__(self, canvas:tk.Canvas, width:int, height:int, scale:int):

        self.canvas = canvas

        self.mode = 0  # [보기모드] 0:전체 1:플레이어중심 2:입체
        self.scale = scale
        self.scale2 = 20

        self.W = width
        self.H = height
        self.cW = self.W * scale
        self.cH = self.H * scale

        self.__reset()

    def __reset(self):
        self.__tasks = []
        self.__params = []
        self.__points = np.zeros((0, 2))
        self.__i = 0

    def __pushPoints(self, points:list[tuple[float, float]]):
        self.__points = np.concatenate((self.__points, np.array(points)), axis=0)

    def __popPoints(self, n) -> arr:
        points = self.__points[self.__i:n+self.__i]
        self.__i += n
        return points

    def __popParams(self, n) -> list:
        params = self.__params[:n]
        self.__params = self.__params[n:]
        return params


    def add_polygon(self, corners:list, fill="grey", width=1, outline="black"):
        self.__tasks.append('polygon')
        self.__params.extend([len(corners), fill, width, outline])
        self.__pushPoints(corners)
    def __draw_polygon(self, corners, fill, width, outline):
        self.canvas.create_polygon(corners.tolist(), fill=fill, width=width, outline=outline)

    def add_line(self, x0, z0, x1, z1, fill="black", width=1):
        self.__tasks.append('line')
        self.__params.extend([fill, width])
        self.__pushPoints([(x0, z0), (x1, z1)])
    def __draw_line(self, x0, z0, x1, z1, fill, width):
        if np.isnan(x0) or np.isnan(x1) or np.isnan(z0) or np.isnan(z1): return
        self.canvas.create_line(x0, z0, x1, z1, fill=fill, width=width)

    def add_rectangle(self, x0, z0, x1, z1, fill="black", width=0, outline=""):
        self.__tasks.append('rectangle')
        self.__params.extend([fill, width, outline])
        if self.mode == 2:
            self.__pushPoints([(x0, z0), (x0, z1), (x1, z1), (x1, z0)])
        else:
            self.__pushPoints([(x0, z0), (x1, z1)])
    def __draw_rectangle(self, x0, z0, x1, z1, fill, width, outline):
        if np.isnan(x0) or np.isnan(x1) or np.isnan(z0) or np.isnan(z1): return
        if  x1 >= 0 and x0 <= self.cW \
        and z1 >= 0 and z0 <= self.cH:
            self.canvas.create_rectangle(x0, z0, x1, z1, fill=fill, width=width, outline=outline)

    def add_circle(self, x, z, r, fill="black", width=0, outline=""):
        self.__tasks.append('circle')
        self.__params.extend([r, fill, width, outline])
        self.__pushPoints([(x, z)])
    def __draw_circle(self, x, z, r, fill, width, outline):
        if np.isnan(x) or np.isnan(z): return
        if  x >= -r and x <= self.cW + r \
        and z >= -r and z <= self.cH + r:
            self.canvas.create_oval(x-r, z-r, x+r, z+r, fill=fill, width=width, outline=outline)

    def add_cell_filling(self, x, z, fill='black', text='', textfill='black'):
        self.add_rectangle(x, z, x+1, z+1, fill, 0, '')
        self.add_text(x+0.5, z+0.5, text, 'center', textfill, ("Consolas", 10))

    def add_cell_stone(self, x, z, fill='black', text='', textfill='black'):
        self.add_circle(x, z, (self.scale if self.mode == 0 else self.scale2)/2, fill, 0, '')
        self.add_text(x, z, text, 'center', textfill, ("Consolas", 10))

    def add_text(self, x, z, text, anchor="center", fill='black', font=("Consolas", 10)):
        self.__tasks.append('text')
        self.__params.extend([text, anchor, fill, font])
        self.__pushPoints([(x, z)])
    def __draw_text(self, x, z, text, anchor, fill, font):
        self.canvas.create_text(x, z,
                                anchor=anchor,#type:ignore
                                text=text,
                                fill=fill,
                                font=font)


    def __transform_points(self, p:Car):
        # 맵 상의 점을 화면상의 위치로 변환
        if self.mode == 0:
            self.__points[:, 1] = self.H - self.__points[:, 1]  # 화면의 y와 우리의 z축 방향 반대
            self.__points *= self.scale

        if self.mode == 1:
            # 플레이어 위치 중심
            self.__points[:, 0] =   (self.__points[:, 0] - p.x)*self.scale2 + self.cW/2
            self.__points[:, 1] = - (self.__points[:, 1] - p.z)*self.scale2 + self.cH/2
        
        elif self.mode == 2:
            # 입체감
            camera_height = 30
            focal_length  = 30
            pitch = -45 * deg_to_rad

            self.__points[:, 0] = self.__points[:, 0] - p.x
            self.__points[:, 1] = self.__points[:, 1] - p.z
            c = np.cos(p.angle_x)
            s = np.sin(p.angle_x)
            R = np.array([
                [c, s],
                [-s,  c],
            ])
            self.__points = self.__points @ R

            y_rel = -camera_height
            x_yaw = self.__points[:, 0]
            z_yaw = self.__points[:, 1] + 15
            c = np.cos(pitch)
            s = np.sin(pitch)
            y_final = y_rel * c - z_yaw * s
            z_final = y_rel * s + z_yaw * c

            z_final_safe = np.where(z_final <= 1, np.nan, z_final)
            factor = focal_length / z_final_safe
            screen_x = x_yaw   * factor
            screen_y = y_final * factor
            
            self.__points = np.column_stack((screen_x, screen_y))
            self.__points *= self.scale2
            self.__points[:, 0] =   self.__points[:, 0] + self.cW/2
            self.__points[:, 1] = - self.__points[:, 1] + self.cH/2


    def draw(self, player:Car):

        self.__transform_points(player)

        for task in self.__tasks:
            if task == 'polygon':
                n, fill, width, outline = self.__popParams(4)
                points = self.__popPoints(n)
                self.__draw_polygon(points, fill, width, outline)
            if task == 'line':
                fill, width = self.__popParams(2)
                points = self.__popPoints(2)
                self.__draw_line(points[0,0], points[0,1], points[1,0], points[1,1], fill, width)
            if task == 'rectangle':
                fill, width, outline = self.__popParams(3)
                if self.mode == 2:
                    points = self.__popPoints(4)
                    self.__draw_polygon(points, fill, width, outline)
                else:
                    points = self.__popPoints(2)
                    self.__draw_rectangle(points[0,0], points[0,1], points[1,0], points[1,1], fill, width, outline)
            if task == 'circle':
                r, fill, width, outline = self.__popParams(4)
                points = self.__popPoints(1)
                self.__draw_circle(points[0,0], points[0,1], r, fill, width, outline)
            if task == 'text':
                text, anchor, fill, font = self.__popParams(4)
                points = self.__popPoints(1)
                self.__draw_text(points[0,0], points[0,1], text, anchor, fill, font)

        self.__reset()



class WorldViewer:
    """
    World의 모습을 tkinter을 이용해 보여준다.
    """

    TRACE_LUT_SIZE = 25

    def __init__(self, world:World=World(), scale:int=0, frame_delay=33, *, auto_update=True):
        self.color = {
            'grid1': '#DDDDDD',
            'grid2': '#BBBBBB',
            'obstacle': '#444',
            'wpoint': '#00EE88',
            'wpoint_past': '#EEAA88',
            'traceF_LUT': generate_lut('#EEEEEE', "#4499FF", WorldViewer.TRACE_LUT_SIZE), # 빠를 수록 진한색으로 표시
            'traceB_LUT': generate_lut('#EEEEEE', "#F877BB", WorldViewer.TRACE_LUT_SIZE),
            'lidar0': '#00EE00',
            'lidar1': '#FF0000',
            'p': 'blue',
        }
        tk_root = tk.Tk()

        self.closed = False

        self.tk_root = tk_root
        self.tk_root.title("시뮬레이터시뮬레이터")

        self.FRAME_DELAY = frame_delay  # 다음 프레임까지 대기 시간 (천분초)
        self.auto_update = auto_update

        self.viewMode = True

        # world
        self.world = world

        self.keys = {}

        # --- GUI 초기화 ---
        scale = scale if scale >= 1 else max(min(int(901/self.world.MAP_H), int(1201/self.world.MAP_W), 32), 1)
        self.scale = scale
        self.CANVAS_W, self.CANVAS_H = self.world.MAP_W * scale, self.world.MAP_H * scale
        self.canvas = tk.Canvas(tk_root, width=self.CANVAS_W, height=self.CANVAS_H, bg="white")
        self.canvas.pack()
        self.fcanvas = FieldCanvasPainter(self.canvas, self.world.MAP_W, self.world.MAP_H, scale)
        print(f'[WorldViewer] canvas:({self.CANVAS_W}, {self.CANVAS_H})  scale:{scale}')

        self.tk_root.bind("<KeyPress>",   self.key_press)
        self.tk_root.bind("<KeyRelease>", self.key_release)
        self.tk_root.protocol("WM_DELETE_WINDOW", self.close)  # 닫기버튼 동작

        if type(self) == WorldViewer:
            self.update()
            if self.auto_update:
                self.occupy_mainloop()

    def occupy_mainloop(self):
        if self.tk_root:
            self.auto_update = True
            self.update()
            self.tk_root.mainloop()

    def close(self):
        if self.tk_root:
            # Tkinter 윈도우가 열려 있으면 닫아줍니다.
            self.tk_root.destroy()
            self.tk_root.quit()
            self.tk_root = None
        if not self.closed:
            self.closed = True
            print('Viewer closed')

    def __del__(self):
        self.close()


    # 키보드 조작
    def key_press(self, event):
        key = event.keysym.lower()
        if key in self.keys:
            self.keys[key] = True
        else:
            self.key_press_etc(key)

    def key_release(self, event):
        key = event.keysym.lower()
        if key in self.keys:
            self.keys[key] = False

    def key_press_etc(self, key):
        if   key == 'tab':    self.viewMode     = not self.viewMode
        elif key == 'v':      self.fcanvas.mode = (self.fcanvas.mode+1) % 3


    def update(self):
        if self.closed: raise Exception('이미 닫힌 뷰어이다.')

        self.step()
        self.draw()

        if self.auto_update:
            self.tk_root.after(self.FRAME_DELAY, self.update)#type:ignore
        else:
            try:
                self.tk_root.update_idletasks()#type:ignore
                self.tk_root.update()#type:ignore
            except tk.TclError as e:
                self.closed = True
                print('tk.TclError로 뷰어 닫음.', e)
                self.close()

    def step(self):
        pass

    def draw(self):
        self._draw()

    def _draw(self):
        """
        그리기
        """

        self.canvas.delete("all")

        c = self.color

        world = self.world
        player = world.player
        px = player.x
        pz = player.z

        # 장애물
        for z in range(world.MAP_H):
            for x in range(world.MAP_W):
                if world.obstacle_map[z][x] == 1:
                    self.fcanvas.add_cell_filling(x, z, c['obstacle'])

        # 자취
        trace_points = world.trace
        for i in range(1, len(trace_points)):
            (p0x, p0z, _), (p1x, p1z, s) = trace_points[i-1], trace_points[i]
            idx = min(int(abs(s) * WorldViewer.TRACE_LUT_SIZE), WorldViewer.TRACE_LUT_SIZE)
            col = c['traceF_LUT'][idx] if s > 0 else c['traceB_LUT'][idx]
            self.fcanvas.add_line(p0x, p0z, p1x, p1z,
                           fill=col,
                           width=5)

        # 격자
        if self.fcanvas.mode == 2:
            x = int(player.x/10)*10
            z = int(player.z/10)*10
            for i in range(-10, 20):
                self.fcanvas.add_line(x+i, z-10, x+i, z+20, fill=c['grid1'])
            for i in range(-10, 20):
                self.fcanvas.add_line(x-10, z+i, x+20, z+i, fill=c['grid1'])
            for i in range(0, world.MAP_W + 1, 10):
              for k in range(0, world.MAP_H, 20):
                self.fcanvas.add_line(i, k, i, k+20, fill=c['grid2'])
            for i in range(0, world.MAP_H + 1, 10):
              for k in range(0, world.MAP_W, 20):
                self.fcanvas.add_line(k, i, k+20, i, fill=c['grid2'])
        else:
            if self.fcanvas.mode != 0 or self.fcanvas.scale >= 4:
                for i in range(world.MAP_W + 1):
                    self.fcanvas.add_line(i, 0, i, world.MAP_H, fill=c['grid1'])
                for i in range(world.MAP_H + 1):
                    self.fcanvas.add_line(0, i, world.MAP_W, i, fill=c['grid1'])
            for i in range(0, world.MAP_W + 1, 10):
                self.fcanvas.add_line(i, 0, i, world.MAP_H, fill=c['grid2'])
            for i in range(0, world.MAP_H + 1, 10):
                self.fcanvas.add_line(0, i, world.MAP_W, i, fill=c['grid2'])

        # 라이다
        for _, _, d, lx, _, lz, hit in world.lidar_points:
            temp = max(0.0, min(1.0, d / world.lidar.r))
            if hit: self.fcanvas.add_line(px, pz, lx, lz,
                                   fill=f'#{255:02X}{int(255*temp):02X}{int(255*temp):02X}',
                                   width=1)
            self.fcanvas.add_circle(lx, lz, 1.5 if hit else 1,
                            fill=c['lidar1'] if hit else c['lidar0'],
                            outline='')  # 감지점
        self.fcanvas.add_line(px, pz, world.lidar_points[0][3], world.lidar_points[0][5],
                       fill='blue', width=1)
        self.fcanvas.add_line(px, pz, world.lidar_points[-1][3], world.lidar_points[-1][5],
                       fill='purple', width=1)

        # 목표점
        waypoints = world.waypoints
        for i in range(1, len(waypoints)): # 선
            (p0x, p0z), (p1x, p1z) = waypoints[i-1], waypoints[i]
            self.fcanvas.add_line(p0x, p0z, p1x, p1z,
                           fill=c['wpoint'] if i>= world.waypoint_idx else c['wpoint_past'],
                           width=2)
        for i in range(world.waypoint_idx): # 점 - 통과
            gx, gz = waypoints[i]
            self.fcanvas.add_cell_stone(gx, gz,
                           c['wpoint_past'],
                           str(i+1), 'black')
        for i in range(world.path_len-1, world.waypoint_idx-1, -1): # 점 - 미통과
            gx, gz = waypoints[i]
            self.fcanvas.add_cell_stone(gx, gz,
                           c['wpoint'],
                           str(i+1), 'black')

        # 플레이어
        corners = player.get_corners()
        self.fcanvas.add_polygon(corners, fill="pink" if world.player_collision else "", outline="black")  # 충돌중이면 빨강
        self.fcanvas.add_line(corners[0][0], corners[0][1], corners[1][0], corners[1][1], fill=c['p'], width=3)  # 앞면
        self.fcanvas.add_circle(px, pz, 3, fill=c['p'], outline='')  # 중심

        self.fcanvas.draw(player)

        if self.viewMode:
            # 조작 상태
            BAR_W   = 4
            BAR_MAX = 40
            PANEL_X = 50
            PANEL_Y = self.CANVAS_H - 50
            controls = world.control_status
            ws_stop    = controls["moveWS"]["command"] == 'STOP' and world.use_stop
            ws_dir     = 1 if controls["moveWS"]["command"] == 'W' else -1
            ad_dir     = 1 if controls["moveAD"]["command"] == 'D' else -1
            ws_weight  = controls["moveWS"]["weight"] if not ws_stop else 0
            ad_weight  = controls["moveAD"]["weight"]
            # WS 막대
            self.canvas.create_rectangle(
                PANEL_X-BAR_W, PANEL_Y, PANEL_X+BAR_W+1, PANEL_Y - ws_weight*BAR_MAX*ws_dir,
                fill='#0088FF' if ws_dir==1 else '#EE2288', outline="", tags="control_status"
            )
            # STOP
            if ws_stop:
                self.canvas.create_oval(PANEL_X-10, PANEL_Y-10, PANEL_X+10, PANEL_Y+10, fill='red', outline='')
            # AD 막대
            self.canvas.create_rectangle(
                PANEL_X, PANEL_Y-BAR_W, PANEL_X + ad_weight*BAR_MAX*ad_dir, PANEL_Y+BAR_W+1,
                fill='#FFAA00' if ad_dir==1 else '#00CC00', outline="", tags="control_status"
            )
            # 가중치 글자
            self.canvas.create_rectangle(
                PANEL_X - BAR_W - 2,
                PANEL_Y + BAR_W + 2,
                PANEL_X - BAR_MAX/2 - 24,
                PANEL_Y + BAR_MAX/2 + 24,
                fill='white', outline="", tags="control_status"
            )
            self.canvas.create_text(PANEL_X - BAR_W - 4, PANEL_Y + BAR_MAX/2,
                                    anchor="e", justify="right",
                                    text=f'{ws_weight:.2f}\n{ad_weight:+.2f}',
                                    fill="black", font=("Consolas", 10))

            # 정보 표시
            status_text = \
                f"Speed: {player.speed:5.2f} m/s  {player.speed*3.600:5.2f} km/h\n"\
                f"Pos: {player.x:6.2f} / {player.z:6.2f}\n"\
                f"angle: {player.angle_x*rad_to_deg:+7.2f}°  {player.angle_x:+4.2f} rad"
            self.canvas.create_text(3, 3, anchor="nw", text=status_text, fill="white", font=("Consolas", 10))
            self.canvas.create_text(5, 3, anchor="nw", text=status_text, fill="white", font=("Consolas", 10))
            self.canvas.create_text(3, 5, anchor="nw", text=status_text, fill="white", font=("Consolas", 10))
            self.canvas.create_text(5, 5, anchor="nw", text=status_text, fill="white", font=("Consolas", 10))
            self.canvas.create_text(5, 6, anchor="nw", text=status_text, fill="white", font=("Consolas", 10))
            self.canvas.create_text(4, 4, anchor="nw", text=status_text, fill="black", font=("Consolas", 10))
            timeLabel = tk.Label(self.canvas,
                             text=f"Time: {world.t_acc/1000:.2f} s",
                             bg="black", fg="white", font=("Consolas", 10))
            self.canvas.create_window(self.CANVAS_W/2, 0, anchor="n", window=timeLabel)
 
            # max_risk = max([min(1/(d+1e-6)*100, 100) for _, _, d, _,_,_,_ in world.lidar_points]) # 현재 스텝의 최대 위험도
            # print(f'max_risk: {max_risk:.2f}')
            text = f'목표({world.waypoint_idx}/{len(world.waypoints)}): {int(world.get_relative_angle_to_wpoint()*rad_to_deg)}° / {world.get_distance_to_wpoint():.1f}m  | 근접장애물 {int(world.obs_nearest_angle*rad_to_deg)}° / {world.obs_nearest_distance:.1f}m'
            self.canvas.create_text(self.CANVAS_W/2, self.CANVAS_H, anchor="s",
                                    text=text, fill="black", font=("Consolas", 10))



class WorldController(WorldViewer):
    """
    World의 모습을 화면으로 보여준다.
    시간경과와 키보드조작으로 World를 동작시킨다.
    Tank Challenge의 키보드 조작과 API 호출 기능을 최대한 따라함.
    """
    def __init__(self, world:World=World(), scale:int=0, config={}, *,
                 url_base='http://127.0.0.1:5000',
                 frame_delay=30,
                 use_real_time=False,  # 스텝 사이 실제 흐른 시간량 사용 / 스텝별 동일 시간량(고정값) 부여
                 time_accel=1          # 배속
        ):
        super().__init__(world, scale, frame_delay)

        infotext = "ESC: 재시작 | Tab: 상태표시 | F2: Tracking Mode 토글 | F3: Log Mode 토글 | F5: 일시정지"
        self.info_label = tk.Label(self.tk_root, text=infotext)
        self.info_label.pack()

        self.trackingMode = config.get('TrackingMode', False)  # T: 서버로 get_action 호출해서 조작. F:키보드조작 가능.
        self.logMode      = config.get('LogMode',      False)  # /info로 정보 보낼지
        self.pause = False     # 일시정지
        self.color['p']      = 'red' if self.trackingMode else 'black'
        self.color['lidar0'] = '#00EE00' if self.logMode else ''

        # world
        self.world_init:World = self.world
        self.world:World = world.clone()

        # 시간
        if time_accel < 1: raise ValueError('time_accel < 1')
        self.timeaccel = time_accel
        self.t_last = 0  # 지난 스텝 시각 (천분초)
        self.t_acc = 0   # 에피소드 경과시간 (천분초)
        self.dt = 0      # 지난스텝~이번스텝 경과시간 (천분초)
        self.t0 = 0      # 에피소드 시작시각 (천분초)
        self.use_real_time = use_real_time
        self.init_time()

        # 서버 API
        self.API_INIT = url_base + '/init'
        self.API_INFO = url_base + '/info'
        self.API_DEST = url_base + '/set_destination'
        self.API_ACT  = url_base + '/get_action'
        self.API_CALL_DELAY = int(config.get('api_delay', 1000) / self.timeaccel)
        if self.API_CALL_DELAY <= 1: self.API_CALL_DELAY = 1
        self.t_last_api = 0

        self.reset_by_server(msg=False)

        self.keys.update({key: False for key in ['w', 's', 'a', 'd']})
        self.canvas.bind("<Button-3>", self.mouse_click_right)

        self.draw_count = 0

        if type(self) == WorldController:
            self.update()
            if self.auto_update:
                self.occupy_mainloop()


    # 마우스 조작
    def mouse_click_right(self, event):
        """
        마우스 우클릭: 목표점 추가/삭제
        """
        if self.fcanvas.mode == 0:
            # 화면 좌표 --> 월드 좌표
            x = event.x / self.scale
            y = event.y / self.scale
            world_x = int(round(x))
            world_z = int(round(self.world.MAP_H - y))

            # 시프트 누른 채 클릭: 유일한 목표점으로 설정, API 호출
            if (event.state & 0x0001) != 0:  # 시프트키 누름
                self.world.waypoints = [(world_x, world_z)]
                self.call_api_dest(world_x, world_z)

            # 컨트롤키 누른 채 클릭: 목표점 삭제
            elif (event.state & 0x0004) != 0:  # 컨트롤키 누름
                # 클릭한 위치 근처의 목표점을 찾아 제거
                DELETE_THRESHOLD = 30.0  # 클릭 오차 허용 범위 (픽셀)
                target_idx = -1

                # 가장 가까운 점 찾기
                min_dist = float('inf')
                for i, (gx, gz) in enumerate(self.world.waypoints):
                    dist = distance_of(gx, gz, world_x, world_z) * self.scale

                    if dist < DELETE_THRESHOLD and dist < min_dist:
                        min_dist = dist
                        target_idx = i

                # 찾았으면 삭제
                if target_idx != -1:
                    removed_pt = self.world.waypoints.pop(target_idx)

                    # 현재 목표 인덱스가 삭제된 인덱스보다 뒤에 있었다면 당겨주기.
                    if self.world.waypoint_idx > target_idx:
                        self.world.waypoint_idx -= 1

            else:  # 목표점 추가
                self.world.waypoints.append((world_x, world_z))


    def reset(self):
        self.world = self.world_init.clone()
        self.init_time()

    def reset_by_server(self, msg=True):
        """
        /init 호출, 응답값 반영
        응답 없으면 일반 reset
        """
        res = api_get(self.API_INIT)
        if res:
            self.world.player.x = res['blStartX']
            self.world.player.z = res['blStartZ']
            self.world.player.speed   = 0.0
            self.world.player.angle_x = 0.0
            self.trackingMode = res['trackingMode']
            self.logMode      = res['logMode']
        else:
            self.reset()
            if msg:
                self.info_label.destroy()
                self.info_label = tk.Label(self.tk_root, text='서버 응답이 없어 내부 초기값으로 초기화함.')
                self.info_label.pack()
        self.init_time()

    def init_time(self):
        self.t0     :int= get_curt()
        self.t_last :int= self.t0
        self.dt     :int= 0


    def step(self):
        """
        1 프레임 진행
        """
        if self.pause: return

        WorldViewer.step(self)

        cur_time = get_curt()
        self.update_time(cur_time)

        if not self.trackingMode:
            self.set_control_status_by_keyboard()

        self.world.step(self.dt)

        if cur_time - self.t_last_api >= self.API_CALL_DELAY:
            self.t_last_api = cur_time
            if self.logMode:      self.call_api_info()
            if self.trackingMode: self.call_api_act()

    def update_time(self, cur_time):
        """
        시간경과량 업데이트
        """
        if self.use_real_time:
            self.dt = (cur_time - self.t_last)
            if self.dt > self.FRAME_DELAY*2: self.dt = self.FRAME_DELAY*2  # 프로그램이 한참 멈춰도 dt가 너무 안 커지게 제한
        else:
            self.dt = self.FRAME_DELAY

        self.t_last = cur_time

    def call_api_info(self):
        """
        /info 호출
        """
        data = self.world.info
        threading.Thread(target=api_post, args=(self.API_INFO, data), daemon=True).start()

    def call_api_dest(self, x, z):
        """
        /set_destination 호출
        API Docs의 Body 예시:
        {
            "destination": "100.0,0.0,250.0"
        }
        """
        data = {
            "destination": f"{x:.1f},0.0,{z:.1f}"
        }

        def _request_and_update():
            response = api_post(self.API_DEST, data) 
            if response and response.get("status") == "OK":
                received_path = response.get("path", None)
                if received_path:
                    self.world.waypoints = received_path
                    print(f"[Info] Way points updated. Total points: {len(self.world.waypoints)}")
                else:
                    print("[Error] Invalid path format received.", received_path)
            else:
                print("[Error] Failed to set destination or invalid response.")

        threading.Thread(target=_request_and_update, daemon=True).start()

    def call_api_act(self):
        """
        /get_action 호출, 응답값으로 조작상태 업데이트
        """
        wi = self.world.info
        data = {
            "position": wi['playerPos'],
            "turret": {# XXX
                "x": wi['playerPos']['x'],
                "y": wi['playerPos']['y'] + 2.5,
                "z": wi['playerPos']['z'],
            }
        }
        threading.Thread(target=self.call_api_act_, args=(data,), daemon=True).start()

    def call_api_act_(self, req_data):
        data = api_post(self.API_ACT, req_data)
        if data:
            self.world.control_status = data


    # 키보드 조작
    def key_press_etc(self, key):
        WorldViewer.key_press_etc(self, key)
        if key == 'escape': self.reset_by_server()
        elif key == 'f2':     self.trackingMode = not self.trackingMode ; self.color['p'] = 'red' if self.trackingMode else 'black'
        elif key == 'f3':     self.logMode      = not self.logMode      ; self.color['lidar0'] = '#00EE00' if self.logMode else ''
        elif key == 'f5':     self.pause        = not self.pause

    def set_control_status_by_keyboard(self):
        """
        키보드로 조작상태 업데이트
        """
        if self.keys['w']:
            self.world.control_status['moveWS']['command'] = 'W'
            self.world.control_status['moveWS']['weight'] = 1
        elif self.keys['s']:
            self.world.control_status['moveWS']['command'] = 'S'
            self.world.control_status['moveWS']['weight'] = 1
        else:
            self.world.control_status['moveWS']['command'] = 'STOP'
            self.world.control_status['moveWS']['weight'] = 0

        if self.keys['a']:
            self.world.control_status['moveAD']['command'] = 'A'
            self.world.control_status['moveAD']['weight'] = 1
        elif self.keys['d']:
            self.world.control_status['moveAD']['command'] = 'D'
            self.world.control_status['moveAD']['weight'] = 1
        else:
            self.world.control_status['moveAD']['command'] = ''
            self.world.control_status['moveAD']['weight'] = 0


    def draw(self):
        # 그리기 주기는 시뮬레이션의 가속과 상관없이 유지: n배속이면 n스텝마다 한 번 그림.
        self.draw_count += 1
        if self.draw_count >= self.timeaccel:
            self._draw()
            self.draw_count = 0

    def _draw(self):
        WorldViewer._draw(self)
        text = f'tracking: {self.trackingMode}\n'\
               f'log: {self.logMode}'
        self.canvas.create_text(self.CANVAS_W, self.CANVAS_H/2, anchor="e",
                                text=text, fill="black", font=("Consolas", 10), justify='right')



def create_sample():

    player = Car({
        'playerPos': {'x': 10, 'z': 10},
        "playerSpeed": 0,
        "playerBodyX": 270 * rad_to_deg,
        "playerBodyY": 0,
        "playerBodyZ": 0,
        "playerHealth": 0,
        "playerTurretX": 0,
        "playerTurretY": 0,
    })

    # obstacle_map, w, h = load_obstacle_map('./map-50.txt')

    waypoints = [(10, 100), (100, 150), (150, 100), (200, 200), (250, 250)]
    waypoints = []

    world = World(
        player=player,
        wh=(MAP_W, MAP_H),
    #   obstacle_map=obstacle_map,
        waypoints=waypoints,
        config={
            'lidar_range': 20,
            'lidar_raynum': 10,
            'angle_start': -pi/4,
            'angle_end': pi/4,
            'use_stop': True,
            'map_border': False,
    })

    return WorldController(
        world,
        time_accel=1,
        use_real_time=False,
        frame_delay=33,
        config={
            'TrackingMode': False,
            'LogMode': True,
            'api_delay': 1000
        })

if __name__ == "__main__":
    app = create_sample()
