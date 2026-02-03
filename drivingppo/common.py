import math, random
import numpy as np
import torch

SPEED_MAX_W:float = 19.44
SPD_MAX_STD = 10.0

# Tank Challenge의 맵 크기
MAP_W = 300
MAP_H = 300

# 설계 파라미터
LOOKAHEAD_POINTS = 4  # 경로 추종을 위한 앞의 N개 점
EACH_POINT_INFO_SIZE = 4
LIDAR_NUM   = 65
LIDAR_RANGE = 30
LIDAR_START = -math.pi
LIDAR_END   =  math.pi

# 상태
OBSERVATION_IND_SPD        = 0
OBSERVATION_IND_WPOINT_0   = 0 + 1
OBSERVATION_IND_WPOINT_1   = 0 + 1 + (EACH_POINT_INFO_SIZE * 1)
OBSERVATION_IND_WPOINT_2   = 0 + 1 + (EACH_POINT_INFO_SIZE * 2)
OBSERVATION_IND_WPOINT_E   = 0 + 1 + (EACH_POINT_INFO_SIZE * LOOKAHEAD_POINTS)

OBSERVATION_IND_LIDAR_DIS_S    = 0 + 1 + (EACH_POINT_INFO_SIZE * LOOKAHEAD_POINTS)
OBSERVATION_IND_LIDAR_DIS_E    = 0 + 1 + (EACH_POINT_INFO_SIZE * LOOKAHEAD_POINTS) + LIDAR_NUM
OBSERVATION_DIM_LIDAR          = LIDAR_NUM

OBSERVATION_DIM            = 0 + 1 + (EACH_POINT_INFO_SIZE * LOOKAHEAD_POINTS) + LIDAR_NUM


def set_seed(seed):
    """
    seed 설정. 맵은 고정되는 듯.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("SEED SET", seed)

# set_seed(0)
