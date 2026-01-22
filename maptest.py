import tkinter as tk

from drivingppo.world import create_empty_map
from drivingppo.simsim import WorldController, World, Car
from world_samples import gen_0, gen_11, gen_12, gen_inv, gen_env_naive
from world_samples import gen_21, gen_22, gen_23, gen_env_plain, gen_env_obs, generate_random_world_plain
from world_samples import generate_random_world_obs_matrix, generate_random_world_narrow, generate_random_world_obs_between, generate_world_square



if __name__ == "__main__":

    wh = 100, 100

    # player = Car({
    #     'playerPos': {'x': 10, 'z': 10}
    # })

    world = generate_world_square()

    app = WorldController(
        world,
        int(900/world.MAP_H),
        time_accel=2,
        use_real_time=False,
        frame_delay=33)
