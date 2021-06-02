import gym
# import tkinter as tk
import highway_env
import matplotlib
from matplotlib import pyplot as plt
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import torch as th

from stable_baselines.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
import datetime
import time
import ray
from ray import tune
import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch


config={
    # "init_speed": tune.grid_search([[0.3,0.4],[0.4,0.5]])
    "init_speed": tune.grid_search([[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.3,0.5],[0.4,0.6],[0.5,0.8]])
}
config_env = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 7,  # !!!!!!!!!!!! not usefull
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features": ["presence","x", "y", "vx","vy"],

        "features_range": {
            "x": [-100, 100],
            "y": [-15, 15],
            "vx": [-30, 30],
            "vy": [-30, 30],

        },

        "see_behind": True,
        "observe_intentions": True,
        "absolute": False,
        "order": "sorted"

    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "initial_lane_id": None,
    "vehicles_count": 30,  # ! !!!!!!!!!!!
    "controlled_vehicles": 1,
    "duration": 50,  # [step]             # !!!!!!!!!!!!!!
    "ego_spacing": 2,
    "initial_spacing": 2,
    "collision_reward": -10,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [0, 30],
    # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "simulation_frequency": 10,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 900,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "vehicles_density": 1,
    "offroad_terminal": False,
    "reward_range_low": -10,  # dns range
    "reward_range_high": 1,  # dns range
    "lane_reward": 0,  # dns_add
    "lane_change_reward": -0.1,  # The reward received at each lane change action.
    "heading_reward": 0,  # dns_add
    "init_speed": [0.7,0.8]
}

"""
change log:

"""


def training_function(config):
    import highway_env.envs
    for step in range(1):
        config_env={
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 7,  # !!!!!!!!!!!! not usefull
                # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features": ["presence","x", "y", "vx","vy"],

                "features_range": {
                    "x": [-100, 100],
                    "y": [-15, 15],
                    "vx": [-30, 30],
                    "vy": [-30, 30],

                },

                "see_behind": True,
                "observe_intentions": True,
                "absolute": False,
                "order": "sorted"

            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "initial_lane_id": None,
            "vehicles_count": 30,  # ! !!!!!!!!!!!
            "controlled_vehicles": 1,
            "duration": 50,  # [step]             # !!!!!!!!!!!!!!
            "ego_spacing": 2,
            "initial_spacing": 2,
            "collision_reward": -10,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [0, 30],
            # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
            "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 900,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False,
            "vehicles_density": 1,
            "offroad_terminal": False,
            "reward_range_low": -10,  # dns range
            "reward_range_high": 1,  # dns range
            "lane_reward": 0,  # dns_add
            "lane_change_reward": -0.1,  # The reward received at each lane change action.
            "heading_reward": 0,  # dns_add
            "init_speed": [0.7,0.8]
        }


        config["init_speed"]=config["init_speed"]

        env = gym.make('highway-v0')
        env.configure(config_env)
        env.reset()

        timetemp = datetime.datetime.now().strftime("DQN_%Y_%m_%d_%H_%M_%S")
        dir = ('../../Data0601/')

        model = DQN(MlpPolicy, env, verbose=1,
                    tensorboard_log=dir + 'tensorboard_log/'+str(config["init_speed"])+'/',
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    learning_rate=0.001,
                    learning_starts=1,
                    gamma=0.9,
                    buffer_size=100000,
                    dns_render=False)
        checkpoint_callback = CheckpointCallback(save_freq=1000,
                                                 save_path=dir + timetemp,
                                                 name_prefix='deeq_highway_check',
                                                 verbose=1)
        E = EvalCallback(eval_env=env, eval_freq=1000, log_path=dir + str(config["init_speed"])+'/',
                         best_model_save_path=dir + str(config["init_speed"])+'/')
        callbacks = CallbackList([checkpoint_callback, E])
        model.learn(100000,callback=callbacks)
        model.save(dir + str(config["init_speed"]) + timetemp)
        intermediate_score=model.dns_episode_reward
        tune.report(score=intermediate_score)






analysis = tune.run(
    training_function,
    config=config,
    resources_per_trial={"cpu": 5, "gpu":0.1})

