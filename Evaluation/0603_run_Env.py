import gym
# import tkinter as tk
import highway_env

import matplotlib
from matplotlib import pyplot as plt
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import torch as th
import numpy as np
from stable_baselines.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
import datetime
import time
config = {
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
    "manual_control": False,
    "lane_change_reward": -0.1,  # The reward received at each lane change action.
    "heading_reward": 0  # dns_add
}

"""change log:"""
config["init_speed"] = [0.3, 0.4]
config["vehicles_density"] = 2
config["manual_control"] = False

# config["ego_spacing"] = 1




env = gym.make('highway-v0')
env.configure(config)
env.reset()

"""log models"""
model_path = {
    "dir": '../../Data0531/',   # train0531/DQN2 DQN3_model
    "dir2": '/home/be-happy/ray_results/Data0601/',  # train0601/DQN2
    "dir3": '/home/be-happy/ray_results/Data0601_2/' , # train0601/DQN3_model
    "dir4": '/home/be-happy/ray_results/Data0601_3/'  # train0601/DQN4
}

model_name = {
    "name1": model_path["dir"] + 'DQN2',  # single 07-08
    "name2": model_path["dir"] + 'DQN3',  # single 03-04  dens 0.5!!

    "name3": model_path["dir2"] + '[0.3, 0.4]DQN_2021_06_01_20_56_00',
    "name4": model_path["dir2"] + '[0.3, 0.5]DQN_2021_06_01_20_55_59',
    "name5": model_path["dir2"] + '[0.4, 0.5]DQN_2021_06_01_20_56_00',
    "name6": model_path["dir2"] + '[0.4, 0.6]DQN_2021_06_01_20_56_00',
    "name7": model_path["dir2"] + '[0.5, 0.6]DQN_2021_06_01_20_56_00',
    "name8": model_path["dir2"] + '[0.5, 0.8]DQN_2021_06_01_20_55_59',
    "name9": model_path["dir2"] + '[0.6, 0.7]DQN_2021_06_01_20_56_00',

    "name10": model_path["dir3"] + '[0.3, 0.4]DQN_2021_06_01_21_18_31',
    "name11": model_path["dir3"] + '[0.3, 0.7]DQN_2021_06_01_21_18_31',
    "name12": model_path["dir3"] + '[0.4, 0.8]DQN_2021_06_01_21_18_31',
    "name13": model_path["dir3"] + '[0.7, 0.8]DQN_2021_06_01_21_18_31',

    "name14": model_path["dir4"] + '[0.3, 0.4]DQN_1',
    "name15": model_path["dir4"] + '[0.3, 0.4]DQN_2',
    "name16": model_path["dir4"] + '[0.3, 0.4]DQN_3',
    "name17": model_path["dir4"] + '[0.3, 0.4]DQN_4',
    "name18": model_path["dir4"] + '[0.3, 0.4]DQN_5',
    "name19": model_path["dir4"] + '[0.3, 0.4]DQN_6',
    "name20": model_path["dir4"] + '[0.3, 0.4]DQN_7',
    "name21": model_path["dir4"] + '[0.3, 0.4]DQN_8',
    "name22": model_path["dir4"] + '[0.3, 0.4]DQN_9',

    "name23": model_path["dir4"] + '[0.7, 0.8]1',
    "name24": model_path["dir4"] + '[0.7, 0.8]2',
    "name25": model_path["dir4"] + '[0.7, 0.8]3',
    "name26": model_path["dir4"] + '[0.7, 0.8]4',
    "name27": model_path["dir4"] + '[0.7, 0.8]5',
    "name28": model_path["dir4"] + '[0.7, 0.8]6',
    "name29": model_path["dir4"] + '[0.7, 0.8]7',
    "name30": model_path["dir4"] + '[0.7, 0.8]8',
    "name31": model_path["dir4"] + '[0.7, 0.8]9',
}





# model=DQN.load(dir+name+'/best_model',env)
# model1 = DQN.load(model_name["name14"] , env)
# model2 = DQN.load(model_name["name15"] , env)
# model3 = DQN.load(model_name["name16"] , env)
# model4 = DQN.load(model_name["name17"] , env)
# model5 = DQN.load(model_name["name18"] , env)
# model6 = DQN.load(model_name["name19"] , env)
# model7 = DQN.load(model_name["name20"] , env)
# model8 = DQN.load(model_name["name21"] , env)
# model9 = DQN.load(model_name["name22"] , env)

model1 = DQN.load(model_name["name3"] , env)

# model=DQN.load(('../../Data/DQN_5_28_00'),env)

obs=env.reset()



i=0
ve=[]
begin = time.time()
dones = False
model=model1
# while not dones:
for i in range(1000):
    model.action_probability(obs)
    action, _state = model.predict(obs)
    # obss = model.action_probability(obs)
    print(obs.round(3))
    # print([env.action_type.ACTIONS_ALL[i] for i in range(5)])
    print(i,model.action_probability(obs).round(2))
    """
    print model
    """
    # print(i, model2.action_probability(obs).round(2))
    # print(i, model3.action_probability(obs).round(2))
    # print(i, model4.action_probability(obs).round(2))
    # print(i, model5.action_probability(obs).round(2))
    # print(i, model6.action_probability(obs).round(2))
    # print(i, model7.action_probability(obs).round(2))
    # print(i, model8.action_probability(obs).round(2))
    # print(i, model9.action_probability(obs).round(2))
    print("action", env.action_type.ACTIONS_ALL[action])
    model.action_probability(obs)
    # obs, reward, dones, info = env.step(action.tolist())
    obs, reward, dones, info = env.step(1)
    ego_speed = obs[0,1] * 30
    ve.append(ego_speed)
    f_speed=obs[1,1]*30+ego_speed
    # print(ego_speed,f_speed)

    # print(obs,reward,dones,info)
    env.render()
    end=time.time()
    # time.sleep(0.1)
    # if end-begin<1:
    #     time.sleep(1-end+begin)
    # begin=end