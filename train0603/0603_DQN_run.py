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
    "lane_change_reward": -0.1,  # The reward received at each lane change action.
    "heading_reward": 0  # dns_add

}

"""
change log:

"""
config["init_speed"]=[0.3,0.4]
config["vehicles_density"]=2
gamma = 0.9

env = gym.make('highway-v0')
env.configure(config)
env.reset()

timetemp=datetime.datetime.now().strftime("DQNtest_%Y_%m_%d_%H_%M_%S")
dir = '../../Data0603/'
name='DQN3.zip'

model = DQN.load(dir+name, env)


'''
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
'''


i=0
ve=[]
begin = time.time()
dones = False
obs=env.reset()
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
    obs, reward, dones, info = env.step(action.tolist())
    # obs, reward, dones, info = env.step(1)
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

