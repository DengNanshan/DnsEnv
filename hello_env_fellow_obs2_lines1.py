import gym
# import tkinter as tk
import highway_env
import matplotlib
from matplotlib import pyplot as plt
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import DQN
import torch as th

from stable_baselines3.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
import datetime



config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 2,
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features": ["x", "vx"],

        "features_range": {
            "x": [-100, 100],
            "vx": [-30, 30],
        },
        "absolute": False,
        "order": "sorted"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 1,
    "initial_lane_id": None,
    "vehicles_count": 1,
    "controlled_vehicles": 1,
    "duration": 50,  # [s]
    "ego_spacing": 2,
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [0, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 10,  # [Hz]
    "policy_frequency": 10,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "vehicles_density": 1,
    "offroad_terminal": False
}


env = gym.make('highway-v0')
env.configure(config)
env.reset()



model= DQN(MlpPolicy,env,verbose=1,
           tensorboard_log="../Data/tensorboard_log_fello/")




timetemp=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
checkpoint_callback=CheckpointCallback(save_freq=105550, save_path='../Data/'+timetemp,name_prefix='deeq_highway_check')
callbacks=CallbackList([checkpoint_callback])
model.learn(20,callback=callbacks)
model.save('../Data/hellohighway'+timetemp)

del model

'''
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }



'''


model=DQN.load(('../Data/hellohighway'+timetemp),env)
obs=env.reset()
i=0
ve=[]
for i in range(10):

    action, _state = model.predict(obs)
    action=int(action)
    print(i,action)
    # print(action,_state)
    print(type(action))
    obs,reward,dones,info=env.step(action)
    ego_speed=obs[0,1]*30
    ve.append(ego_speed)
    f_speed=obs[1,1]*30+ego_speed
    # print(ego_speed,f_speed)

    # print(obs,reward,dones,info)
    env.render()





