import gym
# import tkinter as tk
import highway_env
import matplotlib
from matplotlib import pyplot as plt
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.sac.policies import SACPolicy,MlpPolicy
from stable_baselines import DQN,SAC
import torch as th
import datetime
import time



from stable_baselines.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
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
        "type": "ContinuousAction",
    },
    "lanes_count": 2,
    "initial_lane_id": None,
    "vehicles_count": 10,  # ! !!!!!!!!!!!
    "controlled_vehicles": 1,
    "duration": 500,  # [step]             # !!!!!!!!!!!!!!
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
    "policy_frequency": 10,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 900,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "vehicles_density": 0.5,
    "offroad_terminal": False,
    "reward_range_low": -10,  # dns range
    "reward_range_high": 1,  # dns range
    "lane_reward": 0.4,  # dns_add
    "lane_change_reward": -0.1,  # The reward received at each lane change action.
    "heading_reward": 0.4  # dns_add

}


env = gym.make('highway-v0')
env.configure(config)
env.reset()

timetemp=datetime.datetime.now().strftime("DQNtest_%Y_%m_%d_%H_%M_%S")
dir = ('../../Data0601/')
name=('SAC2')

model= SAC(MlpPolicy,env,verbose=1,
           tensorboard_log=dir+'tensorboard_log/',
           learning_rate=0.001,
           learning_starts=100,
           gamma=0.99,
           batch_size=64,
           buffer_size=100000,
           dns_render=True)

# model= SAC(MlpPolicy,env,verbose=1,
#            tensorboard_log="../../Data3/tensorboard_log_fello/",
#            learning_rate=0.1,
#            learning_starts=100,
#            gamma=0.7,
#            batch_size=64)

checkpoint_callback=CheckpointCallback(save_freq=10000,

                                       save_path=dir+name,
                                       name_prefix='deeq_highway_check',
                                       verbose=1)

E=EvalCallback(eval_env=env,eval_freq=1000,log_path=dir+name,best_model_save_path=dir+name)

callbacks=CallbackList([checkpoint_callback,E])
model.learn(100000,callback=callbacks)

model.save(dir+name)

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
#
# model=DQN.load((dir+'/DQN2'),env)
# # model=DQN.load(('../../Data/DQN_5_28_00'),env)
#
# obs=env.reset()
# i=0
# ve=[]
# begin = time.time()
# dones = False
# for i in range(1000):
#     model.action_probability(obs)
#     action, _state = model.predict(obs)
#     obss=model.action_probability(obs)
#     # print(obss.round(2),action)
#     print("Left",obss[0].round(2))
#     print("Idle", obss[1].round(2))
#     print("Right", obss[2].round(2))
#     print("Fast", obss[3].round(2))
#     print("Slow", obss[4].round(2))
#     print("action", action)
#     # print('action',action)
#     # print(action,_state)
#     obs,reward,dones,info=env.step(action.tolist())
#     print(obs.round(1))
#     # if dones:
#     #     env.reset()
#     ego_speed=obs[0,1]*30
#     ve.append(ego_speed)
#     f_speed=obs[1,1]*30+ego_speed
#     # print(ego_speed,f_speed)
#
#     # print(obs,reward,dones,info)
#     env.render()
#     end=time.time()
#     # time.sleep(0.1)
#     # if end-begin<1:
#     #     time.sleep(1-end+begin)
#     # begin=end
#
#
#
#
#
