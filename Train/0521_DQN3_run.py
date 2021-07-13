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
        "vehicles_count": 5,  # !!!!!!!!!!!!
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features": ["x", "y", "vx","vy"],

        "features_range": {
            "x": [-100, 100],
            "y": [-10, 10],
            "vx": [-30, 30],
            "vy": [-30, 30],
        },
        "absolute": False,
        "order": "sorted"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "initial_lane_id": None,
    "vehicles_count": 10,                # ! !!!!!!!!!!!
    "controlled_vehicles": 1,
    "duration": 50,  # [step]             # !!!!!!!!!!!!!!
    "ego_spacing": 2,
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [0, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "simulation_frequency": 10,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "vehicles_density": 0.5,
    "offroad_terminal": False
}


env = gym.make('highway-v0')
env.configure(config)
env.reset()

#
#
# model= DQN(MlpPolicy,env,verbose=1,
#            tensorboard_log="../../Data2/tensorboard_log_fello/",
#            exploration_fraction= 0.1,
#            exploration_initial_eps = 1.0,
#            exploration_final_eps= 0.05,
#            learning_rate=0.01,
#            learning_starts=100,
#            gamma=0.9)
#
# timetemp=datetime.datetime.now().strftime("DQN3_model%Y_%m_%d_%H_%M_%S")
# checkpoint_callback=CheckpointCallback(save_freq=1000,
#                                        save_path='../../Data2/'+timetemp,
#                                        name_prefix='deeq_highway_check',
#                                        verbose=1)
#
# E=EvalCallback(eval_env=env,eval_freq=1000,log_path='../../Data2/'+timetemp,best_model_save_path='../../Data2/'+timetemp)
#
# callbacks=CallbackList([checkpoint_callback,E])
# model.learn(100000,callback=callbacks)
#
# model.save('../../Data2/DQN3_model')
#
# del model

'''
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }



'''


model=DQN.load(('../../Data2/DQN32021_05_21_00_25_22/best_model'),env)
obs=env.reset()
i=0
ve=[]
begin = time.time()
for i in range(1000):

    action, _state = model.predict(obs)
    action=int(action)
    # print('action',action)
    # print(action,_state)
    # print(type(action))
    obs,reward,dones,info=env.step(action)
    print('action',action,' ',reward)
    ego_speed=obs[0,1]*30
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





