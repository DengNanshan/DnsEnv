import gym
# import tkinter as tk
import highway_env
import matplotlib
from matplotlib import pyplot as plt
import datetime
import time

# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.sac.policies import SACPolicy,MlpPolicy
from stable_baselines import DQN,SAC


from stable_baselines.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 2,  # !!!!!!!!!!!!
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features": ["x","y","vx", "vy", "cos_h"],

        "features_range": {
            "x": [-100, 100],
            "y": [-15, 15],
            "vx": [-30, 30],
            "vy": [-30, 30],

        },

        "see_behind": False,
        "observe_intentions": True,
        "absolute": False,
        "order": "sorted"

    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 2,
    "initial_lane_id": None,
    "vehicles_count": 5,                # ! !!!!!!!!!!!
    "controlled_vehicles": 1,
    "duration": 500,  # [step]             # !!!!!!!!!!!!!!
    "ego_spacing": 2,
    "initial_spacing": 2,
    "collision_reward": -10,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [0, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
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
    "vehicles_density": 1,
    "offroad_terminal": False,
    "reward_range_low": -10,  # dns range
    "reward_range_high": 1,  # dns range
    "lane_reward": -0.1,  # dns_add
    "heading_reward": -1 # dns_add
}




'''
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }



'''

env = gym.make('highway-v0')
env.configure(config)

env.reset()

#
#
# model= SAC(MlpPolicy,env,verbose=1,
#            tensorboard_log="../../Data3/tensorboard_log_fello/",
#            learning_rate=0.004,
#            learning_starts=100,
#            gamma=0.7,
#            batch_size=64)
#
# # model=SAC(MlpPolicy,env=env)
# timetemp=datetime.datetime.now().strftime("SAC_16_%Y_%m_%d_%H_%M_%S")
# checkpoint_callback=CheckpointCallback(save_freq=10000,
#                                        save_path='../../Data3/'+timetemp,
#                                        name_prefix='SAC_highway_check',
#                                        verbose=1)
#
# E=EvalCallback(eval_env=env,eval_freq=10000,log_path='../../Data3/'+timetemp,best_model_save_path='../../Data3/'+timetemp)
#
# callbacks=CallbackList([checkpoint_callback,E])
# model.learn(1000000,callback=callbacks)
#
# model.save('../../Data3/SAC_16')
#
# del model






model=SAC.load('../../Data3/SAC_16',env)




ve=[]
obs=env.reset()
# ve.append(obs[0,2]*30)
i=0
begin = time.time()
dones = False

# obs,reward,dones,info=env.step(4)




for i in range(500):
    print(obs)
    # action=env.action_space
    action=model.predict(obs)
    print(action)
    obs,reward,dones,info=env.step(action[0])
    print(obs,reward,dones,info)


    # print(obs,reward,dones,info)
    env.render()
    end=time.time()
    # time.sleep(0.1)
    # if end-begin<1:
    #     time.sleep(1-end+begin)
    # begin=end

plt.plot(ve)
plt.show()



