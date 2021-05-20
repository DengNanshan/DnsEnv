import gym
import highway_env
from matplotlib import pyplot as plt
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN
# import tensorflow as tf
#
# from stable_baselines.common.callbacks import EvalCallback, CallbackList,CheckpointCallback
import datetime



config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,  # !!!!!!!!!!!!
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features": ["x", "y", "vx","vy", "cos_h", "sin_h"],

        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-30, 30],
            "vy": [-30, 30],
        },
        "absolute": False,
        "order": "sorted"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 2,
    "initial_lane_id": None,
    "vehicles_count": 50,                # ! !!!!!!!!!!!
    "controlled_vehicles": 1,
    "duration": 50,  # [step]             # !!!!!!!!!!!!!!
    "ego_spacing": 2,
    "initial_spacing": 2,
    "collision_reward": -1000,  # The reward received when colliding with a vehicle.
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



# model= DQN(MlpPolicy,env,verbose=1,
#            tensorboard_log="../Date/tensorboard_log/")
#
#
#
#
# timetemp=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# checkpoint_callback=CheckpointCallback(save_freq=100, save_path='../Data/'+timetemp,name_prefix='deeq_highway_check')
# callbacks=CallbackList([checkpoint_callback])
# model.learn(20000,callback=callbacks)
# model.save('../Data/hellohighway')
#
# del model
#
model=DQN.load(('../Data/hellohighway'),env)
obs=env.reset()


while (True):
    action, _state = model.predict(obs)
    obs,reward,dones,info=env.step(action)
    print(obs,reward,dones,info)
    env.render()



