import gym
import highway_env
env = gym.make('highway-v0')
env.reset()
from stable_baselines.deepq.policies import MlpPolicy

while(True):
    env.render()

