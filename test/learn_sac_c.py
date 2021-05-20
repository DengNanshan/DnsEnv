import gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("Pendulum-v0")
model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    print(action,_states)
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    env.render()
    if done:
      obs = env.reset()