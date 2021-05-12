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



plt.plot([1,2,3,4])
plt.show()