import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1,tensorboard_log="./test")
model.learn(total_timesteps=10000)
model.save("deepq_cartpole")

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_cartpole")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#
#     print(_states)
#     obs, rewards, dones, info = env.step(action)
#     env.render()