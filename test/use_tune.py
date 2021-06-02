import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import ray
from ray import tune
# env = gym.make('CartPole-v1')
#
# model = DQN(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=2500)
# model.save("deepq_cartpole")
#
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
#



stop={
    # "total_timesteps":1000
}

config={
    "lr": tune.grid_search([1e-3, 1e-5, 1e-6])
}

def training_function(config):

    for step in range(2):
        env = gym.make('CartPole-v1')
        model = DQN(MlpPolicy, env, verbose=1, learning_rate=config["lr"],learning_starts=1)
        model.learn(total_timesteps=10000)
        intermediate_score=model.episode_reward
        tune.report(mean_loss=intermediate_score)

analysis = tune.run(
    training_function,
    config=config)