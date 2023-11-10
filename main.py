"""
https://github.com/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/dqn_sb3.ipynb
"""

import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("MountainCar-v0", render_mode="rgb_array")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    seed=2,
)
model.learn(total_timesteps=100000, progress_bar=True)

vec_env = model.get_env()
for i in range(10):
    obs = vec_env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")

        if done:
            print(info)
            break
