import os
import pathlib

from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback

from gym_warehouse.envs import WarehouseEnv

os.environ["KMP_WARNINGS"] = "0"

model_name = "dqn"
ckpt_fpath = f"./models/{model_name}"
prefix = f"{model_name}"
total_timesteps = 250000
save_freq = 50000

pathlib.Path(ckpt_fpath).mkdir(exist_ok=True)


checkpoint_callback = CheckpointCallback(
    save_freq=save_freq, save_path=ckpt_fpath, name_prefix=prefix
)
callbacks = CallbackList([checkpoint_callback])

env = WarehouseEnv("6x5_4bins_1item_1slot")
model = DQN("MlpPolicy", env, verbose=1)


model.learn(total_timesteps=total_timesteps, callback=callbacks)
model.save("./models/" + prefix)


obs = env.reset()
n_steps = 300
total = 0
for step in range(n_steps):
    env.render(mode="human", sec_per_frame=0.3)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total += reward
    if done:
        print("Goal Reached! Reward: ", reward)
        env.reset(testing=True)

env.save_as_gif(filename_prefix=prefix, save_dir="./", sec_per_frame=0.6)
env.save_as_mp4(filename_prefix=prefix, save_dir="./", sec_per_frame=0.3)
