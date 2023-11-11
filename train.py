import os
import pathlib

from stable_baselines.common.callbacks import CallbackList, CheckpointCallback

from config import model_config
from gym_warehouse.envs import WarehouseEnv

os.environ["KMP_WARNINGS"] = "0"

model_name = "dqn"
save_freq = 50000
ckpt_fpath = f"./models/{model_name}"
pathlib.Path(ckpt_fpath).mkdir(exist_ok=True)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq, save_path=ckpt_fpath, name_prefix=model_name
)
callbacks = CallbackList([checkpoint_callback])

# Create model and environment
RLModel = model_config[model_name]["model"]
total_timesteps = model_config[model_name]["total_timesteps"]
config = model_config[model_name]["config"]

env = WarehouseEnv("6x5_4bins_1item_1slot")
model = RLModel("MlpPolicy", env, verbose=1, **config)

# Train Model
model.learn(total_timesteps=total_timesteps, callback=callbacks)
model.save(f"{ckpt_fpath}/best")


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

env.save_as_gif(filename_prefix=model_name, save_dir=ckpt_fpath, sec_per_frame=0.6)
env.save_as_mp4(filename_prefix=model_name, save_dir=ckpt_fpath, sec_per_frame=0.3)
