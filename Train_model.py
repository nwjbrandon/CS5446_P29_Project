import gym
from gym_warehouse.envs import WarehouseEnv
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.callbacks import CheckpointCallback, CallbackList
import pathlib
#For diagnostic and _call_plotting
from Utils.CostumCallBacks import PlotCallback as plotcallback
import time
#to avoid CPU log info
import os
os.environ['KMP_WARNINGS'] = '0'


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128,64,32],
                                           layer_norm=True,
                                           feature_extraction="mlp")

prefix = 'Test_1'
total_timesteps = 1000000

pathlib.Path("./models").mkdir(exist_ok=True)
pathlib.Path("./models/checkpoints").mkdir(exist_ok=True)
env = WarehouseEnv('6x5_4bins_1item_1slot')
# env = WarehouseEnv('7x7_4bins_2items_2binslots_1agentslots')


model = DQN(CustomDQNPolicy, env, verbose=1, exploration_fraction=0.99, exploration_initial_eps=1, exploration_final_eps=0.1, batch_size=32, buffer_size=50000)

checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/checkpoints/', name_prefix=prefix)

#episode_plot_freq = n : Update plots every n time steps
#update_stats_every = m: Update stats used in plots every m Episodes
#Note! update_stats_every > 1 would lead to lose of information in the plot (not in the trining process), but increase the performance during training.
plt_callback = plotcallback(episode_plot_freq=10000, update_stats_every=1, average_size=100, verbose=1, plot_prefix=prefix, plot_dir="./Plots")

callbacks = CallbackList([checkpoint_callback])

model.learn(total_timesteps=total_timesteps, callback=callbacks)
model.save("./models/"+prefix)

#model = DQN.load("./models/7x7_4bins_2items_2binslots_1agentslots_128x64x32_1500k.zip")
# model = DQN.load("./models/Test_1.zip")
# Test the trained agent
obs = env.reset()
#env.init_video()
n_steps = 300
total =0
for step in range(n_steps):
    env.render(mode='human', sec_per_frame=0.3)
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    #time.sleep(60)
    total += reward
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        env.reset(testing = True)
        #break
#env.save_video('400k_2items_2slots2')
env.save_as_gif(filename_prefix=prefix, save_dir="./", sec_per_frame=0.6)
env.save_as_mp4(filename_prefix=prefix, save_dir="./", sec_per_frame=0.3)
print("TOTAL: " + str(total))
