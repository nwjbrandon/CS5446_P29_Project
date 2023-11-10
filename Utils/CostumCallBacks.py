import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines.common.callbacks import BaseCallback

matplotlib.use("TkAgg")
import os

import pandas as p

# To avoid training crash due to the plots
# https://github.com/openai/spinningup/issues/16
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# also you may need to install:
# conda install nomkl
import sys
import time


class PlotCallback(BaseCallback):
    """
    Plot Reward, epsilon (from epsilon-greedy alg) and episode time steps per episode
    """

    def __init__(
        self,
        episode_plot_freq: int = 10000,
        update_stats_every: int = 1,
        average_size: int = 100,
        verbose=1,
        plot_prefix="quick_test",
        plot_dir="./Plots",
    ):
        """
        Init Pandas Data frame to store training information:
            Episode Time step,
            Number of times steps in episode
            Total reward per Episode
            Epsilon (from epsilon-greedy alg) value at the end of the episode

        Input:
            -episode_plot_freq = n : Update plots every n time steps
            -update_stats_every = n: Update stats used in plots every n Episodes
            Note! update_stats_every > 1 would lead to lose of information in the plot (not in the trining process), but increase the performance during training.
            -average_size = n: Moving average computed with last n seved episode stats
            -plot_prefix: String to add at the begining of every object saved in disk
            -plot_dir: relative path where plots will be saved
        """
        super(PlotCallback, self).__init__(verbose)

        pathlib.Path(plot_dir).mkdir(exist_ok=True)
        self.episode_plot_freq = episode_plot_freq
        self.update_stats_every = update_stats_every
        self.episode_counter = -1
        self.episode_reward = 0
        self.episode_timestep = 0
        self.episode_num = 0
        self.episode_rewardCuriosity = 0

        # To allow interactive plot (no pause on every show)
        plt.ion()
        # mean of n episodes
        self.average_size = average_size
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix

        # This flags avoid repeating plot legend
        self.legend_flag = True

    def _on_training_start(self) -> bool:
        """Init the pandas data frame and plots. This is not done in __init__ since we dont have still the full environment information there (i.e. self.training_env)"""

        self._init_plot_epsilon()
        self._init_plot_reward()
        self._init_plot_time_steps()

        if self.training_env.curiosity:
            column_name = [
                "Episode",
                "Time_step",
                "Episode_timesteps",
                "Mean_Episode_timesteps",
                "Reward",
                "Mean_Reward",
                "CuriosityReward",
                "Mean_curiosity_reward",
                "Eps_greedy_val",
            ]
            self._init_plot_reward_curiosity()
        else:
            column_name = [
                "Episode",
                "Time_step",
                "Episode_timesteps",
                "Mean_Episode_timesteps",
                "Reward",
                "Mean_Reward",
                "Eps_greedy_val",
            ]

        # init Pandas DF
        self.episode_history_df = p.DataFrame(columns=column_name)

        return True

    def _on_step(self) -> bool:

        self.episode_reward += self.training_env.reward
        self.episode_rewardCuriosity += self.training_env.rewardCuriosity
        self.episode_timestep += 1

        # To update stats
        if self.training_env.done:

            self.episode_counter += 1
            self.episode_num += 1
            # Update stats evere update_stats_every episodes or when first time done
            if (self.episode_counter == self.update_stats_every) or (
                self.episode_counter == 0
            ):
                self._update_df()
                self.episode_counter = 0

            self.episode_reward = 0
            self.episode_timestep = 0
            self.episode_rewardCuriosity = 0

        # Update plots every episode_plot_freq episodes
        if (self.model.num_timesteps % self.episode_plot_freq) == 0:

            N = self.episode_history_df.shape[0]
            # Show plots only if we have some stats
            if N > 0:
                self._plot_epsilon(self.episode_history_df)
                self._plot_reward(self.episode_history_df, N)
                self._plot_time_steps(self.episode_history_df, N)
                if self.training_env.curiosity:
                    self._plot_reward_curiosity(self.episode_history_df, N)
                plt.pause(0.0001)
                plt.show()

                # To avoid plot legend repetition
                self.legend_flag = False

                self.episode_history_df.to_csv(
                    self.plot_dir + "/" + self.plot_prefix + "_plot_data.out",
                    index=False,
                )
            else:
                print("Stats DataFrame empty, skipping plotting...")

        return True

    def _update_df(self):
        """
        Update Pandas data frame containing episode statistics
        """

        current_eps = self.model.exploration.value(self.model.num_timesteps)
        n_mean = self.average_size - 1

        # create mean values
        mean_reward = np.append(
            np.asarray(self.episode_history_df.Reward[-n_mean:]), self.episode_reward
        ).mean()
        mean_ts = np.append(
            np.asarray(self.episode_history_df.Episode_timesteps[-n_mean:]),
            self.episode_timestep,
        ).mean()

        if self.training_env.curiosity:
            mean_reward_curiosity = np.append(
                np.asarray(self.episode_history_df.CuriosityReward[-n_mean:]),
                self.episode_rewardCuriosity,
            ).mean()

            # Update df
            self.episode_history_df = self.episode_history_df.append(
                {
                    "Episode": self.episode_num,
                    "Time_step": self.model.num_timesteps,
                    "Episode_timesteps": self.episode_timestep,
                    "Reward": self.episode_reward,
                    "CuriosityReward": self.episode_rewardCuriosity,
                    "Eps_greedy_val": current_eps,
                    "Mean_Reward": mean_reward,
                    "Mean_curiosity_reward": mean_reward_curiosity,
                    "Mean_Episode_timesteps": mean_ts,
                },
                ignore_index=True,
            )

            # Give format for memory efficiency
            self.episode_history_df["CuriosityReward"] = self.episode_history_df[
                "CuriosityReward"
            ].astype("float")
            self.episode_history_df["Mean_curiosity_reward"] = self.episode_history_df[
                "Mean_curiosity_reward"
            ].astype("float")
        else:  # update df with no curiosity reward
            self.episode_history_df = self.episode_history_df.append(
                {
                    "Episode": self.episode_num,
                    "Time_step": self.model.num_timesteps,
                    "Episode_timesteps": self.episode_timestep,
                    "Reward": self.episode_reward,
                    "Eps_greedy_val": current_eps,
                    "Mean_Reward": mean_reward,
                    "Mean_Episode_timesteps": mean_ts,
                },
                ignore_index=True,
            )

        # Give format for memory efficiency
        self.episode_history_df["Episode"] = self.episode_history_df["Episode"].astype(
            "uint32"
        )
        self.episode_history_df["Time_step"] = self.episode_history_df[
            "Time_step"
        ].astype("uint32")
        self.episode_history_df["Episode_timesteps"] = self.episode_history_df[
            "Episode_timesteps"
        ].astype("uint16")
        self.episode_history_df["Reward"] = self.episode_history_df["Reward"].astype(
            "float"
        )
        self.episode_history_df["Eps_greedy_val"] = self.episode_history_df[
            "Eps_greedy_val"
        ].astype("float")
        self.episode_history_df["Mean_Reward"] = self.episode_history_df[
            "Mean_Reward"
        ].astype("float")
        self.episode_history_df["Mean_Episode_timesteps"] = self.episode_history_df[
            "Mean_Episode_timesteps"
        ].astype("float")

    def _plot_reward(self, df1, N):

        # The "if" is to avoid repeating plot legend
        self.ax_reward.plot(
            df1.Episode,
            df1.Reward,
            color="blue",
            linewidth=0.7,
            label="Reward" if self.legend_flag else "",
        )
        # self.ax_reward.plot(df1.Episode, df1.CuriosityReward, color="green", linewidth=.7, label='CuriosityReward' if self.legend_flag else "")
        # plot dommy plot just to show Time step info in x axis
        self.ax_reward_2.plot(df1.Time_step, df1.Reward, linestyle="None")

        # Plot average of the last self.average_size observations
        self.ax_reward.plot(
            df1.Episode,
            df1.Mean_Reward,
            color="red",
            linewidth=0.7,
            label=str(self.average_size) + " Rewar avg" if self.legend_flag else "",
        )
        self.ax_reward.legend(bbox_to_anchor=(0, 1), loc="upper left")

        self.fig_reward.savefig(
            self.plot_dir + "/" + self.plot_prefix + "_mean_reward.png"
        )
        del df1

    def _plot_reward_curiosity(self, df1, N):

        # The "if" is to avoid repeating plot legend
        self.ax_reward_c.plot(
            df1.Episode,
            df1.CuriosityReward,
            color="blue",
            linewidth=0.7,
            label="CuriosityReward" if self.legend_flag else "",
        )
        # plot dommy plot just to show Time step info in x axis
        self.ax_reward_c_2.plot(df1.Time_step, df1.Reward, linestyle="None")

        # Plot average of the last self.average_size observations
        self.ax_reward_c.plot(
            df1.Episode,
            df1.Mean_curiosity_reward,
            color="red",
            linewidth=0.7,
            label=str(self.average_size) + " Rewar Cur avg" if self.legend_flag else "",
        )

        self.ax_reward_c.set_yscale("log")
        self.ax_reward_c_2.set_yscale("log")

        self.ax_reward_c.legend(bbox_to_anchor=(0, 1), loc="upper left")

        self.fig_reward_c.savefig(
            self.plot_dir + "/" + self.plot_prefix + "_mean_curiosity_reward.png"
        )
        del df1

    def _plot_epsilon(self, df1):

        # Min and Max Epsilon value lines
        self.ax_eps.axhline(
            y=self.model.exploration.initial_p,
            color="green",
            label="Initial Eps" if self.legend_flag else "",
        )
        self.ax_eps.axhline(
            y=self.model.exploration.final_p,
            color="red",
            label="Final Eps" if self.legend_flag else "",
        )
        # Epsilon plot
        self.ax_eps.plot(
            df1.Episode,
            df1["Eps_greedy_val"],
            color="blue",
            label="Eps" if self.legend_flag else "",
        )
        # Dummy Plot for showing Time step information
        self.ax_eps_2.plot(df1.Time_step, df1["Eps_greedy_val"], linestyle="None")
        # Show legend box
        self.ax_eps.legend(bbox_to_anchor=(0.75, 1), loc="upper left")

        # save plot
        self.fig_eps.savefig(self.plot_dir + "/" + self.plot_prefix + "_eps_greedy.png")
        del df1

    def _plot_time_steps(self, df1, N):

        # The "if" is to avoid repeating plot legend
        self.ax_ts.plot(
            df1.Episode,
            df1.Episode_timesteps,
            color="blue",
            linewidth=0.7,
            label="Time steps" if self.legend_flag else "",
        )

        # plot dommy plot just to show Time step info in x axis
        self.ax_ts_2.plot(df1.Time_step, df1.Episode_timesteps, linestyle="None")

        # Plot average of the last self.average_size observations
        self.ax_ts.plot(
            df1.Episode,
            df1.Mean_Episode_timesteps,
            color="red",
            linewidth=0.7,
            label=str(self.average_size) + " Epi ts avg" if self.legend_flag else "",
        )
        self.ax_ts.legend(bbox_to_anchor=(0, 1), loc="upper left")

        self.fig_ts.savefig(
            self.plot_dir + "/" + self.plot_prefix + "_episode_timesteps.png"
        )
        del df1

    def _init_plot_epsilon(self):
        """
        Initialize Epsilon plot.
        """

        self.fig_eps, self.ax_eps = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        self.ax_eps.set_xlabel("Episode", fontsize=14)
        self.ax_eps.set_ylabel("Epsilon", fontsize=16)
        self.ax_eps.grid()
        self.ax_eps.set_facecolor("whitesmoke")
        self.ax_eps.set_ylim([-0.03, 1.03])
        self.fig_eps.suptitle("Epsilon Greedy Value", fontsize=20)
        self.fig_eps.subplots_adjust(bottom=0.2)

        # for second x axis labels, create twin plot
        self.ax_eps_2 = self.ax_eps.twiny()
        # Move twinned axis ticks and label from top to bottom
        self.ax_eps_2.xaxis.set_ticks_position("bottom")
        self.ax_eps_2.xaxis.set_label_position("bottom")
        # Offset the twin axis below the host
        self.ax_eps_2.spines["bottom"].set_position(("axes", -0.15))
        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        self.ax_eps_2.set_frame_on(True)
        self.ax_eps_2.patch.set_visible(False)
        self.ax_eps_2.spines["bottom"].set_visible(True)
        self.ax_eps_2.set_xlabel("Time step", fontsize=14)

    def _init_plot_reward(self):
        self.fig_reward, self.ax_reward = plt.subplots(
            nrows=1, ncols=1, figsize=(13, 6)
        )
        # Add some extra space for the second axis at the bottom
        self.fig_reward.subplots_adjust(bottom=0.2)
        self.ax_reward.set_xlabel("Episode", fontsize=14)
        self.ax_reward.set_ylabel("Reward per Episode", fontsize=16)
        self.ax_reward.grid()
        self.ax_reward.set_facecolor("whitesmoke")
        # for second x axis labels, create twin plot
        self.ax_reward_2 = self.ax_reward.twiny()
        # Move twinned axis ticks and label from top to bottom
        self.ax_reward_2.xaxis.set_ticks_position("bottom")
        self.ax_reward_2.xaxis.set_label_position("bottom")
        # Offset the twin axis below the host
        self.ax_reward_2.spines["bottom"].set_position(("axes", -0.15))
        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        self.ax_reward_2.set_frame_on(True)
        self.ax_reward_2.patch.set_visible(False)
        self.ax_reward_2.spines["bottom"].set_visible(True)
        self.ax_reward_2.set_xlabel("Time step", fontsize=14)
        self.fig_reward.suptitle("Episode Reward", fontsize=20)

    def _init_plot_reward_curiosity(self):
        self.fig_reward_c, self.ax_reward_c = plt.subplots(
            nrows=1, ncols=1, figsize=(13, 6)
        )
        # Add some extra space for the second axis at the bottom
        self.fig_reward_c.subplots_adjust(bottom=0.2)
        self.ax_reward_c.set_xlabel("Episode", fontsize=14)
        self.ax_reward_c.set_ylabel("Curiosity Reward (log scale)", fontsize=16)
        self.ax_reward_c.grid()
        self.ax_reward_c.set_facecolor("whitesmoke")
        # for second x axis labels, create twin plot
        self.ax_reward_c_2 = self.ax_reward_c.twiny()
        # Move twinned axis ticks and label from top to bottom
        self.ax_reward_c_2.xaxis.set_ticks_position("bottom")
        self.ax_reward_c_2.xaxis.set_label_position("bottom")
        # Offset the twin axis below the host
        self.ax_reward_c_2.spines["bottom"].set_position(("axes", -0.15))
        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        self.ax_reward_c_2.set_frame_on(True)
        self.ax_reward_c_2.patch.set_visible(False)
        self.ax_reward_c_2.spines["bottom"].set_visible(True)
        self.ax_reward_c_2.set_xlabel("Time step", fontsize=14)
        self.fig_reward_c.suptitle("Episode Curiosity Reward", fontsize=20)

    def _init_plot_time_steps(self):
        self.fig_ts, self.ax_ts = plt.subplots(nrows=1, ncols=1, figsize=(13, 6))
        # Add some extra space for the second axis at the bottom
        self.fig_ts.subplots_adjust(bottom=0.2)
        self.ax_ts.set_xlabel("Episode", fontsize=14)
        self.ax_ts.set_ylabel("Time Steps", fontsize=16)
        self.ax_ts.grid()
        self.ax_ts.set_facecolor("whitesmoke")
        # for second x axis labels, create twin plot
        self.ax_ts_2 = self.ax_ts.twiny()
        # Move twinned axis ticks and label from top to bottom
        self.ax_ts_2.xaxis.set_ticks_position("bottom")
        self.ax_ts_2.xaxis.set_label_position("bottom")
        # Offset the twin axis below the host
        self.ax_ts_2.spines["bottom"].set_position(("axes", -0.15))
        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        self.ax_ts_2.set_frame_on(True)
        self.ax_ts_2.patch.set_visible(False)
        self.ax_ts_2.spines["bottom"].set_visible(True)
        self.ax_ts_2.set_xlabel("Time step", fontsize=14)
        self.fig_ts.suptitle("Time Steps per Episode", fontsize=20)
