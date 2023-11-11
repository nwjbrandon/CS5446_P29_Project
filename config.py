from stable_baselines import A2C, ACER, ACKTR, DQN, PPO2

model_config = {
    "a2c": {
        "model": A2C,
        "total_timesteps": 1000000,
        "config": {
            "gamma": 0.99,
            "n_steps": 5,
            "vf_coef": 0.25,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "learning_rate": 0.0007,
            "alpha": 0.99,
            "epsilon": 1e-05,
            "lr_schedule": "constant",
            "tensorboard_log": None,
            "_init_setup_model": True,
            "policy_kwargs": None,
            "full_tensorboard_log": False,
            "seed": None,
            "n_cpu_tf_sess": None,
        },
    },
    "acer": {
        "model": ACER,
        "total_timesteps": 1000000,
        "config": {
            "gamma": 0.99,
            "n_steps": 20,
            "num_procs": None,
            "q_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 10,
            "learning_rate": 0.0007,
            "lr_schedule": "linear",
            "rprop_alpha": 0.99,
            "rprop_epsilon": 1e-05,
            "buffer_size": 5000,
            "replay_ratio": 4,
            "replay_start": 1000,
            "correction_term": 10.0,
            "trust_region": True,
            "alpha": 0.99,
            "delta": 1,
            "tensorboard_log": None,
            "_init_setup_model": True,
            "policy_kwargs": None,
            "full_tensorboard_log": False,
            "seed": None,
            "n_cpu_tf_sess": 1,
        },
    },
    "acktr": {
        "model": ACKTR,
        "total_timesteps": 1000000,
        "config": {
            "gamma": 0.99,
            "nprocs": None,
            "n_steps": 20,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
            "vf_fisher_coef": 1.0,
            "learning_rate": 0.25,
            "max_grad_norm": 0.5,
            "kfac_clip": 0.001,
            "lr_schedule": "linear",
            "tensorboard_log": None,
            "_init_setup_model": True,
            "async_eigen_decomp": False,
            "kfac_update": 1,
            "gae_lambda": None,
            "policy_kwargs": None,
            "full_tensorboard_log": False,
            "seed": None,
            "n_cpu_tf_sess": 1,
        },
    },
    "dqn": {
        "model": DQN,
        "total_timesteps": 1000000,
        "config": {
            "gamma": 0.99,
            "learning_rate": 0.0005,
            "buffer_size": 50000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.02,
            "exploration_initial_eps": 1.0,
            "train_freq": 1,
            "batch_size": 32,
            "double_q": True,
            "learning_starts": 1000,
            "target_network_update_freq": 500,
            "prioritized_replay": False,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta0": 0.4,
            "prioritized_replay_beta_iters": None,
            "prioritized_replay_eps": 1e-06,
            "param_noise": False,
            "n_cpu_tf_sess": None,
            "tensorboard_log": None,
            "_init_setup_model": True,
            "policy_kwargs": None,
            "full_tensorboard_log": False,
            "seed": None,
        },
    },
    "ppo2": {
        "model": PPO2,
        "total_timesteps": 1000000,
        "config": {
            "gamma": 0.99,
            "n_steps": 128,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "lam": 0.95,
            "nminibatches": 4,
            "noptepochs": 4,
            "cliprange": 0.2,
            "cliprange_vf": None,
            "tensorboard_log": None,
            "_init_setup_model": True,
            "policy_kwargs": None,
            "full_tensorboard_log": False,
            "seed": None,
            "n_cpu_tf_sess": None,
        },
    },
}
