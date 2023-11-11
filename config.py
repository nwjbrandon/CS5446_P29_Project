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
        "config": {},
    },
    "dqn": {
        "model": DQN,
        "total_timesteps": 1000000,
        "config": {},
    },
    "ppo2": {
        "model": PPO2,
        "total_timesteps": 1000000,
        "config": {},
    },
}
