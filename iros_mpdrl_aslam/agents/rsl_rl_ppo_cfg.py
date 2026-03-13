"""This module contains PPO hyperparameters configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg
from ..config import PHASE


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Hyperparameters for PDRL-ASLAM task."""

    if PHASE == "train":
        num_steps_per_env = 512
        max_iterations = 500
        save_interval = 2
        experiment_name = "IROS_PDRL_ASLAM"
        empirical_normalization = True
        policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=0.1,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
            rnn_type="gru",
            rnn_num_layers=1,
            rnn_hidden_dim=256,
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=4,
            num_mini_batches=25,       # Divisor of number of parallel environments (750).
            learning_rate=3e-4,
            schedule="fixed",
            gamma=0.999,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=0.5,
        )

    else:
        num_steps_per_env = 512
        max_iterations = 500
        save_interval = 2
        experiment_name = "IROS_PDRL_ASLAM"
        empirical_normalization = True
        policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=0.1,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
            rnn_type="gru",
            rnn_num_layers=1,
            rnn_hidden_dim=256,
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=3,
            num_mini_batches=2,       # Divisor of number of parallel environments (4).
            learning_rate=1e-5,
            schedule="fixed",
            gamma=0.999,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=0.5,
        )
