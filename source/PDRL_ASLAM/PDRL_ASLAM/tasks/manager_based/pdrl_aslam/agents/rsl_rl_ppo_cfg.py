"""This module contains PPO hyperparameters configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from ..config import PHASE


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Hyperparameters for PDRL-ASLAM task."""

    if PHASE == "train":
        num_steps_per_env = 256
        max_iterations = 500
        save_interval = 1
        experiment_name = "PDRL_ASLAM_v0"
        empirical_normalization = True
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.3,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=3,
            num_mini_batches=16,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.995,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    else:       # For retraining/playing.
        num_steps_per_env = 2048
        max_iterations = 2000
        save_interval = 1
        experiment_name = "PDRL_ASLAM_v0"
        empirical_normalization = True
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.2,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0075,
            num_learning_epochs=4,
            num_mini_batches=8,
            learning_rate=1e-4,
            schedule="adaptive",
            gamma=0.997,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=0.5,
        )
