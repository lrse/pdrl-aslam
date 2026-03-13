"""This module contains the basic RL rewards configuration."""

from __future__ import annotations


# One-shot penalty
def collision_impulse_on_done(env, term_name: str = "obstacle_too_close"):
    hit = env.termination_manager.get_term(term_name)
    # RewardManager will multiply by dt, so we perform 1/dt to make the weight comparable to Placed-Castellanos.
    return hit.float() / env.step_dt
