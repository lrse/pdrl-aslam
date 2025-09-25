import gymnasium as gym
import torch

class DiffDriveDiscreteActionWrapper:
    """
    Wrapper (NOT a Gym wrapper) that exposes Discrete(3) to the runner
    and maps those actions into the wheels expected by SKRL.

    Placed/Castellanos actions:
      0 -> forward (v=0.30, w=0.0)
      1 -> left (v=0.05, w=+0.3)
      2 -> right (v=0.05, w=-0.3)

    The mapped wheel speeds are normalized to [-1, 1] so the base env (with
    JointVelocityActionCfg(scale=max_wheel_speed)) scales them back to rad/s.
    """

    def __init__(self, env, wheel_radius=0.033, track_width=0.160, max_wheel_speed=10):   # Data from URDF. Changed 6.67 for 9.09
        self.env = env
        self.r = float(wheel_radius)
        self.L = float(track_width)
        self.max_wheel_speed = float(max_wheel_speed)
        self.action_space = gym.spaces.Discrete(3)
        self.single_action_space = self.action_space   # Required for SKRL.

    def _map_discrete_to_wheels_norm(self, idx: torch.Tensor):
        device = idx.device
        # Placed/Castellanos actions.
        pc_v = torch.tensor([0.30, 0.05, 0.05], dtype=torch.float32, device=device)
        pc_w = torch.tensor([0.00, 0.30, -0.30], dtype=torch.float32, device=device)
        v, w = pc_v[idx], pc_w[idx]
        # Linear and angular -> wheel speeds (rad/s).
        wl = (v - 0.5 * self.L * w) / self.r
        wr = (v + 0.5 * self.L * w) / self.r
        # Scale so neither wheel exceeds max.
        mag = torch.maximum(wl.abs(), wr.abs())
        s = torch.clamp(mag / self.max_wheel_speed, min=1.0)
        wl, wr = wl / s, wr / s
        # Normalize to [-1, 1] for the inner env, which will rescale by max_wheel_speed.
        out = torch.stack([wl, wr], dim=-1) / self.max_wheel_speed # Pack the normalized wheel speeds into the last dimension.
        return out.clamp(-1, 1).to(torch.float32)   # The clamp is for potential floating-point errors.

    def step(self, action):
        idx = action    # works because using CategoricalMixin PPO.
        cont = self._map_discrete_to_wheels_norm(idx)  
        return self.env.step(cont)

    #  Attributes SKRL expects get forwarded to the inner env. We get errors without this.
    def __getattr__(self, name):
        return getattr(self.env, name)