"""This module contains the fixedlag SLAM that feeds the pose covariance to our agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""

    two_pi = 2.0 * math.pi
    return torch.remainder(theta + math.pi, two_pi) - math.pi


def se2_compose(pose: torch.Tensor, delta_local: torch.Tensor) -> torch.Tensor:
    """Compose an SE(2) pose with a local-frame increment."""

    x, y, th = pose.unbind(-1)
    dx, dy, dth = delta_local.unbind(-1)

    c = torch.cos(th)
    s = torch.sin(th)

    x2 = x + c * dx - s * dy
    y2 = y + s * dx + c * dy
    th2 = wrap_angle(th + dth)
    return torch.stack([x2, y2, th2], dim=-1)


def se2_between(xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
    """Relative transform from xi to xj, expressed in xi frame."""

    xi_x, xi_y, ti = xi.unbind(-1)
    xj_x, xj_y, tj = xj.unbind(-1)

    dxg = xj_x - xi_x
    dyg = xj_y - xi_y

    c = torch.cos(ti)
    s = torch.sin(ti)

    dx = c * dxg + s * dyg
    dy = -s * dxg + c * dyg
    dth = wrap_angle(tj - ti)
    return torch.stack([dx, dy, dth], dim=-1)


@dataclass
class FixedLagConfig:
    """Fixed-lag pose-graph + scan-to-map local SLAM."""

    # Fixed-lag graph.
    window_size: int = 15
    gn_iters: int = 2
    damping: float = 1e-3

    # Gauge anchor on oldest node (numerical stability).
    prior_sigma_xy: float = 0.15
    prior_sigma_yaw: float = 0.25

    # Odometry noise model.
    odom_sigma_xy_base: float = 0.10
    odom_sigma_xy_gain: float = 0.30
    odom_sigma_yaw_base: float = 0.10
    odom_sigma_yaw_gain: float = 0.30

    # Occupancy/log-odds map (per env).
    map_H: int = 200
    map_W: int = 200
    resolution: float = 0.10
    origin_x: float = -10.0
    origin_y: float = -10.0

    map_max: float = 20.0

    # Signed updates.
    hit_inc: float = 1.0
    miss_dec: float = 0.35

    free_samples: int = 4

    # Map integration:
    integrate_when_confident: bool = True
    map_warmup_steps: int = 75  # only used if integrate_when_confident=True

    # If integrate_when_confident == True:
    map_update_min_w: float = 0.25
    map_update_use_margin: bool = True
    map_update_margin_bias: float = 0.25
    map_update_q_power: float = 2.0

    map_bootstrap_score_thresh: float = 0.30
    map_bootstrap_min_w: float = 0.75

    # LiDAR geometry.
    angle_min: float = -math.pi / 2.0
    angle_max: float = math.pi / 2.0
    max_range: float = 10.0
    lidar_clip_min: float = 0.10

    # Scan usage / performance.
    scan_match_beam_stride: int = 2
    map_update_beam_stride: int = 1

    # Scan match search grid.
    search_xy_range: float = 0.50
    search_xy_steps: int = 7
    search_yaw_range: float = 0.50
    search_yaw_steps: int = 9

    cand_reg_w: float = 0.02
    cand_reg_yaw_w: float = 0.25

    # Acceptance.
    min_valid_points_frac: float = 0.08
    accept_score_min: float = 0.02

    # Quality computation parameters (score and margin -> q in [0,1]).
    q_score_min: float = 0.10
    q_score_ref: float = 0.95
    q_margin_ref: float = 0.25
    q_pose_init: float = 0.15

    # If True, XY confidence uses a mild margin dependence too:
    # q_xy = q_score * (bias + (1-bias)*q_margin).
    # This prevents max-confidence XY updates when the match is ambiguous.
    meas_xy_use_margin: bool = True
    q_xy_margin_bias: float = 0.75      # Closer to 1 => weaker effect; closer to 0 => stronger effect.

    # Measurement covariance from quality.
    meas_sigma_xy_min: float = 0.1
    meas_sigma_yaw_min: float = 0.2
    meas_sigma_xy_max: float = 0.60
    meas_sigma_yaw_max: float = 1.10

    meas_info_scale: float = 0.5

    # Performance setting.
    scanmatch_chunk_cands: int = 96     # Works well on a RTX 4060 8GB VRAM.


class FixedLagSLAMVectorized:
    def __init__(self, cfg: FixedLagConfig, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        L = int(cfg.window_size)
        if L < 2:
            raise ValueError("FixedLagConfig.window_size must be >= 2")
        self.L = L
        self.dim = 3 * L

        self.poses = torch.zeros((self.num_envs, L, 3), device=self.device, dtype=torch.float32)

        self.odom = torch.zeros((self.num_envs, L - 1, 3), device=self.device, dtype=torch.float32)
        self.odom_info = torch.zeros((self.num_envs, L - 1, 3), device=self.device, dtype=torch.float32)  # Diag info.

        self.abs_z = torch.zeros((self.num_envs, L, 3), device=self.device, dtype=torch.float32)
        self.abs_info = torch.zeros((self.num_envs, L, 3), device=self.device, dtype=torch.float32)  # Diag info.

        H, W = int(cfg.map_H), int(cfg.map_W)
        self.map = torch.zeros((self.num_envs, H, W), device=self.device, dtype=torch.float32)

        self._t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        self.mu_pre = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.Sigma_pre = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)

        self._num_beams: Optional[int] = None
        self._cos_full: Optional[torch.Tensor] = None
        self._sin_full: Optional[torch.Tensor] = None
        self._beam_idx_match: Optional[torch.Tensor] = None
        self._beam_idx_map: Optional[torch.Tensor] = None

        self._build_candidate_grid()
        self.weights = torch.ones((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self._init_dbg_fields()

    def _init_dbg_fields(self) -> None:
        N = self.num_envs
        d = self.device
        self.dbg_hit_frac = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_valid_point_frac = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_valid_point_frac_min = torch.tensor(float("nan"), device=d)
        self.dbg_valid_point_frac_p05 = torch.tensor(float("nan"), device=d)

        self.dbg_best_score = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_second_score = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_margin = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_accept_frac = torch.tensor(float("nan"), device=d)

        self.dbg_meas_q = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_meas_sigma_xy = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_meas_sigma_yaw = torch.full((N,), float("nan"), device=d, dtype=torch.float32)

        self.dbg_search_std_xy = torch.tensor(float(self.cfg.search_xy_range), device=d)
        self.dbg_search_std_yaw = torch.tensor(float(self.cfg.search_yaw_range), device=d)

        self.dbg_neff_pre = torch.tensor(float("nan"), device=d)
        self.dbg_neff_post = torch.tensor(float("nan"), device=d)
        self.dbg_did_resample = torch.tensor(float("nan"), device=d)

        self.dbg_map_w = torch.full((N,), float("nan"), device=d, dtype=torch.float32)
        self.dbg_q_map = torch.full((N,), float("nan"), device=d, dtype=torch.float32)

    def _build_candidate_grid(self) -> None:
        cfg = self.cfg
        dx = torch.linspace(
            -cfg.search_xy_range, cfg.search_xy_range,
            steps=max(int(cfg.search_xy_steps), 1),
            device=self.device, dtype=torch.float32
        )
        dy = torch.linspace(
            -cfg.search_xy_range, cfg.search_xy_range,
            steps=max(int(cfg.search_xy_steps), 1),
            device=self.device, dtype=torch.float32
        )
        dth = torch.linspace(
            -cfg.search_yaw_range, cfg.search_yaw_range,
            steps=max(int(cfg.search_yaw_steps), 1),
            device=self.device, dtype=torch.float32
        )
        gdx, gdy, gth = torch.meshgrid(dx, dy, dth, indexing="ij")
        self._cand_dx = gdx.reshape(-1)
        self._cand_dy = gdy.reshape(-1)
        self._cand_dth = gth.reshape(-1)
        self._num_cands = int(self._cand_dx.numel())

        reg_w = float(cfg.cand_reg_w)
        yaw_w = float(cfg.cand_reg_yaw_w)
        self._cand_pen = reg_w * (self._cand_dx**2 + self._cand_dy**2 + yaw_w * (self._cand_dth**2))

    def _ensure_lidar_cache(self, ranges: torch.Tensor) -> None:
        B = int(ranges.shape[1])
        if self._num_beams == B and self._cos_full is not None:
            return

        cfg = self.cfg
        self._num_beams = B

        angles = torch.linspace(
            float(cfg.angle_min), float(cfg.angle_max),
            steps=B, device=self.device, dtype=torch.float32
        )
        self._cos_full = torch.cos(angles)
        self._sin_full = torch.sin(angles)

        s_match = max(int(cfg.scan_match_beam_stride), 1)
        s_map = max(int(cfg.map_update_beam_stride), 1)
        self._beam_idx_match = torch.arange(0, B, step=s_match, device=self.device, dtype=torch.long)
        self._beam_idx_map = torch.arange(0, B, step=s_map, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def reset(self, env_ids=None, init_pose_xyyaw: Optional[torch.Tensor] = None, clear_map: bool = True) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if init_pose_xyyaw is None:
            init_pose_xyyaw = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=torch.float32)
        else:
            init_pose_xyyaw = init_pose_xyyaw.to(device=self.device, dtype=torch.float32)

        self.poses[env_ids] = init_pose_xyyaw[:, None, :].expand(-1, self.L, -1)
        self.odom[env_ids] = 0.0
        self.odom_info[env_ids] = 0.0
        self.abs_z[env_ids] = 0.0
        self.abs_info[env_ids] = 0.0

        if clear_map:
            self.map[env_ids] = 0.0

        self._t[env_ids] = 0
        self.mu_pre[env_ids] = init_pose_xyyaw

        sig_xy = float(self.cfg.prior_sigma_xy)
        sig_yaw = float(self.cfg.prior_sigma_yaw)
        Sigma0 = torch.zeros((env_ids.numel(), 3, 3), device=self.device, dtype=torch.float32)
        Sigma0[:, 0, 0] = sig_xy * sig_xy
        Sigma0[:, 1, 1] = sig_xy * sig_xy
        Sigma0[:, 2, 2] = sig_yaw * sig_yaw
        self.Sigma_pre[env_ids] = Sigma0

        self.weights[env_ids] = 1.0

        self.dbg_map_w[env_ids] = float("nan")
        self.dbg_q_map[env_ids] = float("nan")

    @torch.no_grad()
    def step(
        self,
        v_cmd: torch.Tensor,
        w_cmd: torch.Tensor,
        dt: float,
        ranges: torch.Tensor,
        imu_wz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_lidar_cache(ranges)

        cfg = self.cfg
        N = self.num_envs

        self._t += 1

        v = v_cmd.to(self.device, dtype=torch.float32)
        w = (imu_wz.to(self.device, dtype=torch.float32) if imu_wz is not None else w_cmd.to(self.device, dtype=torch.float32))

        prev_pose = self.poses[:, -1].clone()
        dx = v * float(dt)
        dy = torch.zeros_like(dx)
        dth = w * float(dt)
        delta = torch.stack([dx, dy, dth], dim=-1)
        x_pred = se2_compose(prev_pose, delta)

        # slide window
        self.poses[:, :-1].copy_(self.poses[:, 1:].clone())
        self.odom[:, :-1].copy_(self.odom[:, 1:].clone())
        self.odom_info[:, :-1].copy_(self.odom_info[:, 1:].clone())
        self.abs_z[:, :-1].copy_(self.abs_z[:, 1:].clone())
        self.abs_info[:, :-1].copy_(self.abs_info[:, 1:].clone())

        self.poses[:, -1] = x_pred
        self.odom[:, -1] = delta

        sig_xy = cfg.odom_sigma_xy_base + cfg.odom_sigma_xy_gain * dx.abs()
        sig_yaw = cfg.odom_sigma_yaw_base + cfg.odom_sigma_yaw_gain * dth.abs()
        odom_info = torch.stack(
            [1.0 / (sig_xy * sig_xy + 1e-9),
             1.0 / (sig_xy * sig_xy + 1e-9),
             1.0 / (sig_yaw * sig_yaw + 1e-9)],
            dim=-1
        )
        self.odom_info[:, -1] = odom_info

        # Scan match.
        z_meas, info_diag, q, q_score, q_margin, do_pose_init = self._scan_match_absolute(x_pred, ranges)
        self.abs_z[:, -1] = z_meas
        self.abs_info[:, -1] = info_diag

        self.poses[:, -1] = torch.where(do_pose_init.unsqueeze(-1), z_meas, x_pred)

        self._optimize_window_inplace()

        self.mu_pre.copy_(self.poses[:, -1])
        H = self._build_hessian(self.poses)
        self.Sigma_pre.copy_(self._marginal_cov_last(H))

        # Map update weight: default always integrate.
        if not bool(cfg.integrate_when_confident):
            w_map = torch.ones((N,), device=self.device, dtype=torch.float32)
            q_map_eff = torch.ones((N,), device=self.device, dtype=torch.float32)
        else:
            warm = self._t <= int(cfg.map_warmup_steps)
            base = float(max(0.0, min(1.0, cfg.map_update_min_w)))

            if bool(cfg.map_update_use_margin):
                bias = float(max(0.0, min(1.0, cfg.map_update_margin_bias)))
                q_map = (q_score * (bias + (1.0 - bias) * q_margin)).clamp(0.0, 1.0)
            else:
                q_map = q_score.clamp(0.0, 1.0)

            p = float(cfg.map_update_q_power)
            if (not math.isfinite(p)) or p <= 0.0:
                p = 1.0
            q_map_eff = q_map.pow(p) if abs(p - 1.0) > 1e-6 else q_map

            w_map = base + (1.0 - base) * q_map_eff
            w_map = torch.where(warm, torch.ones_like(w_map), w_map)

            thr = float(cfg.map_bootstrap_score_thresh)
            minw = float(max(0.0, min(1.0, cfg.map_bootstrap_min_w)))
            if math.isfinite(thr) and thr > 0.0 and math.isfinite(minw) and minw > 0.0:
                low_score = (self.dbg_best_score < thr)
                w_map = torch.where(low_score & (~warm), torch.maximum(w_map, torch.full_like(w_map, minw)), w_map)

            w_map = w_map.clamp(0.0, 1.0)

        self.dbg_map_w = w_map.detach()
        self.dbg_q_map = q_map_eff.detach()

        self._integrate_scan(self.mu_pre, ranges, integrate_weight=w_map)

        return self.mu_pre, self.Sigma_pre

    @torch.no_grad()
    def _scan_to_r_and_dirs(
        self, ranges: torch.Tensor, beam_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        r = ranges[:, beam_idx].to(self.device, dtype=torch.float32)
        r = torch.clamp(r, float(cfg.lidar_clip_min), float(cfg.max_range))
        hit = r < (float(cfg.max_range) * 0.995)

        cos_a = self._cos_full[beam_idx]
        sin_a = self._sin_full[beam_idx]
        return r, hit, cos_a, sin_a

    @torch.no_grad()
    def _scan_match_absolute(
        self,
        pose_pred: torch.Tensor,
        ranges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        N = self.num_envs
        Hm, Wm = self.map.shape[1], self.map.shape[2]
        map_flat = self.map.view(N, -1)

        r, hit, cos_a, sin_a = self._scan_to_r_and_dirs(ranges, self._beam_idx_match)
        P = int(r.shape[1])

        hit_count = hit.sum(dim=1).to(torch.float32)
        self.dbg_hit_frac = hit_count / float(max(P, 1))

        px = r * cos_a.unsqueeze(0)
        py = r * sin_a.unsqueeze(0)

        x0 = pose_pred[:, 0]
        y0 = pose_pred[:, 1]
        th0 = pose_pred[:, 2]
        c0 = torch.cos(th0)
        s0 = torch.sin(th0)

        C = self._num_cands
        scores = torch.empty((N, C), device=self.device, dtype=torch.float32)
        vfracs = torch.empty((N, C), device=self.device, dtype=torch.float32)

        chunk = int(max(cfg.scanmatch_chunk_cands, 1))
        map_max = float(cfg.map_max)

        for start in range(0, C, chunk):
            end = min(start + chunk, C)
            K = end - start

            dx = self._cand_dx[start:end]
            dy = self._cand_dy[start:end]
            dth = self._cand_dth[start:end]

            tx = x0[:, None] + c0[:, None] * dx[None, :] - s0[:, None] * dy[None, :]
            ty = y0[:, None] + s0[:, None] * dx[None, :] + c0[:, None] * dy[None, :]

            th = wrap_angle(th0[:, None] + dth[None, :])
            c = torch.cos(th)
            s = torch.sin(th)

            xg = tx[:, :, None] + c[:, :, None] * px[:, None, :] - s[:, :, None] * py[:, None, :]
            yg = ty[:, :, None] + s[:, :, None] * px[:, None, :] + c[:, :, None] * py[:, None, :]

            gx = torch.floor((xg - float(cfg.origin_x)) / float(cfg.resolution)).to(torch.long)
            gy = torch.floor((yg - float(cfg.origin_y)) / float(cfg.resolution)).to(torch.long)

            inb = (gx >= 0) & (gx < Wm) & (gy >= 0) & (gy < Hm)
            valid = inb & hit[:, None, :]

            gx_c = torch.clamp(gx, 0, Wm - 1)
            gy_c = torch.clamp(gy, 0, Hm - 1)
            lin = gy_c * Wm + gx_c

            vals = map_flat.gather(1, lin.reshape(N, K * P)).reshape(N, K, P)

            valid_f = valid.to(torch.float32)
            num = (vals * valid_f).sum(dim=2)
            den = (valid_f.sum(dim=2) * map_max).clamp(min=1e-6)
            score = num / den

            pen = self._cand_pen[start:end]
            score = score - pen.unsqueeze(0)

            vfrac = (valid_f.sum(dim=2) / float(max(P, 1))).clamp(0.0, 1.0)

            scores[:, start:end] = score
            vfracs[:, start:end] = vfrac

        top2 = torch.topk(scores, k=min(2, C), dim=1)
        best_score = top2.values[:, 0]
        best_idx = top2.indices[:, 0]
        second_score = top2.values[:, 1] if C >= 2 else torch.full_like(best_score, -1e9)
        margin = best_score - second_score
        best_vfrac = vfracs.gather(1, best_idx[:, None]).squeeze(1)

        usable = (best_vfrac >= float(cfg.min_valid_points_frac)) & (best_score >= float(cfg.accept_score_min))

        dx_sel = self._cand_dx[best_idx]
        dy_sel = self._cand_dy[best_idx]
        dth_sel = self._cand_dth[best_idx]

        tx = x0 + c0 * dx_sel - s0 * dy_sel
        ty = y0 + s0 * dx_sel + c0 * dy_sel
        th = wrap_angle(th0 + dth_sel)
        z = torch.stack([tx, ty, th], dim=-1)

        z = torch.where(usable.unsqueeze(-1), z, pose_pred)

        s0_q = float(cfg.q_score_min)
        s1_q = float(cfg.q_score_ref)
        denom_s = max(s1_q - s0_q, 1e-6)
        q_score = ((best_score - s0_q) / denom_s).clamp(0.0, 1.0)

        mref = float(max(cfg.q_margin_ref, 1e-6))
        q_margin = (margin / mref).clamp(0.0, 1.0)

        q = (q_score * q_margin).clamp(0.0, 1.0)
        q = torch.where(usable, q, torch.zeros_like(q))
        q_score = torch.where(usable, q_score, torch.zeros_like(q_score))
        q_margin = torch.where(usable, q_margin, torch.zeros_like(q_margin))

        if bool(cfg.meas_xy_use_margin):
            bias = float(cfg.q_xy_margin_bias)
            if not math.isfinite(bias):
                bias = 0.75
            bias = max(0.0, min(1.0, bias))
            q_xy = (q_score * (bias + (1.0 - bias) * q_margin)).clamp(0.0, 1.0)
        else:
            q_xy = q_score

        q_yaw = q

        sig_xy = float(cfg.meas_sigma_xy_max) - q_xy * (float(cfg.meas_sigma_xy_max) - float(cfg.meas_sigma_xy_min))
        sig_yaw = float(cfg.meas_sigma_yaw_max) - q_yaw * (float(cfg.meas_sigma_yaw_max) - float(cfg.meas_sigma_yaw_min))
        sig_xy = sig_xy.clamp(float(cfg.meas_sigma_xy_min), float(cfg.meas_sigma_xy_max))
        sig_yaw = sig_yaw.clamp(float(cfg.meas_sigma_yaw_min), float(cfg.meas_sigma_yaw_max))

        info_diag = torch.stack(
            [1.0 / (sig_xy * sig_xy + 1e-9),
             1.0 / (sig_xy * sig_xy + 1e-9),
             1.0 / (sig_yaw * sig_yaw + 1e-9)],
            dim=-1
        )

        scale = float(max(cfg.meas_info_scale, 0.0))
        info_diag = info_diag * scale
        info_diag = info_diag * usable.to(torch.float32).unsqueeze(-1)

        do_pose_init = usable & (q >= float(cfg.q_pose_init))

        inv_sqrt = (1.0 / math.sqrt(scale)) if scale > 0.0 else 1e6

        # Debug.
        self.dbg_best_score = best_score
        self.dbg_second_score = second_score
        self.dbg_margin = margin
        self.dbg_accept_frac = usable.to(torch.float32).mean()

        self.dbg_meas_q = q
        self.dbg_meas_sigma_xy = sig_xy * inv_sqrt
        self.dbg_meas_sigma_yaw = sig_yaw * inv_sqrt

        self.dbg_valid_point_frac = best_vfrac
        self.dbg_valid_point_frac_min = best_vfrac.min()
        k = max(int(math.ceil(0.05 * max(N, 1))), 1)
        self.dbg_valid_point_frac_p05 = torch.kthvalue(best_vfrac, min(k, max(N, 1))).values

        return z, info_diag, q, q_score, q_margin, do_pose_init

    @torch.no_grad()
    def _integrate_scan(self, pose: torch.Tensor, ranges: torch.Tensor, integrate_weight: torch.Tensor) -> None:
        cfg = self.cfg
        N = self.num_envs
        Hm, Wm = self.map.shape[1], self.map.shape[2]
        map_flat = self.map.view(N, -1)

        w = integrate_weight.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

        r, hit, cos_a, sin_a = self._scan_to_r_and_dirs(ranges, self._beam_idx_map)
        P = int(r.shape[1])

        x = pose[:, 0]
        y = pose[:, 1]
        th = pose[:, 2]
        c = torch.cos(th)
        s = torch.sin(th)

        px_end = r * cos_a.unsqueeze(0)
        py_end = r * sin_a.unsqueeze(0)

        xg = x[:, None] + c[:, None] * px_end - s[:, None] * py_end
        yg = y[:, None] + s[:, None] * px_end + c[:, None] * py_end

        gx = torch.floor((xg - float(cfg.origin_x)) / float(cfg.resolution)).to(torch.long)
        gy = torch.floor((yg - float(cfg.origin_y)) / float(cfg.resolution)).to(torch.long)

        inb = (gx >= 0) & (gx < Wm) & (gy >= 0) & (gy < Hm)
        upd_hit = inb & hit

        gx_c = torch.clamp(gx, 0, Wm - 1)
        gy_c = torch.clamp(gy, 0, Hm - 1)
        lin_hit = (gy_c * Wm + gx_c).to(torch.long)

        src_hit = upd_hit.to(torch.float32) * float(cfg.hit_inc)
        src_hit = src_hit * w.unsqueeze(1)
        map_flat.scatter_add_(1, lin_hit, src_hit)

        K = int(max(cfg.free_samples, 0))
        if K > 0:
            frac = torch.linspace(
                1.0 / (K + 1), float(K) / (K + 1),
                steps=K, device=self.device, dtype=torch.float32
            )
            t = r.unsqueeze(-1) * frac.view(1, 1, K)

            px_free = t * cos_a.view(1, P, 1)
            py_free = t * sin_a.view(1, P, 1)

            xg_f = x.view(N, 1, 1) + c.view(N, 1, 1) * px_free - s.view(N, 1, 1) * py_free
            yg_f = y.view(N, 1, 1) + s.view(N, 1, 1) * px_free + c.view(N, 1, 1) * py_free

            gx_f = torch.floor((xg_f - float(cfg.origin_x)) / float(cfg.resolution)).to(torch.long)
            gy_f = torch.floor((yg_f - float(cfg.origin_y)) / float(cfg.resolution)).to(torch.long)

            inb_f = (gx_f >= 0) & (gx_f < Wm) & (gy_f >= 0) & (gy_f < Hm)
            hit_f = hit.unsqueeze(-1).expand_as(inb_f)
            upd_free = inb_f & hit_f

            gx_fc = torch.clamp(gx_f, 0, Wm - 1)
            gy_fc = torch.clamp(gy_f, 0, Hm - 1)
            lin_free = (gy_fc * Wm + gx_fc).to(torch.long)

            lin_free2 = lin_free.reshape(N, P * K)
            upd_free2 = upd_free.reshape(N, P * K)

            src_free = upd_free2.to(torch.float32) * (-float(cfg.miss_dec))
            src_free = src_free * w.unsqueeze(1)
            map_flat.scatter_add_(1, lin_free2, src_free)

        map_flat.clamp_(-float(cfg.map_max), float(cfg.map_max))

    def _prior_info_diag(self) -> torch.Tensor:
        cfg = self.cfg
        sig_xy = float(cfg.prior_sigma_xy)
        sig_yaw = float(cfg.prior_sigma_yaw)
        return torch.tensor(
            [1.0 / (sig_xy * sig_xy + 1e-9),
             1.0 / (sig_xy * sig_xy + 1e-9),
             1.0 / (sig_yaw * sig_yaw + 1e-9)],
            device=self.device, dtype=torch.float32
        )

    @torch.no_grad()
    def _optimize_window_inplace(self) -> None:
        iters = int(max(self.cfg.gn_iters, 0))
        if iters <= 0:
            return

        poses = self.poses
        for _ in range(iters):
            H, b = self._build_linear_system(poses)
            dx = torch.linalg.solve(H, (-b).unsqueeze(-1)).squeeze(-1)
            poses = poses + dx.view(self.num_envs, self.L, 3)
            poses[:, :, 2] = wrap_angle(poses[:, :, 2])

        self.poses = poses

    @torch.no_grad()
    def _build_linear_system(self, poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        N, L = self.num_envs, self.L
        dim = self.dim

        H = torch.zeros((N, dim, dim), device=self.device, dtype=torch.float32)
        b = torch.zeros((N, dim), device=self.device, dtype=torch.float32)

        w_prior = self._prior_info_diag()
        H[:, 0:3, 0:3] += torch.diag(w_prior).unsqueeze(0)

        for i in range(L - 1):
            xi = poses[:, i, :]
            xj = poses[:, i + 1, :]
            u = self.odom[:, i, :]
            w = self.odom_info[:, i, :]

            rel = se2_between(xi, xj)
            e = rel - u
            e[:, 2] = wrap_angle(e[:, 2])

            ti = xi[:, 2]
            c = torch.cos(ti)
            s = torch.sin(ti)
            dxg = xj[:, 0] - xi[:, 0]
            dyg = xj[:, 1] - xi[:, 1]

            dt_x_dth = (-s) * dxg + c * dyg
            dt_y_dth = (-c) * dxg - s * dyg

            A = torch.zeros((N, 3, 3), device=self.device, dtype=torch.float32)
            B = torch.zeros((N, 3, 3), device=self.device, dtype=torch.float32)

            A[:, 0, 0] = -c
            A[:, 0, 1] = -s
            A[:, 1, 0] = s
            A[:, 1, 1] = -c
            A[:, 0, 2] = dt_x_dth
            A[:, 1, 2] = dt_y_dth
            A[:, 2, 2] = -1.0

            B[:, 0, 0] = c
            B[:, 0, 1] = s
            B[:, 1, 0] = -s
            B[:, 1, 1] = c
            B[:, 2, 2] = 1.0

            w_row = w.unsqueeze(-1)
            A_w = A * w_row
            B_w = B * w_row

            At = A.transpose(1, 2)
            Bt = B.transpose(1, 2)

            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * (i + 1), 3 * (i + 1) + 3)

            H[:, si, si] += At @ A_w
            H[:, si, sj] += At @ B_w
            H[:, sj, sj] += Bt @ B_w
            H[:, sj, si] += Bt @ A_w

            we = e * w
            b[:, si] += (At @ we.unsqueeze(-1)).squeeze(-1)
            b[:, sj] += (Bt @ we.unsqueeze(-1)).squeeze(-1)

        for i in range(L):
            z = self.abs_z[:, i, :]
            w = self.abs_info[:, i, :]
            xi = poses[:, i, :]

            e = xi - z
            e[:, 2] = wrap_angle(e[:, 2])

            si = slice(3 * i, 3 * i + 3)
            H[:, si, si] += torch.diag_embed(w)
            b[:, si] += w * e

        if cfg.damping > 0.0:
            idx = torch.arange(dim, device=self.device)
            H[:, idx, idx] += float(cfg.damping)

        return H, b

    @torch.no_grad()
    def _build_hessian(self, poses: torch.Tensor) -> torch.Tensor:
        H, _ = self._build_linear_system(poses)
        return H

    @torch.no_grad()
    def _marginal_cov_last(self, H: torch.Tensor) -> torch.Tensor:
        N = H.shape[0]
        dim = H.shape[1]
        idx_last = torch.arange(dim - 3, dim, device=self.device, dtype=torch.long)

        E = torch.zeros((N, dim, 3), device=self.device, dtype=torch.float32)
        E[:, idx_last[0], 0] = 1.0
        E[:, idx_last[1], 1] = 1.0
        E[:, idx_last[2], 2] = 1.0

        X = torch.linalg.solve(H, E)
        Sigma_last = X[:, idx_last, :]
        return 0.5 * (Sigma_last + Sigma_last.transpose(1, 2))