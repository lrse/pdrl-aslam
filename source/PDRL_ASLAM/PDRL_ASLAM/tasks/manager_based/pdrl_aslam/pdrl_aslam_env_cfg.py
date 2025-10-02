"""This module contains the RL environment configuration."""

from __future__ import annotations

# General modules.
import numpy as np
from scipy.spatial.transform import Rotation as R

# Isaac Lab modules.
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg
)
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import RayCasterCfg, patterns, CameraCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaacsim.core.utils.viewports import set_camera_view

# Custom modules.
from .config import PROFILE, ENVIRONMENT, PHASE, DEBUG
from .mdp import (
    env1_maze,
    env2_maze,
    env3_maze,
    DiffDriveContCfg,
    DiffDriveCont,
    horizontal_scan,
    slam_x_term,
    slam_y_term,
    slam_b_term,
    collision_impulse_on_done,
    rf,
    pose_coverage2d_delta,
    is_too_close,
    reset_visit_event,
    BOUNDS_XY,
    RESOLUTION,
    FOOTPRINT_R,
    POSE_RANGE
)
from .turtlebot import TURTLEBOT3_CFG


# Config helpers.
def _set_environment():
    if ENVIRONMENT == "env1":
        return env1_maze
    elif ENVIRONMENT == "env2":
        return env2_maze
    elif ENVIRONMENT == "env3":
        return env3_maze
    else:
        raise ValueError(f"Wrong ENVIRONMENT '{ENVIRONMENT}'")


if PROFILE not in ["NO_SLAM", "SLAM_and_occupancy_grid"]:
    raise ValueError(f"Wrong PROFILE '{PROFILE}'")

##
# Scene definition.
##


@configclass
class PdrlAslamSimCfg(InteractiveSceneCfg):
    # Ground plane.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1e-4)),
    )

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Terrain.
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8, 8),
            color_scheme="random",
            horizontal_scale=0.04,
            vertical_scale=0.005,
            sub_terrains={
                "ring_maze": _set_environment()
            },
            border_width=0.0,
        ),
        visual_material=None,
    )

    # Robot.
    turtlebot3_burger: ArticulationCfg = TURTLEBOT3_CFG.replace(prim_path="{ENV_REGEX_NS}/turtlebot3_burger")

    # 2D lidar.
    horizontal_scanner_1 = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/turtlebot3_burger/base_footprint",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.13)),
        mesh_prim_paths=["/World/obstacleTerrain/terrain/mesh"],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=[0, 0],
            horizontal_fov_range=[-90, 90],
            horizontal_res=1.8,     # Number of rays is (|-90|+90)/horizontal_res = 100.
        ),
        debug_vis=False,     # Set to True to see the lidar in the simulator.
        max_distance=10.0,
    )

    # Stereo cameras for cuVSLAM.
    if PROFILE == "SLAM_and_occupancy_grid":
        cam_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/turtlebot3_burger/base_footprint/base_link/cam_left",
            update_period=1 / 30,
            height=96,
            width=128,
            offset=CameraCfg.OffsetCfg(
                pos=(0.20, +0.06, 0.20),
                rot=(-0.5, 0.5, -0.5, 0.5),
                convention="ros",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 1.0e5),
            ),
            data_types=["rgb"],
        )

        cam_right = cam_left.replace(
            prim_path="{ENV_REGEX_NS}/turtlebot3_burger/base_footprint/base_link/cam_right",
            offset=CameraCfg.OffsetCfg(
                pos=(0.20, -0.06, 0.20),
                rot=(-0.5, 0.5, -0.5, 0.5),
                convention="ros"
            ),
        )


##
# MDP settings.
##


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    base_cmd = DiffDriveContCfg(
        v_bounds=(0.05, 0.3),
        w_bounds=(-0.3, 0.3),
        class_type=DiffDriveCont,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        horizontal_scan_1 = ObsTerm(
            func=horizontal_scan,
            params={
                "sensor_cfg": SceneEntityCfg("horizontal_scanner_1"),
                "offset": 0.0,      # No offset since we want the exact lidar measurement (for future real-life testing).
                "max_range": 10.0
            },
            noise=GaussianNoiseCfg(mean=0, std=0.01),
            clip=(0.1, 10.0),
        )

        slam_x = ObsTerm(func=slam_x_term, clip=(-1.0, 1.0))
        slam_y = ObsTerm(func=slam_y_term, clip=(-1.0, 1.0))
        slam_b = ObsTerm(func=slam_b_term, clip=(0.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True   # True means that noise can be applied.
            self.concatenate_terms = True

    # @configclass
    # class CriticCfg(ObsGroup):          #Privileged information class. Unused for now.

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    collision_impulse = RewardTermCfg(
        func=collision_impulse_on_done,
        params={"term_name": "obstacle_too_close"},
        weight=-100.0,
    )

    extrinsic = RewardTermCfg(
        func=rf,
        weight=1.0,
    )

    if PROFILE == "SLAM_and_occupancy_grid":
        coverage2d = RewardTermCfg(
            func=pose_coverage2d_delta,
            params={
                "bounds_xy": BOUNDS_XY,    # Using bigger than real bounds since we do not want to give real map information to the robot.
                "resolution": RESOLUTION,
                "footprint_radius_m": FOOTPRINT_R,
            },
            weight=1000.0,  # "Finishing" the map is 10x more important than colliding (-100).
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    obstacle_too_close = DoneTerm(
        func=is_too_close,
        params={"sensor_cfg": SceneEntityCfg(name="horizontal_scanner_1"), "safe_distance": 0.15},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="turtlebot3_burger"),
            # z = -0.5 because if not it spawns on the air.
            "pose_range": POSE_RANGE,
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg(name="turtlebot3_burger"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # Resetting the map created by the SLAM.
    if PROFILE == "SLAM_and_occupancy_grid":
        reset_visit_state = EventTerm(func=reset_visit_event, mode="reset")


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class PdrlAslamEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the environment."""
    # Scene settings.
    scene = PdrlAslamSimCfg(num_envs=1, env_spacing=0.0)

    # Basic settings.
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    # Dummy settings.
    commands = CommandsCfg()
    curriculum = CurriculumCfg()

    # SLAM.
    if PROFILE == "NO_SLAM":
        use_slam_obs: bool = False
    else:
        use_slam_obs: bool = True
        # Warmup period to avoid distribution shift.
        if DEBUG == "yes" or PHASE == "play":
            slam_warmup_env_steps: int = 0
        else:
            slam_warmup_env_steps: int = 50000

    def __post_init__(self):
        # Viewer settings.
        self.viewer.eye = [-4.0, 0.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        # Step settings.
        self.decimation = 12

        # Simulation settings.
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.antialiasing_mode = None

        # Episode/control settings. Recall that control_dt = self.decimation * self.sim.dt. In this case it is 0.1s.
        steps_per_episode = 2000
        self.episode_length_s = steps_per_episode * self.decimation * self.sim.dt
        self.is_finite_horizon = False

        # Sensor update periods.
        if self.scene.horizontal_scanner_1 is not None:
            self.scene.horizontal_scanner_1.update_period = self.decimation * self.sim.dt

        if PROFILE == "SLAM_and_occupancy_grid":
            target_cam_hz = 20.0
            period = 1.0 / target_cam_hz

            if self.scene.cam_left is not None:
                self.scene.cam_left.update_period = period
            if self.scene.cam_right is not None:
                self.scene.cam_right.update_period = period


def camera_follow(env):
    if env.unwrapped.scene.num_envs == 1:
        robot_position = env.unwrapped.scene["turtlebot3_burger"].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene["turtlebot3_burger"].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler("zyx")[0]
        yaw_rotation = R.from_euler("z", yaw).as_matrix()
        set_camera_view(yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position, robot_position)
