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

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns, ImuCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaacsim.core.utils.viewports import set_camera_view

# Custom modules.
from .config import ENVIRONMENT, warehouse_bool
from .mdp import (
    env1_maze,
    env2_maze,
    env3_maze,
    DiffDriveContCfg,
    DiffDriveCont,
    horizontal_scan,
    collision_impulse_on_done,
    is_too_close,
    slam_U, 
    imu_yaw_rate,
    POSE_RANGE,
    RBPFSLAMActiveReward,
)

from .turtlebot import TURTLEBOT3_CFG

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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


# In case a colored robot is required.
from pathlib import Path

NEON_CYAN = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.00, 1.00, 0.60),
    emissive_color=(0.00, 0.45, 0.25),
    metallic=0.0,
    roughness=0.45,
)

_BASE_DIR = Path(__file__).resolve().parent 
USD_PATH = (_BASE_DIR / "asset" / "turtlebot3_description" / "urdf" / "turtlebot3_burger.usd").as_posix()

from dataclasses import replace as dc_replace
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import clone, make_uninstanceable, bind_visual_material

@clone 
def spawn_usd_neon_material(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
    ):

    cfg_no_mat = dc_replace(cfg, visual_material=None)
    prim = from_files._spawn_from_usd_file(
    prim_path, cfg_no_mat.usd_path, cfg_no_mat, translation, orientation, **kwargs
    )

    make_uninstanceable(prim_path)

    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else: 
            material_path = cfg.visual_material_path

    cfg.visual_material.func(material_path, cfg.visual_material)
    bind_visual_material(prim_path, material_path)

    return prim


##
# Scene definition.
##


@configclass
class IrosMpdrlAslamSimCfg(InteractiveSceneCfg):
    # Ground plane.
    if warehouse_bool == False:
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
    if warehouse_bool == False:
        terrain = TerrainImporterCfg(
            prim_path="/World/obstacleTerrain",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(10, 10),
                horizontal_scale=0.04,
                vertical_scale=0.005,
                sub_terrains={
                    "ring_maze": _set_environment()
                },
                border_width=0.0,
            ),
            visual_material=None,
        )

    # Full warehouse.
    if warehouse_bool == True:
        warehouse = AssetBaseCfg(
            prim_path="/World/Full_warehouse",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd")    #or full_warehouse.usd
        )

    # No-colored Robot.
    # turtlebot3_burger: ArticulationCfg = TURTLEBOT3_CFG.replace(prim_path="{ENV_REGEX_NS}/turtlebot3_burger")

    # In case a colored robot is required.
    turtlebot3_burger: ArticulationCfg = TURTLEBOT3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/turtlebot3_burger",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_usd_neon_material,
            usd_path=USD_PATH,
            visible=True,
            activate_contact_sensors=False,
            rigid_props=TURTLEBOT3_CFG.spawn.rigid_props,
            articulation_props=TURTLEBOT3_CFG.spawn.articulation_props,
            visual_material=NEON_CYAN,
            visual_material_path="material",
        ),
    )

# Important: SLAM will read from the LiDAR directly which already contains noise.

    # 2D lidar.
    if warehouse_bool == False:
        horizontal_scanner_1 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/turtlebot3_burger/base_footprint",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.13)),
            mesh_prim_paths=["/World/obstacleTerrain/terrain/mesh"],
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=[0, 0],
                horizontal_fov_range=[-90, 90],
                horizontal_res=180 / (128 - 1),     # Number of rays is 128.
            ),
            debug_vis=False,     # Set to True to see the lidar in the simulator.
            max_distance=10.0,

            # Sensor noise.
            ray_cast_drift_range={"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
        )

    # Multimesh 2D Lidar.
    if warehouse_bool == True:
        horizontal_scanner_1 = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/turtlebot3_burger/base_footprint",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.13)),
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/Full_warehouse",
                    is_shared=True,
                    merge_prim_meshes=True,
                    track_mesh_transforms=False
                )
            ],

            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=[0, 0],
                horizontal_fov_range=[-90, 90],
                horizontal_res=180/(128 - 1),
            ),
            debug_vis=False,
            max_distance=10.0,

            # Sensor noise.
            ray_cast_drift_range={"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
        )

    # Imu.
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/turtlebot3_burger/base_footprint",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        debug_vis=False,     # Set to True to see the Imu in the simulator.
    )


##
# MDP settings.
##


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    base_cmd = DiffDriveContCfg(
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
            clip=(0.1, 10.0),
        )

        imu_yaw_vel = ObsTerm(
            func=imu_yaw_rate,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )

        slam_log_dopt = ObsTerm(                
            func=slam_U,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False   # True means that noise can be applied.
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
        weight=-500.0,
    )

    slam_active = RewardTermCfg(
        func=RBPFSLAMActiveReward,
        params={
            "robot_cfg": SceneEntityCfg(name="turtlebot3_burger"),
            "lidar_cfg": SceneEntityCfg(name="horizontal_scanner_1"),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
        weight=1.0,
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
                "yaw": (-3.14159, 3.14159),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg(name="turtlebot3_burger"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class IrosMpdrlAslamEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the environment."""
    # Scene settings.
    scene = IrosMpdrlAslamSimCfg(num_envs=1, env_spacing=0.0)

    # Basic settings.
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    # Unused (for now) settings.
    commands = CommandsCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        # Viewer settings.
        self.viewer.eye = [-4.0, 0.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        # Step settings.
        self.decimation = 6    # Was 12.

        # Simulation settings.
        self.sim.dt = 1 / 60   # Was 1/120.
        self.sim.render_interval = self.decimation
        self.sim.render.antialiasing_mode = None

        # Episode/control settings. Recall that control_dt = self.decimation * self.sim.dt. In this case it is 0.1s.
        steps_per_episode = 5000    # X mins
        self.episode_length_s = steps_per_episode * self.decimation * self.sim.dt
        self.is_finite_horizon = False

        # Sensor update periods.
        if self.scene.horizontal_scanner_1 is not None:
            self.scene.horizontal_scanner_1.update_period = self.decimation * self.sim.dt

        if self.scene.imu is not None:
            self.scene.imu.update_period = self.decimation * self.sim.dt


def camera_follow(env):
    if env.unwrapped.scene.num_envs == 1:
        robot_position = env.unwrapped.scene["turtlebot3_burger"].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene["turtlebot3_burger"].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler("zyx")[0]
        yaw_rotation = R.from_euler("z", yaw).as_matrix()
        set_camera_view(yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position, robot_position)
