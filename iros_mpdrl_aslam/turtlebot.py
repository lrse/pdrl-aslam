"""This module contains the TurtleBot3 Burger import specifications."""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from pathlib import Path
import os


_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_ASSETS_DIR = _BASE_DIR / "asset" / "turtlebot3_description" / "urdf"

_ASSETS_DIR = Path(os.getenv("PDRL_ASLAM_ASSETS", str(_DEFAULT_ASSETS_DIR))).resolve()
_URDF = _ASSETS_DIR / "turtlebot3_burger.urdf"
_USD_DIR = _ASSETS_DIR
_USD_NAME = "turtlebot3_burger.usd"

if not _URDF.exists():
    raise FileNotFoundError(
        "Could not find TurtleBot3 URDF at:\n"
        f"  {_URDF}\n"
    )


TURTLEBOT3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF.as_posix(),
        usd_dir=_USD_DIR.as_posix(),
        usd_file_name=_USD_NAME,
        fix_base=False,
        visible=True,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "wheel_left_joint": 0.0,
            "wheel_right_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=16.0,
            effort_limit=20.0,
        ),
    },
)
