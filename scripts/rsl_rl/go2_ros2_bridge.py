"""
cuvSLAM What to do:

Helper: https://chatgpt.com/c/68a36894-d650-832a-a7db-5979f810ff57

FROM HERE!!!

./workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/run_dev.sh 

sudo apt-get update
sudo apt-get install -y ros-humble-isaac-ros-visual-slam ros-humble-isaac-ros-visual-slam-interfaces


cd /workspaces/isaac_ros-dev

# Build just this package and source THIS terminal
colcon build --symlink-install --packages-select my_vslam_bringup
source install/setup.bash

#FOR 1 ENV
ros2 launch my_vslam_bringup my_vslam.launch.py             

#FOR 2 ENV:
TERMINAL A:

source install/setup.bash

ros2 launch my_vslam_bringup my_vslam.launch.py \
  ns:=unitree_go2_0 \
  base_frame:=unitree_go2_0/base_footprint    #CHANGED FROM base_link


TERMINAL B:

./workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/run_dev.sh 

source install/setup.bash

ros2 launch my_vslam_bringup my_vslam.launch.py \
  ns:=unitree_go2_1 \
  base_frame:=unitree_go2_1/base_footprint   #CHANGED FROM base_link

  
TERMINAL C:

./workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/run_dev.sh 

source install/setup.bash

ros2 launch my_vslam_bringup my_vslam.launch.py \
  ns:=unitree_go2_2 \
  base_frame:=unitree_go2_2/base_footprint   #CHANGED FROM base_link



TERMINAL D:

./workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/run_dev.sh 

source install/setup.bash

ros2 launch my_vslam_bringup my_vslam.launch.py \
  ns:=unitree_go2_3 \
  base_frame:=unitree_go2_3/base_footprint   #CHANGED FROM base_link

NUMBER 5 MADE TRAINING STOP DONT KNOW WHY AT 37625 STEPS 
TERMINAL E: 

./workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/run_dev.sh 

source install/setup.bash

ros2 launch my_vslam_bringup my_vslam.launch.py \
  ns:=unitree_go2_4 \
  base_frame:=unitree_go2_4/base_footprint   #CHANGED FROM base_link


NUMBER 6 MADE GPU MEMORY COLLAPSE  
# TERMINAL F: 

# ./workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/run_dev.sh 

# source install/setup.bash

# ros2 launch my_vslam_bringup my_vslam.launch.py \
#   ns:=unitree_go2_5 \
#   base_frame:=unitree_go2_5/base_footprint   #CHANGED FROM base_link


  

TO CHECK
ros2 topic echo /unitree_go2_0/visual_slam/tracking/vo_pose_covariance --once
ros2 topic echo /unitree_go2_1/visual_slam/tracking/vo_pose_covariance --once

"""

import time
import math
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo  # IMU optional
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# import omni.graph.core as og


#FOR LIDAR FOR SLAM_TOOLBOX
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
#FOR LIDAR FOR SLAM_TOOLBOX

#NEW ADDITION FOR PABLO'S IDEA
from nav_msgs.msg import Odometry
#NEW ADDITION FOR PABLO'S IDEA


#IMU
# from sensor_msgs.msg import Imu
# from array import array
#IMU 

#RESETER
from isaac_ros_visual_slam_interfaces.srv import Reset
#RESETER

# --------------------------
# Helpers (dtype/quat/math)
# --------------------------

#JUST ADDED for POSE COMING FROM SLAM
def _quat_wxyz_to_yaw(qw, qx, qy, qz):
    # yaw (Z) from quaternion [w,x,y,z]
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)
#JUST ADDED for POSE COMING FROM SLAM




def _to_numpy_u8_rgb(img_t: torch.Tensor) -> np.ndarray:
    arr = img_t
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,W,C) tensor, got shape {tuple(arr.shape)}")
    if arr.dtype != torch.uint8:
        arr = (arr.clamp(0, 1) * 255.0).to(torch.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.detach().cpu().numpy()


def _rgb_to_mono8(rgb_u8: np.ndarray) -> np.ndarray:
    y = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.uint8)
    return y


# def _quat_conj(q: torch.Tensor) -> torch.Tensor:
#     return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


# def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
#     w1, x1, y1, z1 = q1.unbind(-1)
#     w2, x2, y2, z2 = q2.unbind(-1)
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
#     return torch.stack([w, x, y, z], dim=-1)


# def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
#     w, x, y, z = q.unbind(-1)
#     ww, xx, yy, zz = w * w, x * x, y * y, z * z
#     wx, wy, wz = w * x, w * y, w * z
#     xy, xz, yz = x * y, x * z, y * z
#     r00 = ww + xx - yy - zz
#     r01 = 2 * (xy - wz)
#     r02 = 2 * (xz + wy)
#     r10 = 2 * (xy + wz)
#     r11 = ww - xx + yy - zz
#     r12 = 2 * (yz - wx)
#     r20 = 2 * (xz - wy)
#     r21 = 2 * (yz + wx)
#     r22 = ww - xx - yy + zz
#     return torch.stack(
#         [
#             torch.stack([r00, r01, r02], dim=-1),
#             torch.stack([r10, r11, r12], dim=-1),
#             torch.stack([r20, r21, r22], dim=-1),
#         ],
#         dim=-2,
#     )


def _torch_all_finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def _normalize_quat_wxyz(q: torch.Tensor):
    if not _torch_all_finite(q):
        return None
    n = torch.linalg.norm(q).item()
    if n < 1e-8:
        return None
    return q / n


def _camera_K_from_sensor(sensor, i: int, H: int, W: int):
    K = getattr(sensor.data, "intrinsic_matrices", None)
    if K is not None:
        Kn = K[i].detach().cpu().numpy().reshape(3, 3).astype(float)
        return float(Kn[0, 0]), float(Kn[1, 1]), float(Kn[0, 2]), float(Kn[1, 2])
    cfg = sensor.cfg.spawn
    fx = W * float(cfg.focal_length) / float(cfg.horizontal_aperture)
    # if vertical_aperture missing, infer fy from aspect ratio
    fy = (H * float(cfg.focal_length) / float(getattr(cfg, "vertical_aperture", 0.0))) if getattr(cfg, "vertical_aperture", 0.0) else fx * (H / float(W))
    cx, cy = W * 0.5, H * 0.5
    return fx, fy, cx, cy


# --------------------------
# Main ROS 2 bridge node
# --------------------------
class RobotDataManager(Node):
    """
    For each env (namespace unitree_go2_i):
      Publishes:
        - visual_slam/image_0 (mono8), visual_slam/camera_info_0
        - visual_slam/image_1 (mono8), visual_slam/camera_info_1
        - (optional) dynamic TF: odom -> base_footprint   #CHANGED FROM base_link
        - (once) static TF: base_footprint -> {left,right}_camera_optical_frame   #CHANGED FROM base_link
      Subscribes:
        - visual_slam/tracking/vo_pose_covariance
    """

    def __init__(self, env, lidar_annotators=None, cameras=None, cfg=None):
        super().__init__("robot_data_manager")

        # Isaac handles
        self.env = env
        self.scene = env.unwrapped.scene
        self.num_envs = self.scene.num_envs

        # sensors
        try:
            self.cam_left = self.scene.sensors["cam_left"]
            self.cam_right = self.scene.sensors["cam_right"]
            # self.imu = self.scene.sensors["imu"]
        except KeyError as e:
            raise RuntimeError(
                f"Missing sensor '{e.args[0]}' in scene.sensors. "
                "Add two Camera sensors named cam_left/cam_right and an Imu named imu in File 1."
            )
        
        #FOR LIDAR FOR SLAM_TOOLBOX
        self.lidar_name = "horizontal_scanner_1"
        if self.lidar_name not in self.scene.sensors:
            raise RuntimeError(
                f"Missing LiDAR/raycaster sensor '{self.lidar_name}' in scene.sensors. "
                "Create a RayCaster sensor and name it 'horizontal_scanner_1' or change self.lidar_name."
            )
        self.lidar = self.scene.sensors[self.lidar_name]
        #FOR LIDAR FOR SLAM_TOOLBOX

        #IMU
        # IMU_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
        #                     history=HistoryPolicy.KEEP_LAST, depth=1)
        # self.pub_imu = []
        # for i in range(self.num_envs):
        #     ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
        #     self.pub_imu.append(self.create_publisher(Imu, f"{ns}/visual_slam/imu", IMU_QOS))
        # self._last_imu_pub_wall = 0.0
        # self.imu_hz = 1.0 / float(self.imu.cfg.update_period)
        #IMU

        #RESETTER
        self._reset_clients = []
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            self._reset_clients.append(self.create_client(Reset, f"{ns}/visual_slam/reset"))
        #RESETTER


        # QoS
        IMG_QOS  = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1) #CHANGED FROM RELIABLE SINCE PROLLY NOT NEEDED AND LESS GPU
        INFO_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1) #CHANGED FROM RELIABLE SINCE PROLLY NOT NEEDED AND LESS GPU

        # publishers
        self.pub_left_img, self.pub_right_img = [], []
        self.pub_left_info, self.pub_right_info = [], []
        #FOR LIDAR FOR SLAM_TOOLBOX
        self.pub_scan = []
        #FOR LIDAR FOR SLAM_TOOLBOX

        # TF broadcasters
        # self.tf_broadcaster = TransformBroadcaster(self)
        # self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # covariance state
        self._latest_pose_cov6 = [None] * self.num_envs
        self._pose_stamp = [0.0] * self.num_envs
        self._cov_seq = [0] * self.num_envs  # increments on each new covariance

        # JUST ADDED POSE FROM SLAM
        self._latest_slam_xy  = [None] * self.num_envs
        self._latest_slam_yaw = [None] * self.num_envs
        self._slam_pose_stamp = [0.0]  * self.num_envs
        # JUST ADDED POSE FROM SLAM

        # subscriptions and publishers
        self._pose_cov_subs = []
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            self.pub_left_img.append(self.create_publisher(Image, f"{ns}/visual_slam/image_0", IMG_QOS))
            self.pub_right_img.append(self.create_publisher(Image, f"{ns}/visual_slam/image_1", IMG_QOS))
            self.pub_left_info.append(self.create_publisher(CameraInfo, f"{ns}/visual_slam/camera_info_0", INFO_QOS))
            self.pub_right_info.append(self.create_publisher(CameraInfo, f"{ns}/visual_slam/camera_info_1", INFO_QOS))
            # self._pose_cov_subs.append(
            #     self.create_subscription(
            #         PoseWithCovarianceStamped,
            #         f"{ns}/visual_slam/tracking/vo_pose_covariance",
            #         lambda msg, env_idx=i: self._pose_cov_cb(msg, env_idx),
            #         QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE),
            #     )
            # )
            #FOR LIDAR FOR SLAM_TOOLBOX
            self.pub_scan.append(self.create_publisher(LaserScan, f"{ns}/scan", QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)))
            #FOR LIDAR FOR SLAM_TOOLBOX
        

        #NEW ADDITION FOR PABLO'S IDEA
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            # subscribe to SLAM odometry
            self.create_subscription(
                Odometry,
                f"{ns}/visual_slam/tracking/odometry",
                lambda msg, env_idx=i: self._slam_odom_cb(msg, env_idx),
                QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
        #NEW ADDITION FOR PABLO'S IDEA

        # TF/baseline state
        self._tf_ready = False
        self._baseline_x = [float("nan")] * self.num_envs

        # publish rates from sensor configs (for info)
        self.stereo_hz = 1.0 / float(self.cam_left.cfg.update_period) if self.cam_left.cfg.update_period else 25.0
        #FOR LIDAR FOR SLAM_TOOLBOX
        self.scan_hz = 1.0 / float(self.lidar.cfg.update_period) if self.lidar.cfg.update_period else 15.0
        self._last_scan_pub_wall = 0.0
        #FOR LIDAR FOR SLAM_TOOLBOX

        # cuVSLAM is not publishing odom-> base_footprint, so keep this True   #CHANGED FROM base_link
        self.publish_sim_odom_tf = False       #USED TO BE true but 9/9 changes

        # warn baseline once
        self._warned_tx = [False] * self.num_envs

        # optional /clock graph (harmless if unused)
        self._maybe_create_ros_time_graph()
    
    # ---- public API (called from sim thread) ----
    def pub_ros2_data(self):
        """Publish TF and stereo once. Call from the sim thread after each env.step()."""
        stamp = self.get_clock().now().to_msg()
        # self._maybe_publish_static_tf_from_scene()            #COMMENTED FOR NOW SINCE I THINK MIGHT NOT BE NEEDED SINCE IT IS BEING PASSED ON VSLAM LAUNCHPY
        # if self.publish_sim_odom_tf:          #USED TO BE uncommented but 9/9 changes
        #     self._publish_odom_tf(stamp)      #USED TO BE uncommented but 9/9 changes
        self._publish_stereo_batch(stamp)
        #FOR LIDAR FOR SLAM_TOOLBOX
        now_wall = time.time()
        if (now_wall - self._last_scan_pub_wall) >= (1.0 / self.scan_hz):
            self._last_scan_pub_wall = now_wall
            self._publish_scan_batch(stamp)
        #FOR LIDAR FOR SLAM_TOOLBOX

        #IMU
        # now_wall = time.time()
        # if (now_wall - self._last_imu_pub_wall) >= (1.0 / self.imu_hz):
        #     self._last_imu_pub_wall = now_wall
        #     self._publish_imu_batch(stamp)
        #IMU

    #FOR LIDAR FOR SLAM_TOOLBOX
    def _publish_scan_batch(self, stamp):
        for i in range(self.num_envs):
            scan_msg = self._build_scan_msg(i, stamp)
            if scan_msg is not None:
                self.pub_scan[i].publish(scan_msg)

    def _build_scan_msg(self, env_idx: int, stamp) -> LaserScan | None:
        # Isaac tensors (N, R, 3) and (N, 3)
        hits_w = self.lidar.data.ray_hits_w
        pos_w = self.lidar.data.pos_w
        if hits_w is None or pos_w is None:
            return None

        # to numpy
        hits = hits_w[env_idx].detach().cpu().numpy()   # (R, 3), inf if no hit
        origin = pos_w[env_idx].detach().cpu().numpy()  # (3,)

        # distances; clamp no-hit to max_range
        d = np.linalg.norm(hits - origin[None, :], axis=1)  # (R,)
        max_range = float(self.lidar.cfg.max_distance)
        d = np.where(np.isfinite(d), d, max_range)

        # pattern config → angles
        pcfg = self.lidar.cfg.pattern_cfg
        ang_min = math.radians(float(pcfg.horizontal_fov_range[0]))
        ang_max_cfg = math.radians(float(pcfg.horizontal_fov_range[1]))
        ang_res_deg = float(pcfg.horizontal_res)
        angle_increment = math.radians(ang_res_deg)
        channels = int(pcfg.channels)

        # choose a single ring for 2D LaserScan (middle ring if multi-channel)
        if channels > 1:
            expected_per_ring = int(round(
                (float(pcfg.horizontal_fov_range[1]) - float(pcfg.horizontal_fov_range[0]))
                / ang_res_deg
            )) + 1
            total = d.size
            if expected_per_ring > 0 and total == channels * expected_per_ring:
                rings = d.reshape(channels, expected_per_ring)
                ranges = rings[channels // 2].astype(np.float32)
            else:
                rays_per_ring = total // channels if channels > 0 else total
                if rays_per_ring == 0:
                    return None
                start = (channels // 2) * rays_per_ring
                ranges = d[start:start + rays_per_ring].astype(np.float32)
        else:
            ranges = d.astype(np.float32)

        n = int(ranges.shape[0])
        ang_max = ang_min + (max(n, 1) - 1) * angle_increment
        ang_max = min(ang_max, ang_max_cfg)

        ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{env_idx}"
        scan = LaserScan()
        scan.header.frame_id = f"{ns}/lidar_frame"
        scan.header.stamp = stamp
        scan.angle_min = ang_min
        scan.angle_max = ang_max
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / self.scan_hz
        scan.range_min = 0.02
        scan.range_max = max_range
        scan.ranges = ranges.tolist()
        return scan
    #FOR LIDAR FOR SLAM_TOOLBOX



    #IMU
    # def _imu_timer_cb(self):
    #     stamp = self.get_clock().now().to_msg()
    #     self._publish_imu_batch(stamp)


    # def _publish_imu_batch(self, stamp):
    #     ang = self.imu.data.ang_vel_b
    #     lin = self.imu.data.lin_acc_b
    #     for i in range(self.num_envs):
    #         ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
    #         msg = Imu()
    #         msg.header.stamp = stamp
    #         msg.header.frame_id = f"{ns}/base_footprint"

    #         # orientation not provided:
    #         msg.orientation_covariance[0] = -1.0

    #         msg.angular_velocity.x = float(ang[i, 0])
    #         msg.angular_velocity.y = float(ang[i, 1])
    #         msg.angular_velocity.z = float(ang[i, 2])
    #         # option A: plain Python floats
    #         msg.angular_velocity_covariance = [0.0025, 0.0, 0.0,
    #                                         0.0,    0.0025, 0.0,
    #                                         0.0,    0.0,    0.0025]
    #         # option B: enforce doubles explicitly
    #         # msg.angular_velocity_covariance = array('d', [0.0025, 0.0, 0.0,
    #         #                                               0.0,    0.0025, 0.0,
    #         #                                               0.0,    0.0,    0.0025])

    #         msg.linear_acceleration.x = float(lin[i, 0])
    #         msg.linear_acceleration.y = float(lin[i, 1])
    #         msg.linear_acceleration.z = float(lin[i, 2])
    #         msg.linear_acceleration_covariance = [0.04, 0.0, 0.0,
    #                                             0.0,  0.04, 0.0,
    #                                             0.0,  0.0,  0.04]

    #         self.pub_imu[i].publish(msg)
    #IMU

    #RESETTER
    def reset_slam(self, env_ids=None):
        ids = range(self.num_envs) if env_ids is None else (env_ids.tolist() if hasattr(env_ids, "tolist") else env_ids)
        # clear cached pose/cov so rewards wait for fresh data
        for i in ids:
            self._latest_slam_xy[i] = None
            self._slam_pose_stamp[i] = 0.0
            self._latest_pose_cov6[i] = None
            self._pose_stamp[i] = 0.0
        # call the service (non-blocking; poll while executor spins in background)
        for i in ids:
            try:
                cli = self._reset_clients[i]
                if not cli.service_is_ready():
                    cli.wait_for_service(timeout_sec=3.0)
                fut = cli.call_async(Reset.Request())
                t0 = time.time()
                while not fut.done() and (time.time() - t0) < 5.0:
                    time.sleep(0.01)
                if not fut.done():
                    self.get_logger().warn(f"[reset_slam] Reset service timed out for env {i}")
            except Exception as e:
                try:
                    self.get_logger().warn(f"[reset_slam] Failed for env {i}: {e}")
                except Exception:
                    pass
    #RESETTER

    #DEBUGGING SLAM TRAINING
    def get_latest_slam_yaw(self, env_idx: int = 0, max_age_s: float = 0.5):
        yaw = self._latest_slam_yaw[env_idx]
        if yaw is None:
            return None
        if (time.time() - self._slam_pose_stamp[env_idx]) > max_age_s:
            return None
        return yaw
    #DEBUGGING SLAM TRAINING

    #NEW ADDITION FOR PABLO'S IDEA
    def _slam_odom_cb(self, msg: Odometry, env_idx: int):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._latest_slam_xy[env_idx]  = (float(p.x), float(p.y))
        self._latest_slam_yaw[env_idx] = _quat_wxyz_to_yaw(float(o.w), float(o.x), float(o.y), float(o.z))
        self._slam_pose_stamp[env_idx] = time.time()

    #ADDEDCOVLASTMINUTE
        cov_flat = msg.pose.covariance  # length 36 (row-major)
        if cov_flat is not None and len(cov_flat) == 36:
            C6 = np.asarray(cov_flat, dtype=np.float64).reshape(6, 6)
            C6 = 0.5 * (C6 + C6.T)
            if np.isfinite(C6).all():
                self._latest_pose_cov6[env_idx] = C6
                self._pose_stamp[env_idx] = time.time()
                self._cov_seq[env_idx] += 1
    #ADDEDCOVLASTMINUTE

    def get_latest_slam_xy(self, env_idx: int = 0, max_age_s: float = 0.5):
        xy = self._latest_slam_xy[env_idx]
        if xy is None: return None
        if (time.time() - self._slam_pose_stamp[env_idx]) > max_age_s: return None
        return xy
    

    def get_latest_pose_cov(self, env_idx: int = 0):
        cov = self._latest_pose_cov6[env_idx]
        if cov is None:
            return None
        if (time.time() - self._pose_stamp[env_idx]) > 0.5:
            return None
        return cov
    
    #JUST ADDED for POSE COMING FROM SLAM
    # def get_latest_vo_xy(self, env_idx: int = 0, max_age_s: float = 0.5):
    #     xy = self._latest_vo_xy[env_idx]
    #     if xy is None:
    #         return None
    #     if (time.time() - self._vo_pose_stamp[env_idx]) > max_age_s:
    #         return None
    #     return xy

    # def get_latest_vo_yaw(self, env_idx: int = 0, max_age_s: float = 0.5):
    #     yaw = self._latest_vo_yaw[env_idx]
    #     if yaw is None:
    #         return None
    #     if (time.time() - self._vo_pose_stamp[env_idx]) > max_age_s:
    #         return None
    #     return yaw
    #JUST ADDED for POSE COMING FROM SLAM

    def get_cov_seq(self, env_idx: int = 0) -> int:
        return self._cov_seq[env_idx]

    def wait_for_new_cov_samples(self, last_seq, *, require_all: bool = True,
                                 min_ready_envs: int = 1, timeout: float | None = None) -> bool:
        """Block until new covariance since last_seq. Used by training wrapper."""
        t0 = time.time()
        N = self.num_envs

        def count_new():
            return sum(1 for i in range(N) if self._cov_seq[i] > last_seq[i])

        if (count_new() == N) if require_all else (count_new() >= min_ready_envs):
            return True

        while True:
            if (count_new() == N) if require_all else (count_new() >= min_ready_envs):
                return True
            if timeout is not None and (time.time() - t0) >= timeout:
                return False
            time.sleep(0.001)

    # ---- subscribers / publishers (safe to call on sim thread) ----
    def _pose_cov_cb(self, msg: PoseWithCovarianceStamped, env_idx: int):
        cov = np.array(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        self._latest_pose_cov6[env_idx] = cov
        self._pose_stamp[env_idx] = time.time()
        self._cov_seq[env_idx] += 1

        #JUST ADDED for POSE COMING FROM SLAM
        # p = msg.pose.pose.position
        # o = msg.pose.pose.orientation
        # self._latest_vo_xy[env_idx] = (float(p.x), float(p.y))
        # self._latest_vo_yaw[env_idx] = _quat_wxyz_to_yaw(float(o.w), float(o.x), float(o.y), float(o.z))
        # self._vo_pose_stamp[env_idx] = time.time()
        #JUST ADDED for POSE COMING FROM SLAM


    def _publish_odom_tf(self, stamp):
        robot = self.scene["unitree_go2"].data
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            odom_frame = f"{ns}/odom"
            base_frame = f"{ns}/base_footprint"     #CHANGED FROM base_link

            pos = robot.root_state_w[i, :3]
            quat = robot.root_state_w[i, 3:7]  # (w,x,y,z)
            qn = _normalize_quat_wxyz(quat)
            if qn is None:
                continue

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = odom_frame
            t.child_frame_id = base_frame
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.w = float(qn[0])
            t.transform.rotation.x = float(qn[1])
            t.transform.rotation.y = float(qn[2])
            t.transform.rotation.z = float(qn[3])
            self.tf_broadcaster.sendTransform(t)


    def _publish_stereo_batch(self, stamp):                 #BEFORE NITROS
        rgbL_all = self.cam_left.data.output.get("rgb", None)
        rgbR_all = self.cam_right.data.output.get("rgb", None)
        if rgbL_all is None or rgbR_all is None:
            return
        _, H, W, _ = rgbL_all.shape
        H, W = int(H), int(W)

        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            frame_left = f"{ns}/left_camera_optical_frame"
            frame_right = f"{ns}/right_camera_optical_frame"

            rgb_l = _to_numpy_u8_rgb(self.cam_left.data.output["rgb"][i])
            rgb_r = _to_numpy_u8_rgb(self.cam_right.data.output["rgb"][i])
            mono_l = _rgb_to_mono8(rgb_l)
            mono_r = _rgb_to_mono8(rgb_r)

            msg_l = Image()
            msg_l.header.stamp = stamp
            msg_l.header.frame_id = frame_left
            msg_l.height = H
            msg_l.width = W
            msg_l.encoding = "mono8"
            msg_l.is_bigendian = 0
            msg_l.step = W
            msg_l.data = mono_l.tobytes()

            msg_r = Image()
            msg_r.header.stamp = stamp
            msg_r.header.frame_id = frame_right
            msg_r.height = H
            msg_r.width = W
            msg_r.encoding = "mono8"
            msg_r.is_bigendian = 0
            msg_r.step = W
            msg_r.data = mono_r.tobytes()

            fx_l, fy_l, cx_l, cy_l = _camera_K_from_sensor(self.cam_left, i, H, W)
            fx_r, fy_r, cx_r, cy_r = _camera_K_from_sensor(self.cam_right, i, H, W)

            ci_l = CameraInfo()
            ci_l.header.stamp = stamp
            ci_l.header.frame_id = frame_left
            ci_l.height = H
            ci_l.width = W
            ci_l.distortion_model = "plumb_bob"
            ci_l.k = [fx_l, 0.0, cx_l, 0.0, fy_l, cy_l, 0.0, 0.0, 1.0]
            ci_l.p = [fx_l, 0.0, cx_l, 0.0, 0.0, fy_l, cy_l, 0.0, 0.0, 0.0, 1.0, 0.0]
            ci_l.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ci_l.d = [0.0, 0.0, 0.0, 0.0, 0.0]

 
            baseline_m = 0.12  # your 0.06 left + 0.06 right
            Tx = -fx_r * baseline_m

            ci_r = CameraInfo()
            ci_r.header.stamp = stamp
            ci_r.header.frame_id = frame_right
            ci_r.height = H
            ci_r.width = W
            ci_r.distortion_model = "plumb_bob"
            ci_r.k = [fx_r, 0.0, cx_r, 0.0, fy_r, cy_r, 0.0, 0.0, 1.0]
            ci_r.p = [fx_r, 0.0, cx_r, Tx, 0.0, fy_r, cy_r, 0.0, 0.0, 0.0, 1.0, 0.0]
            ci_r.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ci_r.d = [0.0, 0.0, 0.0, 0.0, 0.0]

            self.pub_left_img[i].publish(msg_l)
            self.pub_right_img[i].publish(msg_r)
            self.pub_left_info[i].publish(ci_l)
            self.pub_right_info[i].publish(ci_r)

    # def _publish_stereo_batch(self, stamp):  #after nitros
    #     rgbL_all = self.cam_left.data.output.get("rgb", None)
    #     rgbR_all = self.cam_right.data.output.get("rgb", None)
    #     if rgbL_all is None or rgbR_all is None:
    #         return
    #     _, H, W, _ = rgbL_all.shape
    #     H, W = int(H), int(W)

    #     for i in range(self.num_envs):
    #         ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
    #         frame_left = f"{ns}/left_camera_optical_frame"
    #         frame_right = f"{ns}/right_camera_optical_frame"

    #         # Publish RGB8 and let GPU converters make mono8
    #         rgb_l = _to_numpy_u8_rgb(self.cam_left.data.output["rgb"][i])
    #         rgb_r = _to_numpy_u8_rgb(self.cam_right.data.output["rgb"][i])

    #         msg_l = Image()
    #         msg_l.header.stamp = stamp
    #         msg_l.header.frame_id = frame_left
    #         msg_l.height = H
    #         msg_l.width = W
    #         msg_l.encoding = "rgb8"
    #         msg_l.is_bigendian = 0
    #         msg_l.step = W * 3
    #         msg_l.data = rgb_l.tobytes()

    #         msg_r = Image()
    #         msg_r.header.stamp = stamp
    #         msg_r.header.frame_id = frame_right
    #         msg_r.height = H
    #         msg_r.width = W
    #         msg_r.encoding = "rgb8"
    #         msg_r.is_bigendian = 0
    #         msg_r.step = W * 3
    #         msg_r.data = rgb_r.tobytes()

    #         fx_l, fy_l, cx_l, cy_l = _camera_K_from_sensor(self.cam_left, i, H, W)
    #         fx_r, fy_r, cx_r, cy_r = _camera_K_from_sensor(self.cam_right, i, H, W)

    #         ci_l = CameraInfo()
    #         ci_l.header.stamp = stamp
    #         ci_l.header.frame_id = frame_left
    #         ci_l.height = H
    #         ci_l.width = W
    #         ci_l.distortion_model = "plumb_bob"
    #         ci_l.k = [fx_l, 0.0, cx_l, 0.0, fy_l, cy_l, 0.0, 0.0, 1.0]
    #         ci_l.p = [fx_l, 0.0, cx_l, 0.0, 0.0, fy_l, cy_l, 0.0, 0.0, 0.0, 1.0, 0.0]
    #         ci_l.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    #         ci_l.d = [0.0, 0.0, 0.0, 0.0, 0.0]

    #         baseline_m = 0.12  # 0.06 + 0.06 (adjust if you changed the camera offsets)
    #         Tx = -fx_r * baseline_m

    #         ci_r = CameraInfo()
    #         ci_r.header.stamp = stamp
    #         ci_r.header.frame_id = frame_right
    #         ci_r.height = H
    #         ci_r.width = W
    #         ci_r.distortion_model = "plumb_bob"
    #         ci_r.k = [fx_r, 0.0, cx_r, 0.0, fy_r, cy_r, 0.0, 0.0, 1.0]
    #         ci_r.p = [fx_r, 0.0, cx_r, Tx, 0.0, fy_r, cy_r, 0.0, 0.0, 0.0, 1.0, 0.0]
    #         ci_r.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    #         ci_r.d = [0.0, 0.0, 0.0, 0.0, 0.0]

    #         self.pub_left_img[i].publish(msg_l)
    #         self.pub_right_img[i].publish(msg_r)
    #         self.pub_left_info[i].publish(ci_l)
    #         self.pub_right_info[i].publish(ci_r)


    # def _maybe_publish_static_tf_from_scene(self):            #ALREADY PUBLISHING ON THE MY_VSLAM.LAUNCH.PY
    #     if self._tf_ready:
    #         return

    #     robot = self.scene["unitree_go2"].data
    #     try:
    #         if not (_torch_all_finite(robot.root_state_w) and
    #                 _torch_all_finite(self.cam_left.data.pos_w) and
    #                 _torch_all_finite(self.cam_right.data.pos_w) and
    #                 _torch_all_finite(self.cam_left.data.quat_w) and
    #                 _torch_all_finite(self.cam_right.data.quat_w)):
    #             return
    #     except Exception:
    #         return

    #     base_pos_w  = robot.root_state_w[:, :3]
    #     base_quat_w = robot.root_state_w[:, 3:7]        # [w,x,y,z]
    #     left_pos_w   = self.cam_left.data.pos_w
    #     right_pos_w  = self.cam_right.data.pos_w
    #     left_quat_w  = getattr(self.cam_left.data,  "quat_w_ros",  self.cam_left.data.quat_w)
    #     right_quat_w = getattr(self.cam_right.data, "quat_w_ros", self.cam_right.data.quat_w)

    #     msgs = []
    #     for i in range(self.num_envs):
    #         ns   = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
    #         base = f"{ns}/base_link"
    #         left = f"{ns}/left_camera_optical_frame"
    #         right= f"{ns}/right_camera_optical_frame"

    #         q_base  = _normalize_quat_wxyz(base_quat_w[i])
    #         q_left  = _normalize_quat_wxyz(left_quat_w[i])
    #         q_right = _normalize_quat_wxyz(right_quat_w[i])
    #         if q_base is None or q_left is None or q_right is None:
    #             return

    #         R_base = _quat_to_rotmat(q_base)   # base->world
    #         R_w2b  = R_base.transpose(0, 1)    # world->base

    #         t_b_l = torch.matmul(R_w2b, (left_pos_w[i]  - base_pos_w[i]))
    #         t_b_r = torch.matmul(R_w2b, (right_pos_w[i] - base_pos_w[i]))

    #         q_rel_l = _quat_mul(_quat_conj(q_base), q_left)
    #         q_rel_r = _quat_mul(_quat_conj(q_base), q_right)
    #         q_rel_l = _normalize_quat_wxyz(q_rel_l)
    #         q_rel_r = _normalize_quat_wxyz(q_rel_r)
    #         if q_rel_l is None or q_rel_r is None:
    #             return

    #         R_left    = _quat_to_rotmat(q_left)   # left_optical->world
    #         R_w2left  = R_left.transpose(0, 1)
    #         d_lr_w    = right_pos_w[i] - left_pos_w[i]
    #         d_lr_left = torch.matmul(R_w2left, d_lr_w)
    #         self._baseline_x[i] = float(d_lr_left[0])

    #         for child, t_vec, q in [(left, t_b_l, q_rel_l), (right, t_b_r, q_rel_r)]:
    #             t = TransformStamped()
    #             t.header.stamp = self.get_clock().now().to_msg()
    #             t.header.frame_id = base
    #             t.child_frame_id = child
    #             t.transform.translation.x = float(t_vec[0])
    #             t.transform.translation.y = float(t_vec[1])
    #             t.transform.translation.z = float(t_vec[2])
    #             t.transform.rotation.w = float(q[0])
    #             t.transform.rotation.x = float(q[1])
    #             t.transform.rotation.y = float(q[2])
    #             t.transform.rotation.z = float(q[3])
    #             msgs.append(t)

    #     self.tf_static_broadcaster.sendTransform(msgs)
    #     self._tf_ready = True

    def _maybe_create_ros_time_graph(self):
        try:
            keys = og.Controller.Keys
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [("PublishClock.inputs:topicName", "/clock")],
                },
            )
        except Exception:
            pass
