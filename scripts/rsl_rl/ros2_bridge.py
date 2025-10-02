"""This module contains the bridge "Isaac Lab - ROS 2" configuration."""

from __future__ import annotations

import time
import math
import numpy as np
import torch

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo

from nav_msgs.msg import Odometry

# For resetting SLAM.
from isaac_ros_visual_slam_interfaces.srv import Reset

# For slam_toolbox occupancy grid.
from sensor_msgs.msg import LaserScan

# Will be used to avoid one-frame lag.
import threading

##
# Helpers.
##


# Get yaw from quaternion.
def _quat_wxyz_to_yaw(qw, qx, qy, qz):
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)


# ROS 2 requires uint8 on the CPU.
def _to_numpy_u8_rgb(img_t: torch.Tensor) -> np.ndarray:
    arr = img_t
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,W,C) tensor, got shape {tuple(arr.shape)}")
    if arr.dtype != torch.uint8:
        arr = (arr.clamp(0, 1) * 255.0).to(torch.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.detach().cpu().numpy()


# Conversion needed for cuVSLAM. It is faster doing it here than sending RGB.
def _rgb_to_mono8(rgb_u8: np.ndarray) -> np.ndarray:
    y = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.uint8)
    return y


def _camera_K_from_sensor(sensor, i: int, H: int, W: int):
    K = getattr(sensor.data, "intrinsic_matrices", None)
    if K is not None:
        Kn = K[i].detach().cpu().numpy().reshape(3, 3).astype(float)
        return float(Kn[0, 0]), float(Kn[1, 1]), float(Kn[0, 2]), float(Kn[1, 2])
    cfg = sensor.cfg.spawn
    fx = W * float(cfg.focal_length) / float(cfg.horizontal_aperture)
    fy = (H * float(cfg.focal_length) / float(getattr(cfg, "vertical_aperture", 0.0))) if getattr(cfg, "vertical_aperture", 0.0) else fx * (H / float(W))
    cx, cy = W * 0.5, H * 0.5
    return fx, fy, cx, cy


##
# ROS 2 bridge node.
##


class RobotDataManager(Node):
    """
    For each env (namespace unitree_go2_i):
      Publishes:
        - visual_slam/image_0 and visual_slam/camera_info_0.
        - visual_slam/image_1 and visual_slam/camera_info_1.
      Subscribes:
        - visual_slam/tracking/odometry.

    Optional (currently turned on):
      Publishes:
        - scan (for slam_toolbox occupancy grid).
    """

    def __init__(self, env):
        super().__init__("robot_data_manager")

        self.env = env
        self.scene = env.unwrapped.scene
        self.num_envs = self.scene.num_envs

        # Camera sensors.
        try:
            self.cam_left = self.scene.sensors["cam_left"]
            self.cam_right = self.scene.sensors["cam_right"]
        except KeyError as e:
            raise RuntimeError(
                f"Missing sensor '{e.args[0]}' in scene.sensors. "
                "Add two Camera sensors named cam_left/cam_right."
            )

        # For slam_toolbox occupancy grid.
        self.lidar_name = "horizontal_scanner_1"
        if self.lidar_name not in self.scene.sensors:
            raise RuntimeError(
                f"Missing LiDAR sensor '{self.lidar_name}' in scene.sensors. "
                "Create a LiDAR sensor and name it 'horizontal_scanner_1'"
            )
        self.lidar = self.scene.sensors[self.lidar_name]

        # For resetting SLAM.
        self._reset_clients = []
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            self._reset_clients.append(self.create_client(Reset, f"{ns}/visual_slam/reset"))

        # QoS.
        IMG_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        INFO_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Publishers lists.
        self.pub_left_img, self.pub_right_img = [], []
        self.pub_left_info, self.pub_right_info = [], []
        # For slam_toolbox occupancy grid.
        self.pub_scan = []

        # SLAM.
        self._latest_slam_xy = [None] * self.num_envs
        self._latest_slam_yaw = [None] * self.num_envs
        self._slam_pose_stamp = [0.0] * self.num_envs

        # Publishers.
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            self.pub_left_img.append(self.create_publisher(Image, f"{ns}/visual_slam/image_0", IMG_QOS))
            self.pub_right_img.append(self.create_publisher(Image, f"{ns}/visual_slam/image_1", IMG_QOS))
            self.pub_left_info.append(self.create_publisher(CameraInfo, f"{ns}/visual_slam/camera_info_0", INFO_QOS))
            self.pub_right_info.append(self.create_publisher(CameraInfo, f"{ns}/visual_slam/camera_info_1", INFO_QOS))
            # For slam_toolbox occupancy grid.
            self.pub_scan.append(self.create_publisher(LaserScan, f"{ns}/scan", QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)))

        # Subscribers.
        for i in range(self.num_envs):
            ns = "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{i}"
            self.create_subscription(
                Odometry,
                f"{ns}/visual_slam/tracking/odometry",
                lambda msg, env_idx=i: self._slam_odom_cb(msg, env_idx),
                QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )

        # Publish rates.
        # For slam_toolbox occupancy grid.
        self.scan_hz = 1.0 / float(self.lidar.cfg.update_period) if self.lidar.cfg.update_period else 15.0
        self._last_scan_pub_wall = 0.0

        # cuVSLAM publishes odom, so we set False here to make it explicit. Not used but good as a reminder.
        self.publish_sim_odom_tf = False

        # To avoid one-frame lag.
        self._odom_seq = [0] * self.num_envs
        self._xy_at_odom_seq = [None] * self.num_envs
        self._odom_lock = threading.Lock()

    def pub_ros2_data(self):
        """Publish stereo and scan. Call from the sim thread after each env.step()."""

        # Gets a ROS 2 timestamp.
        stamp = self.get_clock().now().to_msg()

        self._publish_stereo_batch(stamp)
        # For slam_toolbox occupancy grid.
        now_wall = time.time()
        if (now_wall - self._last_scan_pub_wall) >= (1.0 / self.scan_hz):
            self._last_scan_pub_wall = now_wall
            self._publish_scan_batch(stamp)

    # For slam_toolbox occupancy grid.
    def _publish_scan_batch(self, stamp):
        for i in range(self.num_envs):
            scan_msg = self._build_scan_msg(i, stamp)
            if scan_msg is not None:
                self.pub_scan[i].publish(scan_msg)

    def _build_scan_msg(self, env_idx: int, stamp) -> LaserScan | None:
        hits_w = self.lidar.data.ray_hits_w
        pos_w = self.lidar.data.pos_w
        if hits_w is None or pos_w is None:
            return None

        hits = hits_w[env_idx].detach().cpu().numpy()
        origin = pos_w[env_idx].detach().cpu().numpy()

        d = np.linalg.norm(hits - origin[None, :], axis=1)
        max_range = float(self.lidar.cfg.max_distance)
        d = np.where(np.isfinite(d), d, max_range)

        pcfg = self.lidar.cfg.pattern_cfg
        ang_min = math.radians(float(pcfg.horizontal_fov_range[0]))
        ang_max_cfg = math.radians(float(pcfg.horizontal_fov_range[1]))
        ang_res_deg = float(pcfg.horizontal_res)
        angle_increment = math.radians(ang_res_deg)
        channels = int(pcfg.channels)

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

    # For resetting SLAM.
    def reset_slam(self, env_ids=None):
        ids = range(self.num_envs) if env_ids is None else (env_ids.tolist() if hasattr(env_ids, "tolist") else env_ids)
        # Clear cached data.
        for i in ids:
            self._latest_slam_xy[i] = None
            self._slam_pose_stamp[i] = 0.0
            self._odom_seq[i] = 0
            self._xy_at_odom_seq[i] = None

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

    def _slam_odom_cb(self, msg: Odometry, env_idx: int):
        # Gets latest odometry sent by CuVSLAM message.
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation

        self._latest_slam_xy[env_idx] = (float(p.x), float(p.y))
        self._latest_slam_yaw[env_idx] = _quat_wxyz_to_yaw(float(o.w), float(o.x), float(o.y), float(o.z))
        self._slam_pose_stamp[env_idx] = time.time()

        with self._odom_lock:
            self._odom_seq[env_idx] += 1
            self._xy_at_odom_seq[env_idx] = (float(p.x), float(p.y))

    def get_slam_xy_at_last_odom(self, env_idx: int = 0):
        with self._odom_lock:
            return self._xy_at_odom_seq[env_idx]

    def get_slam_xy_latest(self, env_idx: int = 0):
        return self._latest_slam_xy[env_idx]

    def get_odom_seq_and_xy(self, env_idx: int = 0):
        with self._odom_lock:
            return self._odom_seq[env_idx], self._xy_at_odom_seq[env_idx]

    def _snapshot_odom_seqs(self):
        with self._odom_lock:
            return list(self._odom_seq)

    def _publish_stereo_batch(self, stamp):
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

            baseline_m = 0.12       # 0.06 from left camera + 0.06 from right camera.
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
