"""This module contains the bridge "Isaac Lab - ROS 2" configuration."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Iterable

import numpy as np

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

try:
    from slam_toolbox.srv import Reset as SlamToolboxReset
except Exception:
    SlamToolboxReset = None

try:
    from slam_toolbox.srv import ClearQueue as SlamToolboxClearQueue
except Exception:
    SlamToolboxClearQueue = None


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _stamp_to_ns(stamp) -> int:
    try:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except Exception:
        return 0


def _quat_wxyz_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    return _quat_multiply(_quat_multiply(q, vq), _quat_conjugate(q))[1:]


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _yaw_to_quat_msg(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(0.5 * yaw)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(0.5 * yaw)
    return q


def _quat_wxyz_to_msg(q: np.ndarray) -> Quaternion:
    out = Quaternion()
    out.w = float(q[0])
    out.x = float(q[1])
    out.y = float(q[2])
    out.z = float(q[3])
    return out


def _as_numpy_1d(x, env_idx: int) -> np.ndarray:
    try:
        row = x[env_idx]
    except Exception:
        row = x
    if hasattr(row, "detach"):
        row = row.detach().cpu().numpy()
    return np.asarray(row, dtype=np.float64).reshape(-1)


def _get_first_existing_attr(obj, names: Iterable[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"None of the attributes {tuple(names)} exist on {type(obj).__name__}.")


def _scene_lookup(scene, key: str):
    try:
        return scene[key]
    except Exception:
        pass
    for container_name in (
        "articulations",
        "rigid_objects",
        "rigid_object_collections",
        "deformable_objects",
    ):
        container = getattr(scene, container_name, None)
        if container is None:
            continue
        try:
            if key in container:
                return container[key]
        except Exception:
            pass
        try:
            return getattr(container, key)
        except Exception:
            pass
    return None


def _iter_container_values(container):
    if container is None:
        return []
    try:
        return list(container.values())
    except Exception:
        pass
    try:
        return [v for _, v in container.items()]
    except Exception:
        pass
    if isinstance(container, (list, tuple)):
        return list(container)
    return []


def _resolve_robot_asset(scene):
    for key in ("robot", "unitree_go2", "go2", "turtlebot3_burger"):
        asset = _scene_lookup(scene, key)
        if asset is not None and hasattr(getattr(asset, "data", None), "root_pos_w"):
            return asset, key

    for container_name in (
        "articulations",
        "rigid_objects",
        "rigid_object_collections",
        "deformable_objects",
    ):
        container = getattr(scene, container_name, None)
        for asset in _iter_container_values(container):
            if hasattr(getattr(asset, "data", None), "root_pos_w"):
                return asset, container_name

    raise RuntimeError("Unable to locate the robot asset in the Isaac Lab scene.")


class RobotDataManager(Node):
    def __init__(
        self,
        env,
        *,
        attempt_hard_reset_on_env_reset: bool = True,
        clear_queue_on_env_reset: bool = True,
        publish_trajectory: bool = False,
    ):
        super().__init__("robot_data_manager")

        self.env = env
        self.scene = env.unwrapped.scene
        self.num_envs = self.scene.num_envs
        self._attempt_hard_reset_on_env_reset = bool(attempt_hard_reset_on_env_reset)
        self._clear_queue_on_env_reset = bool(clear_queue_on_env_reset)
        self._publish_trajectory = bool(publish_trajectory)

        self.lidar_name = "horizontal_scanner_1"
        if self.lidar_name not in self.scene.sensors:
            raise RuntimeError(
                f"Missing LiDAR sensor '{self.lidar_name}' in scene.sensors. "
                "Create a LiDAR sensor and name it 'horizontal_scanner_1'."
            )
        self.lidar = self.scene.sensors[self.lidar_name]
        self.robot, self.robot_key = _resolve_robot_asset(self.scene)

        RELIABLE_QOS = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        PATH_QOS = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_scan = []
        self.pub_odom = []
        self.pub_path = []
        self._reset_clients = []
        self._clear_queue_clients = []
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        self._latest_slam_xy = [None] * self.num_envs
        self._latest_slam_yaw = [None] * self.num_envs
        self._latest_slam_planar_cov = [None] * self.num_envs
        self._slam_pose_stamp = [0.0] * self.num_envs
        self._min_pose_stamp_ns = [0] * self.num_envs

        self._odom_seq = [0] * self.num_envs
        self._xy_at_odom_seq = [None] * self.num_envs
        self._odom_lock = threading.Lock()

        self._last_lidar_sensor_stamp = [-1.0] * self.num_envs
        self._published_static_lidar_tf = [False] * self.num_envs
        self._odom_origin_xyyaw = [None] * self.num_envs
        self._path_buffers = [deque(maxlen=5000) for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            ns = self._ns(i)
            self.pub_scan.append(self.create_publisher(LaserScan, f"{ns}/scan", RELIABLE_QOS))
            self.pub_odom.append(self.create_publisher(Odometry, f"{ns}/odom", RELIABLE_QOS))

            if self._publish_trajectory:
                self.pub_path.append(self.create_publisher(Path, f"{ns}/trajectory", PATH_QOS))
            else:
                self.pub_path.append(None)

            self.create_subscription(
                PoseWithCovarianceStamped,
                f"{ns}/pose",
                lambda msg, env_idx=i: self._slam_pose_cb(msg, env_idx),
                RELIABLE_QOS,
            )

            if self._attempt_hard_reset_on_env_reset and SlamToolboxReset is not None:
                self._reset_clients.append(self.create_client(SlamToolboxReset, f"{ns}/slam_toolbox/reset"))
            else:
                self._reset_clients.append(None)

            if self._clear_queue_on_env_reset and SlamToolboxClearQueue is not None:
                self._clear_queue_clients.append(self.create_client(SlamToolboxClearQueue, f"{ns}/slam_toolbox/clear_queue"))
            else:
                self._clear_queue_clients.append(None)

    def pub_ros2_data(self, force_scan_envs=None) -> list[bool]:
        stamp = self.get_clock().now().to_msg()
        self._publish_odom_and_tf_batch(stamp)
        return self._publish_scan_batch(stamp, force_scan_envs=force_scan_envs)

    def reset_odometry_reference(self, env_ids=None):
        ids = self._normalize_env_ids(env_ids)
        for i in ids:
            pos_xy, yaw = self._get_robot_pose_local_xyyaw(i)
            self._odom_origin_xyyaw[i] = (float(pos_xy[0]), float(pos_xy[1]), float(yaw))

    def clear_trajectory(self, env_ids=None):
        ids = self._normalize_env_ids(env_ids)
        for i in ids:
            self._path_buffers[i].clear()
        if not self._publish_trajectory:
            return
        stamp = self.get_clock().now().to_msg()
        for i in ids:
            if self.pub_path[i] is None:
                continue
            empty_path = Path()
            empty_path.header.stamp = stamp
            empty_path.header.frame_id = f"{self._ns(i)}/map"
            empty_path.poses = []
            self.pub_path[i].publish(empty_path)

    def _call_service_best_effort(self, cli, request, timeout_sec: float = 0.5) -> bool:
        if cli is None:
            return False
        try:
            if not cli.service_is_ready():
                cli.wait_for_service(timeout_sec=timeout_sec)
            if not cli.service_is_ready():
                return False
            fut = cli.call_async(request)
            t0 = time.time()
            while not fut.done() and (time.time() - t0) < timeout_sec:
                time.sleep(0.01)
            return bool(fut.done())
        except Exception:
            return False

    def reset_slam(self, env_ids=None, pause_after_reset: bool = False):
        ids = self._normalize_env_ids(env_ids)
        cutoff_ns = _stamp_to_ns(self.get_clock().now().to_msg())

        self.reset_odometry_reference(ids)
        for i in ids:
            self._latest_slam_xy[i] = None
            self._latest_slam_yaw[i] = None
            self._latest_slam_planar_cov[i] = None
            self._slam_pose_stamp[i] = 0.0
            self._min_pose_stamp_ns[i] = cutoff_ns
            self._last_lidar_sensor_stamp[i] = -1.0
            with self._odom_lock:
                self._xy_at_odom_seq[i] = None

        self.clear_trajectory(ids)

        if self._attempt_hard_reset_on_env_reset and SlamToolboxReset is not None:
            for i in ids:
                cli = self._reset_clients[i]
                if cli is None:
                    continue
                req = SlamToolboxReset.Request()
                if hasattr(req, "pause_new_measurements"):
                    req.pause_new_measurements = bool(pause_after_reset)
                self._call_service_best_effort(cli, req, timeout_sec=0.75)

        if self._clear_queue_on_env_reset and SlamToolboxClearQueue is not None:
            for i in ids:
                cli = self._clear_queue_clients[i]
                if cli is None:
                    continue
                self._call_service_best_effort(cli, SlamToolboxClearQueue.Request(), timeout_sec=0.50)

        cutoff_ns = _stamp_to_ns(self.get_clock().now().to_msg())
        for i in ids:
            self._min_pose_stamp_ns[i] = cutoff_ns

    def get_slam_xy_at_last_odom(self, env_idx: int = 0):
        with self._odom_lock:
            return self._xy_at_odom_seq[env_idx]

    def get_slam_xy_latest(self, env_idx: int = 0):
        return self._latest_slam_xy[env_idx]

    def get_latest_slam_yaw(self, env_idx: int = 0):
        return self._latest_slam_yaw[env_idx]

    def get_latest_slam_planar_cov(self, env_idx: int = 0):
        with self._odom_lock:
            cov = self._latest_slam_planar_cov[env_idx]
            return None if cov is None else cov.copy()

    def get_odom_seq_and_xy(self, env_idx: int = 0):
        with self._odom_lock:
            return self._odom_seq[env_idx], self._xy_at_odom_seq[env_idx]

    def get_odom_seq_xy_cov(self, env_idx: int = 0):
        with self._odom_lock:
            xy = self._xy_at_odom_seq[env_idx]
            cov = self._latest_slam_planar_cov[env_idx]
            if cov is not None:
                cov = cov.copy()
            return self._odom_seq[env_idx], xy, cov

    def get_odom_seq_xy_yaw_cov(self, env_idx: int = 0):
        with self._odom_lock:
            xy = self._xy_at_odom_seq[env_idx]
            cov = self._latest_slam_planar_cov[env_idx]
            if cov is not None:
                cov = cov.copy()
            yaw = self._latest_slam_yaw[env_idx]
            return self._odom_seq[env_idx], xy, yaw, cov

    def _snapshot_odom_seqs(self):
        with self._odom_lock:
            return list(self._odom_seq)

    def wait_for_new_odom(self, env_ids, last_seqs, timeout_s: float | None):
        ids = self._normalize_env_ids(env_ids)
        t0 = time.time()
        while True:
            seqs = self._snapshot_odom_seqs()
            if all(seqs[i] > int(last_seqs[i]) for i in ids):
                return seqs, False
            if timeout_s is not None and (time.time() - t0) >= float(timeout_s):
                return seqs, True
            time.sleep(0.001)

    def _graph_topic_names(self, env_idx: int) -> dict[str, str]:
        ns = self._ns(env_idx)
        return {
            "scan": f"/{ns}/scan",
            "pose": f"/{ns}/pose",
            "odom": f"/{ns}/odom",
        }

    def get_toolbox_graph_status(self, env_idx: int) -> dict[str, int]:
        topics = self._graph_topic_names(env_idx)
        status = {
            "scan_subscribers": 0,
            "pose_publishers": 0,
            "odom_subscribers": 0,
            "tf_subscribers": 0,
        }
        try:
            status["scan_subscribers"] = int(self.count_subscribers(topics["scan"]))
        except Exception:
            pass
        try:
            status["pose_publishers"] = int(self.count_publishers(topics["pose"]))
        except Exception:
            pass
        try:
            status["odom_subscribers"] = int(self.count_subscribers(topics["odom"]))
        except Exception:
            pass
        try:
            status["tf_subscribers"] = int(self.count_subscribers("/tf"))
        except Exception:
            pass
        return status

    def wait_for_toolbox_graph(self, env_ids=None, timeout_s: float = 10.0):
        ids = self._normalize_env_ids(env_ids)
        t0 = time.time()
        last = {i: self.get_toolbox_graph_status(i) for i in ids}
        while True:
            ready = True
            for i in ids:
                st = self.get_toolbox_graph_status(i)
                last[i] = st
                if st["pose_publishers"] <= 0 or st["scan_subscribers"] <= 0:
                    ready = False
            if ready:
                return True, last
            if timeout_s is not None and (time.time() - t0) >= float(timeout_s):
                return False, last
            time.sleep(0.05)

    def _normalize_env_ids(self, env_ids) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if hasattr(env_ids, "tolist"):
            env_ids = env_ids.tolist()
        return [int(i) for i in env_ids]

    def _ns(self, env_idx: int) -> str:
        return "unitree_go2" if self.num_envs == 1 else f"unitree_go2_{env_idx}"

    def _slam_pose_cb(self, msg: PoseWithCovarianceStamped, env_idx: int):
        msg_stamp_ns = _stamp_to_ns(msg.header.stamp)
        cutoff_ns = int(self._min_pose_stamp_ns[env_idx])
        if msg_stamp_ns > 0 and cutoff_ns > 0 and msg_stamp_ns < cutoff_ns:
            return

        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        frame_id = (msg.header.frame_id or f"{self._ns(env_idx)}/map").lstrip("/")

        planar_cov = None
        try:
            cov6 = np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
            planar_cov = cov6[np.ix_([0, 1, 5], [0, 1, 5])]
            planar_cov = 0.5 * (planar_cov + planar_cov.T)
            if not np.all(np.isfinite(planar_cov)):
                planar_cov = None
        except Exception:
            planar_cov = None

        self._latest_slam_xy[env_idx] = (float(p.x), float(p.y))
        self._latest_slam_yaw[env_idx] = _quat_wxyz_to_yaw(float(o.w), float(o.x), float(o.y), float(o.z))
        self._slam_pose_stamp[env_idx] = time.time()

        with self._odom_lock:
            self._odom_seq[env_idx] += 1
            self._xy_at_odom_seq[env_idx] = (float(p.x), float(p.y))
            self._latest_slam_planar_cov[env_idx] = None if planar_cov is None else planar_cov.copy()

        if not self._publish_trajectory or self.pub_path[env_idx] is None:
            return

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(p.x)
        pose.pose.position.y = float(p.y)
        pose.pose.position.z = float(p.z)
        pose.pose.orientation.x = float(o.x)
        pose.pose.orientation.y = float(o.y)
        pose.pose.orientation.z = float(o.z)
        pose.pose.orientation.w = float(o.w)
        self._path_buffers[env_idx].append(pose)

        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = frame_id
        path.poses = list(self._path_buffers[env_idx])
        self.pub_path[env_idx].publish(path)

    def _get_env_origin(self, env_idx: int) -> np.ndarray:
        env_origins = getattr(self.scene, "env_origins", None)
        if env_origins is None:
            return np.zeros(3, dtype=np.float64)
        return _as_numpy_1d(env_origins, env_idx)[:3]

    def _get_robot_pose_local_xyyaw(self, env_idx: int) -> tuple[np.ndarray, float]:
        data = self.robot.data
        pos_attr = _get_first_existing_attr(data, ("root_pos_w", "root_link_pos_w"))
        quat_attr = _get_first_existing_attr(data, ("root_quat_w", "root_link_quat_w"))

        pos_w = _as_numpy_1d(pos_attr, env_idx)[:3]
        quat_w = _as_numpy_1d(quat_attr, env_idx)[:4]
        pos_local = pos_w - self._get_env_origin(env_idx)
        yaw = _quat_wxyz_to_yaw(float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3]))
        return pos_local[:2], yaw

    def _get_robot_state_local(self, env_idx: int):
        data = self.robot.data
        pos_attr = _get_first_existing_attr(data, ("root_pos_w", "root_link_pos_w"))
        quat_attr = _get_first_existing_attr(data, ("root_quat_w", "root_link_quat_w"))
        lin_attr = _get_first_existing_attr(data, ("root_lin_vel_w", "root_link_lin_vel_w"))
        ang_attr = _get_first_existing_attr(data, ("root_ang_vel_w", "root_link_ang_vel_w"))

        pos_w = _as_numpy_1d(pos_attr, env_idx)[:3]
        quat_w = _as_numpy_1d(quat_attr, env_idx)[:4]
        lin_vel_w = _as_numpy_1d(lin_attr, env_idx)[:3]
        ang_vel_w = _as_numpy_1d(ang_attr, env_idx)[:3]

        pos_local = pos_w - self._get_env_origin(env_idx)
        yaw = _quat_wxyz_to_yaw(float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3]))
        return pos_local, quat_w, yaw, lin_vel_w, ang_vel_w

    def _get_lidar_state_local(self, env_idx: int):
        data = self.lidar.data
        pos_w = _as_numpy_1d(data.pos_w, env_idx)[:3]
        quat_w = _as_numpy_1d(data.quat_w, env_idx)[:4]
        pos_local = pos_w - self._get_env_origin(env_idx)
        return pos_local, quat_w

    def _project_pose_to_episode_odom(self, pos_xy: np.ndarray, yaw: float, env_idx: int) -> tuple[float, float, float]:
        origin = self._odom_origin_xyyaw[env_idx]
        if origin is None:
            self._odom_origin_xyyaw[env_idx] = (float(pos_xy[0]), float(pos_xy[1]), float(yaw))
            origin = self._odom_origin_xyyaw[env_idx]

        ox, oy, oyaw = origin
        dx_w = float(pos_xy[0]) - ox
        dy_w = float(pos_xy[1]) - oy

        c0 = math.cos(oyaw)
        s0 = math.sin(oyaw)
        x_rel = c0 * dx_w + s0 * dy_w
        y_rel = -s0 * dx_w + c0 * dy_w
        yaw_rel = _wrap_to_pi(float(yaw) - oyaw)
        return x_rel, y_rel, yaw_rel

    def _publish_odom_and_tf_batch(self, stamp):
        for i in range(self.num_envs):
            ns = self._ns(i)
            odom_frame = f"{ns}/odom"
            base_frame = f"{ns}/base_footprint"
            lidar_frame = f"{ns}/lidar_frame"

            base_pos_local, _base_quat_w, base_yaw, lin_vel_w, ang_vel_w = self._get_robot_state_local(i)
            x_rel, y_rel, yaw_rel = self._project_pose_to_episode_odom(base_pos_local[:2], base_yaw, i)

            cy = math.cos(base_yaw)
            sy = math.sin(base_yaw)
            vx_b = cy * float(lin_vel_w[0]) + sy * float(lin_vel_w[1])
            vy_b = -sy * float(lin_vel_w[0]) + cy * float(lin_vel_w[1])
            wz_b = float(ang_vel_w[2])

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = odom_frame
            odom.child_frame_id = base_frame
            odom.pose.pose.position.x = float(x_rel)
            odom.pose.pose.position.y = float(y_rel)
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = _yaw_to_quat_msg(yaw_rel)
            odom.twist.twist.linear.x = vx_b
            odom.twist.twist.linear.y = vy_b
            odom.twist.twist.linear.z = 0.0
            odom.twist.twist.angular.x = 0.0
            odom.twist.twist.angular.y = 0.0
            odom.twist.twist.angular.z = wz_b
            self.pub_odom[i].publish(odom)

            tf_odom_base = TransformStamped()
            tf_odom_base.header.stamp = stamp
            tf_odom_base.header.frame_id = odom_frame
            tf_odom_base.child_frame_id = base_frame
            tf_odom_base.transform.translation.x = float(x_rel)
            tf_odom_base.transform.translation.y = float(y_rel)
            tf_odom_base.transform.translation.z = 0.0
            tf_odom_base.transform.rotation = _yaw_to_quat_msg(yaw_rel)
            self._tf_broadcaster.sendTransform(tf_odom_base)

            if not self._published_static_lidar_tf[i]:
                lidar_pos_local, lidar_quat_w = self._get_lidar_state_local(i)
                base_q_flat = _yaw_to_quat_wxyz(base_yaw)
                q_rel = _quat_multiply(_quat_conjugate(base_q_flat), lidar_quat_w)
                p_rel = _quat_rotate(_quat_conjugate(base_q_flat), lidar_pos_local - base_pos_local)

                tf_base_lidar = TransformStamped()
                tf_base_lidar.header.stamp = stamp
                tf_base_lidar.header.frame_id = base_frame
                tf_base_lidar.child_frame_id = lidar_frame
                tf_base_lidar.transform.translation.x = float(p_rel[0])
                tf_base_lidar.transform.translation.y = float(p_rel[1])
                tf_base_lidar.transform.translation.z = float(p_rel[2])
                tf_base_lidar.transform.rotation = _quat_wxyz_to_msg(q_rel)
                self._static_tf_broadcaster.sendTransform(tf_base_lidar)
                self._published_static_lidar_tf[i] = True

    def _lidar_has_fresh_sample(self, env_idx: int) -> bool:
        sensor_ts = getattr(self.lidar, "_timestamp", None)
        if sensor_ts is None:
            return True
        try:
            ts = float(sensor_ts[env_idx])
        except Exception:
            try:
                ts = float(sensor_ts[env_idx].item())
            except Exception:
                return True
        if ts <= self._last_lidar_sensor_stamp[env_idx] + 1e-12:
            return False
        self._last_lidar_sensor_stamp[env_idx] = ts
        return True

    def _publish_scan_batch(self, stamp, force_scan_envs=None) -> list[bool]:
        _ = self.lidar.data
        force_ids = set(self._normalize_env_ids(force_scan_envs)) if force_scan_envs is not None else set()
        published = [False] * self.num_envs
        for i in range(self.num_envs):
            if i not in force_ids and not self._lidar_has_fresh_sample(i):
                continue
            scan_msg = self._build_scan_msg(i, stamp)
            if scan_msg is None:
                continue
            self.pub_scan[i].publish(scan_msg)
            published[i] = True
        return published

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
            expected_per_ring = int(round((float(pcfg.horizontal_fov_range[1]) - float(pcfg.horizontal_fov_range[0])) / ang_res_deg)) + 1
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

        ns = self._ns(env_idx)
        scan = LaserScan()
        scan.header.frame_id = f"{ns}/lidar_frame"
        scan.header.stamp = stamp
        scan.angle_min = ang_min
        scan.angle_max = ang_max
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = float(self.lidar.cfg.update_period) if self.lidar.cfg.update_period else 0.0
        scan.range_min = 0.02
        scan.range_max = max_range
        scan.ranges = ranges.tolist()
        return scan
