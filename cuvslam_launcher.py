#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""All-in-one slam_toolbox launcher that opens GUI terminals for one or two robots.

This keeps your original filename for convenience, but it now launches slam_toolbox
instead of Isaac ROS Visual SLAM.

The defaults in this version are tuned to be friendlier for interactive RViz mapping:
- async mode by default,
- map/raster limits matched to the simulated 10 m lidar,
- matcher / loop-closure params biased toward conservative loop closure in simulation,
- lifecycle transitions retried so transient "Node not found" races are less likely,
- per-robot map topics are forced with map_name:=/<ns>/map so two robots do not
  collide on the global /map topic,
- optional RViz launch with an optional saved .rviz config.

Default: two-agent
  - env0: ns:=unitree_go2_0 base_frame:=unitree_go2_0/base_footprint
  - env1: ns:=unitree_go2_1 base_frame:=unitree_go2_1/base_footprint

--single: one-agent
  - env0 auto-swaps to ns:=unitree_go2 base_frame:=unitree_go2/base_footprint

Usage examples:
  python3 cuvslam_launcher.py
  python3 cuvslam_launcher.py --single
  python3 cuvslam_launcher.py --mode sync
  python3 cuvslam_launcher.py --single --rviz
  python3 cuvslam_launcher.py --rviz --rviz-config ~/my_slam.rviz
  python3 cuvslam_launcher.py --overlay ~/my_ws/install/setup.bash
"""

import argparse
import os
import shlex
import subprocess
import sys
import shutil
from pathlib import Path


def run(cmd, check=True, capture=False, shell=False):
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True
    if isinstance(cmd, list):
        proc = subprocess.run(cmd, **kwargs)
    else:
        proc = subprocess.run(cmd if shell else shlex.split(cmd), shell=shell, **kwargs)
    if check and proc.returncode != 0:
        if capture:
            raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\nOutput:\n{proc.stdout}")
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")
    return proc


def which(name):
    return shutil.which(name)


def ensure_binary(name, env_var=None, fallback=None):
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    path = which(fallback or name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH. Install it or set {env_var}.")
    return path


def pick_terminal():
    for term in ["gnome-terminal", "konsole", "xfce4-terminal", "xterm"]:
        path = which(term)
        if path:
            return term, path
    return None, None


def open_terminal(term, cmd_str):
    # Keep the window open after the process exits, but avoid re-triggering
    # user shell startup hooks (some Conda plugin setups print unrelated errors).
    hold_tail = "; echo; echo [done]; exec bash --noprofile --norc"
    if term == "gnome-terminal":
        return run(f'gnome-terminal -- bash -c {shlex.quote(cmd_str + hold_tail)}', check=False, shell=True)
    if term == "konsole":
        return run(f'konsole -e bash -c {shlex.quote(cmd_str + hold_tail)}', check=False, shell=True)
    if term == "xfce4-terminal":
        return run(
            f'xfce4-terminal --hold --command {shlex.quote("bash -c " + shlex.quote(cmd_str + hold_tail))}',
            check=False,
            shell=True,
        )
    if term == "xterm":
        return run(f'xterm -hold -e {shlex.quote("bash -c " + shlex.quote(cmd_str + hold_tail))}', check=False, shell=True)
    raise RuntimeError("No supported terminal emulator found. Install gnome-terminal or konsole.")


def build_source_prefix(ros_distro: str, overlay: str | None) -> str:
    ros_setup = Path(f"/opt/ros/{ros_distro}/setup.bash")
    if not ros_setup.exists():
        raise RuntimeError(f"ROS setup script not found: {ros_setup}")

    parts = [
        "true",
        f"source {shlex.quote(str(ros_setup))}",
    ]
    if overlay:
        overlay_path = Path(overlay).expanduser().resolve()
        if not overlay_path.exists():
            raise RuntimeError(f"Overlay setup script not found: {overlay_path}")
        parts.append(f"source {shlex.quote(str(overlay_path))}")
    return "; ".join(parts)


def check_slam_toolbox_available(ros_distro: str, overlay: str | None):
    source_prefix = build_source_prefix(ros_distro, overlay)
    check_cmd = f"{source_prefix}; ros2 pkg prefix slam_toolbox >/dev/null 2>&1"
    proc = run(["bash", "-c", check_cmd], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "slam_toolbox is not available in the sourced ROS environment.\n"
            f"Install it with: sudo apt install ros-{ros_distro}-slam-toolbox\n"
            "or pass --overlay <workspace>/install/setup.bash if you built it from source."
        )


def check_rviz_available(ros_distro: str, overlay: str | None):
    source_prefix = build_source_prefix(ros_distro, overlay)
    check_cmd = f"{source_prefix}; command -v rviz2 >/dev/null 2>&1"
    proc = run(["bash", "-c", check_cmd], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "rviz2 is not available in the sourced ROS environment.\n"
            f"Install it with: sudo apt install ros-{ros_distro}-rviz2"
        )


def build_slam_command(args, ns: str, base_frame: str) -> str:
    exe = "sync_slam_toolbox_node" if args.mode == "sync" else "async_slam_toolbox_node"
    source_prefix = build_source_prefix(args.ros_distro, args.overlay)

    odom_frame = f"{ns}/odom"
    map_frame = f"{ns}/map"
    scan_topic = f"/{ns}/scan"
    map_topic = f"/{ns}/map"
    full_node_name = f"/{ns}/slam_toolbox"

    ros_args = [
        "ros2",
        "run",
        "slam_toolbox",
        exe,
        "--ros-args",
        "-r", "__node:=slam_toolbox",
        "-r", f"__ns:=/{ns}",
        # Solver params.
        "-p", "solver_plugin:=solver_plugins::CeresSolver",
        "-p", "ceres_linear_solver:=SPARSE_NORMAL_CHOLESKY",
        "-p", "ceres_preconditioner:=SCHUR_JACOBI",
        "-p", "ceres_trust_strategy:=LEVENBERG_MARQUARDT",
        "-p", "ceres_dogleg_type:=TRADITIONAL_DOGLEG",
        "-p", f"ceres_loss_function:={args.ceres_loss_function}",
        # Core frames / topics.
        "-p", f"use_sim_time:={'true' if args.use_sim_time else 'false'}",
        "-p", "mode:=mapping",
        "-p", f"odom_frame:={odom_frame}",
        "-p", f"map_frame:={map_frame}",
        "-p", f"base_frame:={base_frame}",
        "-p", f"scan_topic:={scan_topic}",
        # IMPORTANT:
        # slam_toolbox defaults map_name to /map (absolute), which causes all robots
        # to publish into the same global topic. Force a unique map topic per namespace.
        "-p", f"map_name:={map_topic}",
        # Toolbox params.
        "-p", f"scan_queue_size:={int(args.scan_queue_size)}",
        "-p", f"restamp_tf:={'true' if args.restamp_tf else 'false'}",
        "-p", f"transform_publish_period:={float(args.transform_publish_period)}",
        "-p", f"map_update_interval:={float(args.map_update_interval)}",
        "-p", f"resolution:={float(args.resolution)}",
        "-p", f"min_laser_range:={float(args.min_laser_range)}",
        "-p", f"max_laser_range:={float(args.max_laser_range)}",
        "-p", f"minimum_time_interval:={float(args.minimum_time_interval)}",
        "-p", f"transform_timeout:={float(args.transform_timeout)}",
        "-p", f"tf_buffer_duration:={float(args.tf_buffer_duration)}",
        "-p", "stack_size_to_use:=40000000",
        "-p", f"debug_logging:={'true' if args.debug else 'false'}",
        "-p", f"use_map_saver:={'true' if args.use_map_saver else 'false'}",
        "-p", f"enable_interactive_mode:={'true' if args.enable_interactive_mode else 'false'}",
        "-p", f"throttle_scans:={int(args.throttle_scans)}",
        # Matching / graph params.
        "-p", f"use_scan_matching:={'true' if args.use_scan_matching else 'false'}",
        "-p", f"use_scan_barycenter:={'true' if args.use_scan_barycenter else 'false'}",
        "-p", f"minimum_travel_distance:={float(args.minimum_travel_distance)}",
        "-p", f"minimum_travel_heading:={float(args.minimum_travel_heading)}",
        "-p", "check_min_dist_and_heading_precisely:=false",
        "-p", f"scan_buffer_size:={int(args.scan_buffer_size)}",
        "-p", f"scan_buffer_maximum_scan_distance:={float(args.scan_buffer_maximum_scan_distance)}",
        "-p", f"link_match_minimum_response_fine:={float(args.link_match_minimum_response_fine)}",
        "-p", f"link_scan_maximum_distance:={float(args.link_scan_maximum_distance)}",
        "-p", f"loop_search_maximum_distance:={float(args.loop_search_maximum_distance)}",
        "-p", f"do_loop_closing:={'true' if args.do_loop_closing else 'false'}",
        "-p", f"loop_match_minimum_chain_size:={int(args.loop_match_minimum_chain_size)}",
        "-p", f"loop_match_maximum_variance_coarse:={float(args.loop_match_maximum_variance_coarse)}",
        "-p", f"loop_match_minimum_response_coarse:={float(args.loop_match_minimum_response_coarse)}",
        "-p", f"loop_match_minimum_response_fine:={float(args.loop_match_minimum_response_fine)}",
        # Correlation / scan matcher params.
        "-p", f"correlation_search_space_dimension:={float(args.correlation_search_space_dimension)}",
        "-p", f"correlation_search_space_resolution:={float(args.correlation_search_space_resolution)}",
        "-p", f"correlation_search_space_smear_deviation:={float(args.correlation_search_space_smear_deviation)}",
        "-p", f"loop_search_space_dimension:={float(args.loop_search_space_dimension)}",
        "-p", f"loop_search_space_resolution:={float(args.loop_search_space_resolution)}",
        "-p", f"loop_search_space_smear_deviation:={float(args.loop_search_space_smear_deviation)}",
        "-p", f"distance_variance_penalty:={float(args.distance_variance_penalty)}",
        "-p", f"angle_variance_penalty:={float(args.angle_variance_penalty)}",
        "-p", f"fine_search_angle_offset:={float(args.fine_search_angle_offset)}",
        "-p", f"coarse_search_angle_offset:={float(args.coarse_search_angle_offset)}",
        "-p", f"coarse_angle_resolution:={float(args.coarse_angle_resolution)}",
        "-p", f"minimum_angle_penalty:={float(args.minimum_angle_penalty)}",
        "-p", f"minimum_distance_penalty:={float(args.minimum_distance_penalty)}",
        "-p", f"use_response_expansion:={'true' if args.use_response_expansion else 'false'}",
        "-p", f"min_pass_through:={int(args.min_pass_through)}",
        "-p", f"occupancy_threshold:={float(args.occupancy_threshold)}",
    ]
    ros_cmd = " ".join(shlex.quote(x) for x in ros_args)

    lifecycle_node = shlex.quote(full_node_name)
    return (
        f"{source_prefix}; "
        f"{ros_cmd} & "
        "SLAM_PID=$!; "
        "cleanup(){ kill $SLAM_PID 2>/dev/null || true; wait $SLAM_PID 2>/dev/null || true; }; "
        "trap cleanup EXIT INT TERM; "
        "transition_node(){ local action=\"$1\"; "
        f"for i in $(seq 1 100); do ros2 lifecycle set {lifecycle_node} \"$action\" >/dev/null 2>&1 && return 0; sleep 0.1; done; "
        f"echo \"[warn] Failed to transition {full_node_name} -> $action\"; return 1; }}; "
        f"for i in $(seq 1 100); do ros2 lifecycle get {lifecycle_node} >/dev/null 2>&1 && break; sleep 0.1; done; "
        "transition_node configure || true; "
        "transition_node activate || true; "
        "wait $SLAM_PID"
    )


def build_rviz_command(args) -> str:
    source_prefix = build_source_prefix(args.ros_distro, args.overlay)

    rviz_args = ["rviz2"]
    if args.rviz_config:
        rviz_cfg = Path(args.rviz_config).expanduser().resolve()
        if not rviz_cfg.exists():
            raise RuntimeError(f"RViz config not found: {rviz_cfg}")
        rviz_args += ["-d", str(rviz_cfg)]

    return f"{source_prefix}; " + " ".join(shlex.quote(x) for x in rviz_args)


def main():
    parser = argparse.ArgumentParser(description="All-in-one slam_toolbox launcher for one or two namespaced robots.")
    parser.add_argument("--single", action="store_true", help="Launch only one terminal (env0).")
    parser.add_argument("--ns0", default="unitree_go2_0", help="Namespace for env 0.")
    parser.add_argument("--ns1", default="unitree_go2_1", help="Namespace for env 1.")
    parser.add_argument("--base0", default="unitree_go2_0/base_footprint", help="base_frame for env 0.")
    parser.add_argument("--base1", default="unitree_go2_1/base_footprint", help="base_frame for env 1.")
    parser.add_argument("--ros-distro", default=os.environ.get("ROS_DISTRO", "humble"), help="ROS 2 distro to source.")
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional setup.bash to source after /opt/ros/<distro>/setup.bash (for source builds).",
    )
    parser.add_argument("--mode", choices=["sync", "async"], default="async", help="slam_toolbox node type.")
    parser.add_argument("--use-sim-time", action="store_true", help="Set use_sim_time:=true. Keep this off unless /clock is published.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Map resolution in meters/cell.")
    parser.add_argument("--map-update-interval", type=float, default=2.0, help="Occupancy grid update interval in seconds.")
    parser.add_argument("--transform-publish-period", type=float, default=0.02, help="map->odom TF publish period.")
    parser.add_argument("--scan-queue-size", type=int, default=1, help="slam_toolbox scan queue size.")
    parser.add_argument("--min-laser-range", type=float, default=0.02, help="slam_toolbox min_laser_range for occupancy rasterization.")
    parser.add_argument(
        "--max-laser-range",
        type=float,
        default=10.0,
        help="slam_toolbox max_laser_range parameter. Match this to your simulated LiDAR when possible.",
    )
    parser.add_argument(
        "--minimum-time-interval",
        type=float,
        default=0.2,
        help="Minimum interval between accepted scans in seconds.",
    )
    parser.add_argument(
        "--minimum-travel-distance",
        type=float,
        default=0.20,
        help=(
            "Minimum translation before accepting a new pose node. "
            "Raised from the previous 0.10 m to reduce overly dense pose graphs and "
            "make loop closure more conservative."
        ),
    )
    parser.add_argument(
        "--minimum-travel-heading",
        type=float,
        default=0.20,
        help=(
            "Minimum heading change (rad) before accepting a new pose node. "
            "Raised from the previous 0.10 rad to reduce near-duplicate scan nodes."
        ),
    )
    parser.add_argument(
        "--no-restamp-tf",
        dest="restamp_tf",
        action="store_false",
        help="Do not restamp slam_toolbox map->odom TF with the current time.",
    )
    parser.add_argument("--no-loop-closing", dest="do_loop_closing", action="store_false", help="Disable loop closing.")
    parser.add_argument("--debug", action="store_true", help="Enable slam_toolbox debug logging.")
    parser.add_argument("--rviz", action="store_true", help="Also launch RViz2 in a separate terminal.")
    parser.add_argument("--rviz-config", default=None, help="Optional .rviz config file to load.")
    # More conservative matcher defaults for simulated robots where false loop closures can damage the map.
    parser.add_argument("--use-map-saver", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--enable-interactive-mode", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--throttle-scans", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--use-scan-matching", dest="use_scan_matching", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-scan-matching", dest="use_scan_matching", action="store_false", help="Disable scan matching. In some simulator setups with very good odometry, this can improve map quality.")
    parser.add_argument("--use-scan-barycenter", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--scan-buffer-size", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--scan-buffer-maximum-scan-distance", type=float, default=10.0, help=argparse.SUPPRESS)
    parser.add_argument("--link-match-minimum-response-fine", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--link-scan-maximum-distance", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--loop-search-maximum-distance", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--loop-match-minimum-chain-size", type=int, default=15, help=argparse.SUPPRESS)
    parser.add_argument("--loop-match-maximum-variance-coarse", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--loop-match-minimum-response-coarse", type=float, default=0.45, help=argparse.SUPPRESS)
    parser.add_argument("--loop-match-minimum-response-fine", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--correlation-search-space-dimension", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--correlation-search-space-resolution", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--correlation-search-space-smear-deviation", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--loop-search-space-dimension", type=float, default=5.0, help=argparse.SUPPRESS)
    parser.add_argument("--loop-search-space-resolution", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--loop-search-space-smear-deviation", type=float, default=0.03, help=argparse.SUPPRESS)
    parser.add_argument("--distance-variance-penalty", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--angle-variance-penalty", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--fine-search-angle-offset", type=float, default=0.00349, help=argparse.SUPPRESS)
    parser.add_argument("--coarse-search-angle-offset", type=float, default=0.349, help=argparse.SUPPRESS)
    parser.add_argument("--coarse-angle-resolution", type=float, default=0.0349, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-angle-penalty", type=float, default=0.9, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-distance-penalty", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--use-response-expansion", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--min-pass-through", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--occupancy-threshold", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--transform-timeout", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--tf-buffer-duration", type=float, default=30.0, help=argparse.SUPPRESS)
    parser.add_argument("--ceres-loss-function", default="HuberLoss", choices=["None", "HuberLoss", "CauchyLoss"], help=argparse.SUPPRESS)
    parser.set_defaults(do_loop_closing=True, restamp_tf=True)
    args = parser.parse_args()

    DEFAULT_NS0 = "unitree_go2_0"
    DEFAULT_BASE0 = "unitree_go2_0/base_footprint"
    if args.single:
        if args.ns0 == DEFAULT_NS0:
            args.ns0 = "unitree_go2"
        if args.base0 == DEFAULT_BASE0:
            args.base0 = "unitree_go2/base_footprint"

    ensure_binary("bash", fallback="bash")
    term, _term_path = pick_terminal()

    # Validate the ROS environment before opening terminals.
    check_slam_toolbox_available(args.ros_distro, args.overlay)
    if args.rviz:
        check_rviz_available(args.ros_distro, args.overlay)

    cmd0 = build_slam_command(args, args.ns0, args.base0)
    cmd1 = build_slam_command(args, args.ns1, args.base1)
    rviz_cmd = build_rviz_command(args) if args.rviz else None

    if not term:
        print("[error] No supported terminal emulator found (gnome-terminal/konsole/xfce4-terminal/xterm).", file=sys.stderr)
        print("Manual commands you can run in host terminals:", file=sys.stderr)
        print("Terminal 0:", cmd0, file=sys.stderr)
        if not args.single:
            print("Terminal 1:", cmd1, file=sys.stderr)
        if rviz_cmd:
            print("RViz:", rviz_cmd, file=sys.stderr)
        sys.exit(1)

    print(f"[info] Using ROS distro: {args.ros_distro}")
    if args.overlay:
        print(f"[info] Sourcing overlay: {Path(args.overlay).expanduser().resolve()}")
    print(f"[info] Launch mode: {args.mode}")
    print("[info] Conservative loop-closure defaults are active (stricter closure checks + Huber loss).")
    print(f"[info] Launching {'one' if args.single else 'two'} SLAM {term} window(s)...")
    if args.rviz:
        print(f"[info] Launching RViz2 in an additional {term} window...")

    open_terminal(term, cmd0)
    if not args.single:
        open_terminal(term, cmd1)
    if rviz_cmd:
        open_terminal(term, rviz_cmd)

    print("[done] Launch complete.")
    print("Topics / frames expected by the bridge and SLAM:")
    print(f"  - /{args.ns0}/scan   with frames {args.ns0}/odom -> {args.base0} -> {args.ns0}/lidar_frame")
    print(f"  - /{args.ns0}/odom")
    print(f"  - /{args.ns0}/map")
    print(f"  - /{args.ns0}/trajectory")
    if not args.single:
        print(f"  - /{args.ns1}/scan   with frames {args.ns1}/odom -> {args.base1} -> {args.ns1}/lidar_frame")
        print(f"  - /{args.ns1}/odom")
        print(f"  - /{args.ns1}/map")
        print(f"  - /{args.ns1}/trajectory")

    print("RViz2 setup:")
    if args.single:
        print(f"  - Global Options > Fixed Frame: {args.ns0}/map")
        print(f"  - Add a Map display on topic: /{args.ns0}/map")
        print(f"  - Add a Path display on topic: /{args.ns0}/trajectory")
        print(f"  - Optionally add LaserScan on topic: /{args.ns0}/scan")
        print(f"  - Optionally add Odometry on topic: /{args.ns0}/odom")
    else:
        print("  - Use the matching <ns>/map frame for the robot you want to visualize")
        print("  - Add a Map display on topic: /<ns>/map")
        print("  - Add a Path display on topic: /<ns>/trajectory")
        print("  - Optionally add LaserScan on topic: /<ns>/scan")
        print("  - Optionally add Odometry on topic: /<ns>/odom")
        print("  - Note: RViz can use only one Fixed Frame at a time, so switch the Fixed Frame")
        print("    between unitree_go2_0/map and unitree_go2_1/map depending on which robot/map")
        print("    you want to inspect in that RViz window.")

    if args.rviz and args.rviz_config:
        print(f"  - RViz was launched with config: {Path(args.rviz_config).expanduser().resolve()}")
    elif args.rviz:
        print("  - RViz was launched without a saved config, so add the displays above once and save the config for reuse.")
        print("  - If the map topic exists but RViz still shows nothing, set the Map display QoS to:")
        print("    Reliability = Reliable, Durability = Transient Local")

    print("Tip: the SLAM play script should usually be run one episode at a time for a clean map.")
    print("Tip: if false loop closures still damage the map, the next easy test is: --no-scan-matching")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user.")
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
