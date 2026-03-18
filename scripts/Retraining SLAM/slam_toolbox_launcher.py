#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""All-in-one slam_toolbox launcher for one or many namespaced robots.

Training-friendly defaults:
- process scans aggressively so PPO can get a fresh covariance every step,
- keep map publication relatively infrequent to save CPU,
- keep RViz-facing features optional,
- allow launching N independent namespaced slam_toolbox instances that match the
  Isaac-Lab bridge naming convention:
    * num_envs == 1  -> namespace "unitree_go2"
    * num_envs > 1   -> namespaces "unitree_go2_0" ... "unitree_go2_{N-1}"
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MULTI_NS_PREFIX = "unitree_go2"
DEFAULT_SINGLE_NS = "unitree_go2"
DEFAULT_BASE_FRAME_SUFFIX = "base_footprint"


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
    parts = ["true", f"source {shlex.quote(str(ros_setup))}"]
    if overlay:
        overlay_path = Path(overlay).expanduser().resolve()
        if not overlay_path.exists():
            raise RuntimeError(f"Overlay setup script not found: {overlay_path}")
        parts.append(f"source {shlex.quote(str(overlay_path))}")
    return "; ".join(parts)


def check_slam_toolbox_available(ros_distro: str, overlay: str | None):
    source_prefix = build_source_prefix(ros_distro, overlay)
    check_cmd = f'{source_prefix}; ros2 pkg prefix slam_toolbox >/dev/null 2>&1'
    proc = run(["bash", "-c", check_cmd], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "slam_toolbox is not available in the sourced ROS environment.\n"
            f"Install it with: sudo apt install ros-{ros_distro}-slam-toolbox\n"
            "or pass --overlay <workspace>/install/setup.bash if you built it from source."
        )


def _normalize_serialized_map_basename(path_str: str) -> str:
    p = Path(path_str).expanduser().resolve()
    s = str(p)
    for suffix in (".posegraph", ".data"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def build_slam_command(args, ns: str, base_frame: str) -> str:
    source_prefix = build_source_prefix(args.ros_distro, args.overlay)

    odom_frame = f"{ns}/odom"
    map_frame = f"{ns}/map"
    scan_topic = f"/{ns}/scan"
    full_node_name = f"/{ns}/slam_toolbox"

    if args.session == "localization":
        exe = "localization_slam_toolbox_node"
        mode_param = "localization"
    else:
        exe = "sync_slam_toolbox_node" if args.mode == "sync" else "async_slam_toolbox_node"
        mode_param = "mapping"

    ros_args = [
        "ros2", "run", "slam_toolbox", exe,
        "--ros-args",
        "-r", "__node:=slam_toolbox",
        "-r", f"__ns:=/{ns}",
        "-p", "solver_plugin:=solver_plugins::CeresSolver",
        "-p", "ceres_linear_solver:=SPARSE_NORMAL_CHOLESKY",
        "-p", "ceres_preconditioner:=SCHUR_JACOBI",
        "-p", "ceres_trust_strategy:=LEVENBERG_MARQUARDT",
        "-p", "ceres_dogleg_type:=TRADITIONAL_DOGLEG",
        "-p", f"ceres_loss_function:={args.ceres_loss_function}",
        "-p", f"use_sim_time:={'true' if args.use_sim_time else 'false'}",
        "-p", f"mode:={mode_param}",
        "-p", f"odom_frame:={odom_frame}",
        "-p", f"map_frame:={map_frame}",
        "-p", f"base_frame:={base_frame}",
        "-p", f"scan_topic:={scan_topic}",
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

    if args.session == "localization":
        serialized_map = _normalize_serialized_map_basename(args.serialized_map)
        ros_args += ["-p", f"map_file_name:={serialized_map}"]

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
        f"get_state(){{ ros2 lifecycle get {lifecycle_node} 2>/dev/null | tr '[:upper:]' '[:lower:]'; }}; "
        f"for i in $(seq 1 100); do st=$(get_state); [ -n \"$st\" ] && break; sleep 0.1; done; "
        "if echo \"$st\" | grep -q 'active'; then :; "
        "elif echo \"$st\" | grep -q 'inactive'; then transition_node activate || true; "
        "elif echo \"$st\" | grep -q 'unconfigured'; then transition_node configure || true; transition_node activate || true; "
        "else transition_node configure || true; transition_node activate || true; fi; "
        "wait $SLAM_PID"
    )


def launch_background_process(cmd_str: str, logfile: Path) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = open(logfile, "w")
    proc = subprocess.Popen(
        ["bash", "-lc", cmd_str],
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    fh.close()
    return int(proc.pid)


def resolve_targets(args):
    if args.single:
        num_envs = 1
    elif args.num_envs is not None:
        num_envs = int(args.num_envs)
    else:
        num_envs = 2

    if num_envs < 1:
        raise RuntimeError("--num-envs must be >= 1")

    default_ns0 = f"{args.ns_prefix}_0"
    default_ns1 = f"{args.ns_prefix}_1"
    default_base0 = f"{default_ns0}/{args.base_frame_suffix}"
    default_base1 = f"{default_ns1}/{args.base_frame_suffix}"

    targets = []
    if num_envs == 1:
        if args.ns0 != default_ns0:
            ns = args.ns0
        else:
            ns = args.single_ns
        if args.base0 != default_base0:
            base = args.base0
        else:
            base = f"{ns}/{args.base_frame_suffix}"
        targets.append((0, ns, base))
        return targets

    for i in range(num_envs):
        if i == 0 and args.ns0 != default_ns0:
            ns = args.ns0
        elif i == 1 and args.ns1 != default_ns1:
            ns = args.ns1
        else:
            ns = f"{args.ns_prefix}_{i}"

        if i == 0 and args.base0 != default_base0:
            base = args.base0
        elif i == 1 and args.base1 != default_base1:
            base = args.base1
        else:
            base = f"{ns}/{args.base_frame_suffix}"

        targets.append((i, ns, base))
    return targets


def main():
    parser = argparse.ArgumentParser(description="All-in-one slam_toolbox launcher for one or many namespaced robots.")
    parser.add_argument("--single", action="store_true", help="Compatibility shortcut for --num-envs 1.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of independent slam_toolbox instances to launch.")
    parser.add_argument("--spawn-mode", choices=["terminals", "background"], default="terminals", help="Open one terminal per slam_toolbox process or launch detached background processes with logs.")
    parser.add_argument("--logs-dir", default="slam_toolbox_logs", help="Used only with --spawn-mode background.")
    parser.add_argument("--stagger-seconds", type=float, default=0.10, help="Small delay between launches to reduce startup spikes.")

    parser.add_argument("--ns-prefix", default=DEFAULT_MULTI_NS_PREFIX, help="Prefix for multi-env namespaces. Multi-env launch uses <prefix>_0, <prefix>_1, ...")
    parser.add_argument("--single-ns", default=DEFAULT_SINGLE_NS, help="Namespace used when launching exactly one environment.")
    parser.add_argument("--base-frame-suffix", default=DEFAULT_BASE_FRAME_SUFFIX, help="Base-frame suffix appended to each namespace.")

    # Backward-compatible overrides for the first two envs.
    parser.add_argument("--ns0", default=f"{DEFAULT_MULTI_NS_PREFIX}_0", help="Optional explicit namespace override for env 0.")
    parser.add_argument("--ns1", default=f"{DEFAULT_MULTI_NS_PREFIX}_1", help="Optional explicit namespace override for env 1.")
    parser.add_argument("--base0", default=f"{DEFAULT_MULTI_NS_PREFIX}_0/{DEFAULT_BASE_FRAME_SUFFIX}", help="Optional explicit base_frame override for env 0.")
    parser.add_argument("--base1", default=f"{DEFAULT_MULTI_NS_PREFIX}_1/{DEFAULT_BASE_FRAME_SUFFIX}", help="Optional explicit base_frame override for env 1.")

    parser.add_argument("--ros-distro", default=os.environ.get("ROS_DISTRO", "humble"), help="ROS 2 distro to source.")
    parser.add_argument("--overlay", default=None, help="Optional setup.bash to source after /opt/ros/<distro>/setup.bash.")
    parser.add_argument("--session", choices=["mapping", "localization"], default="mapping")
    parser.add_argument("--mode", choices=["sync", "async"], default="async", help="Mapping node type. Ignored in localization mode.")
    parser.add_argument("--serialized-map", default=None)
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--map-update-interval", type=float, default=30.0, help="Slow this down during training to save CPU.")
    parser.add_argument("--transform-publish-period", type=float, default=0.02)
    parser.add_argument("--scan-queue-size", type=int, default=1)
    parser.add_argument("--min-laser-range", type=float, default=0.02)
    parser.add_argument("--max-laser-range", type=float, default=10.0)
    parser.add_argument("--minimum-time-interval", type=float, default=0.0, help="0.0 lets slam_toolbox accept back-to-back scans.")
    parser.add_argument("--minimum-travel-distance", type=float, default=0.0, help="0.0 is important if you want a fresh covariance every RL step.")
    parser.add_argument("--minimum-travel-heading", type=float, default=0.0, help="0.0 is important if you want a fresh covariance every RL step.")
    parser.add_argument("--transform-timeout", type=float, default=0.2)
    parser.add_argument("--tf-buffer-duration", type=float, default=30.0)
    parser.add_argument("--throttle-scans", type=int, default=1)
    parser.add_argument("--scan-buffer-size", type=int, default=1)
    parser.add_argument("--scan-buffer-maximum-scan-distance", type=float, default=10.0)
    parser.add_argument("--link-match-minimum-response-fine", type=float, default=0.1)
    parser.add_argument("--link-scan-maximum-distance", type=float, default=1.5)
    parser.add_argument("--loop-search-maximum-distance", type=float, default=3.0)
    parser.add_argument("--loop-match-minimum-chain-size", type=int, default=3)
    parser.add_argument("--loop-match-maximum-variance-coarse", type=float, default=3.0)
    parser.add_argument("--loop-match-minimum-response-coarse", type=float, default=0.35)
    parser.add_argument("--loop-match-minimum-response-fine", type=float, default=0.45)
    parser.add_argument("--correlation-search-space-dimension", type=float, default=0.5)
    parser.add_argument("--correlation-search-space-resolution", type=float, default=0.01)
    parser.add_argument("--correlation-search-space-smear-deviation", type=float, default=0.1)
    parser.add_argument("--loop-search-space-dimension", type=float, default=8.0)
    parser.add_argument("--loop-search-space-resolution", type=float, default=0.05)
    parser.add_argument("--loop-search-space-smear-deviation", type=float, default=0.03)
    parser.add_argument("--distance-variance-penalty", type=float, default=0.5)
    parser.add_argument("--angle-variance-penalty", type=float, default=1.0)
    parser.add_argument("--fine-search-angle-offset", type=float, default=0.00349)
    parser.add_argument("--coarse-search-angle-offset", type=float, default=0.349)
    parser.add_argument("--coarse-angle-resolution", type=float, default=0.0349)
    parser.add_argument("--minimum-angle-penalty", type=float, default=0.9)
    parser.add_argument("--minimum-distance-penalty", type=float, default=0.5)
    parser.add_argument("--min-pass-through", type=int, default=2)
    parser.add_argument("--occupancy-threshold", type=float, default=0.1)
    parser.add_argument("--ceres-loss-function", default="HuberLoss")
    parser.add_argument("--no-restamp-tf", action="store_true")
    parser.add_argument("--no-loop-closing", action="store_true")
    parser.add_argument("--no-scan-matching", action="store_true")
    parser.add_argument("--no-scan-barycenter", action="store_true")
    parser.add_argument("--no-use-map-saver", action="store_true")
    parser.add_argument("--enable-interactive-mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-response-expansion", action="store_true")
    args = parser.parse_args()

    if args.session == "localization" and not args.serialized_map:
        raise RuntimeError("Localization mode requires --serialized-map /path/to/your_serialized_map_basename")

    args.restamp_tf = not args.no_restamp_tf
    args.do_loop_closing = not args.no_loop_closing
    args.use_scan_matching = not args.no_scan_matching
    args.use_scan_barycenter = not args.no_scan_barycenter
    args.use_map_saver = not args.no_use_map_saver
    args.use_response_expansion = not args.no_response_expansion

    ensure_binary("ros2")
    check_slam_toolbox_available(args.ros_distro, args.overlay)

    if args.spawn_mode == "terminals":
        term, _ = pick_terminal()
        if not term:
            print("[error] No supported terminal emulator found.", file=sys.stderr)
            sys.exit(1)
    else:
        term = None

    targets = resolve_targets(args)

    print(f"[info] Using ROS distro: {args.ros_distro}")
    print(f"[info] Session mode: {args.session}")
    if args.session == "mapping":
        print(f"[info] Mapping node type: {args.mode}")
    else:
        print(f"[info] Serialized map basename: {_normalize_serialized_map_basename(args.serialized_map)}")
    print(f"[info] Launching {len(targets)} slam_toolbox instance(s) with spawn mode: {args.spawn_mode}")

    if args.spawn_mode == "background":
        log_root = Path(args.logs_dir).expanduser().resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        print(f"[info] Background logs/pids will be written under: {log_root}")
    else:
        log_root = None
        print(f"[info] Using terminal emulator: {term}")

    launched = []
    for env_idx, ns, base in targets:
        cmd = build_slam_command(args, ns, base)
        print(f"[info] env={env_idx} ns={ns} base_frame={base}")
        if args.spawn_mode == "terminals":
            open_terminal(term, cmd)
        else:
            logfile = log_root / f"{ns}.log"
            pidfile = log_root / f"{ns}.pid"
            pid = launch_background_process(cmd, logfile)
            pidfile.write_text(f"{pid}\n", encoding="utf-8")
            launched.append((ns, pid, logfile, pidfile))
        if args.stagger_seconds > 0.0:
            time.sleep(float(args.stagger_seconds))

    print("[done] Launch complete.")
    if args.spawn_mode == "background" and launched:
        print("[info] Background processes:")
        for ns, pid, logfile, pidfile in launched:
            print(f"  - {ns}: pid={pid}  log={logfile}  pidfile={pidfile}")
        print("[info] To stop them later:")
        print("""pgrep -a -f '(async|sync|localization)_slam_toolbox_node'; \
pkill -INT -f '(async|sync|localization)_slam_toolbox_node'; \
sleep 1; \
pgrep -a -f '(async|sync|localization)_slam_toolbox_node' && \
pkill -TERM -f '(async|sync|localization)_slam_toolbox_node'; \
sleep 0.5; \
pgrep -a -f '(async|sync|localization)_slam_toolbox_node' && \
pkill -KILL -f '(async|sync|localization)_slam_toolbox_node'""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user.")
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
