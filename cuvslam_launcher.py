#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
All-in-one cuVSLAM launcher that opens GUI terminals for the launches.

Default: two-agent
  - env0: ns:=unitree_go2_0 base_frame:=unitree_go2_0/base_footprint
  - env1: ns:=unitree_go2_1 base_frame:=unitree_go2_1/base_footprint

--single: one-agent
  - env0 auto-swaps to ns:=unitree_go2 base_frame:=unitree_go2/base_footprint

Overview:
  1) Start the Isaac ROS dev container via run_dev.sh
  2) Inside the container (as user 'admin'):
       - source ~/.bashrc.
       - sudo apt-get update and install cuVSLAM required packages.
       - source /opt/ros/humble/setup.bash
       - colcon build --symlink-install --packages-select my_vslam_bringup
  3) Open GUI terminal(s), each running ros2 launch under the 'admin' user.

  # two-agent (default):
  python3 cuvslam_launcher.py

  # one-agent:
  python3 cuvslam_launcher.py --single
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import shutil
from pathlib import Path

ADMIN_USER = "admin"
ADMIN_HOME = f"/home/{ADMIN_USER}"

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
        else:
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

def detect_container_name(DOCKER, container_hint, patterns):
    if container_hint:
        try:
            ps = run(f"{DOCKER} ps --filter name={shlex.quote(container_hint)} --format '{{{{.Names}}}}'", capture=True)
            if any(line.strip() == container_hint for line in ps.stdout.strip().splitlines()):
                return container_hint
        except Exception:
            pass
    ps = run(f"{DOCKER} ps --format '{{{{.Names}}}}'", capture=True).stdout.strip().splitlines()
    for name in ps:
        for pat in patterns:
            if re.search(pat, name, re.IGNORECASE):
                return name
    return None

def docker_exec(DOCKER, container, cmd, tty=False, user=None, env_extra=None, check=True):
    flags = "-it" if tty else "-i"
    user_part = f"--user {shlex.quote(user)}" if user else ""
    env_part = ""
    if env_extra:
        for k, v in env_extra.items():
            env_part += f" -e {shlex.quote(k)}={shlex.quote(v)}"
    wrapped = f'{DOCKER} exec {flags} {user_part}{env_part} {shlex.quote(container)} bash -lc {shlex.quote(cmd)}'
    return run(wrapped, check=check)

def pick_terminal():
    # Prefer gnome-terminal on Ubuntu 22.04
    for term in ["gnome-terminal", "konsole", "xfce4-terminal", "xterm"]:
        path = which(term)
        if path:
            return term, path
    return None, None

def open_terminal(term, cmd_str):
    hold_tail = "; echo; echo [done]; exec bash"
    if term == "gnome-terminal":
        return run(f'gnome-terminal -- bash -lc {shlex.quote(cmd_str + hold_tail)}', check=False, shell=True)
    elif term == "konsole":
        return run(f'konsole -e bash -lc {shlex.quote(cmd_str + hold_tail)}', check=False, shell=True)
    elif term == "xfce4-terminal":
        return run(f'xfce4-terminal --hold --command {shlex.quote("bash -lc " + shlex.quote(cmd_str + hold_tail))}', check=False, shell=True)
    elif term == "xterm":
        return run(f'xterm -hold -e {shlex.quote("bash -lc " + shlex.quote(cmd_str + hold_tail))}', check=False, shell=True)
    else:
        raise RuntimeError("No supported terminal emulator found. Install gnome-terminal or konsole.")

def main():
    parser = argparse.ArgumentParser(description="All-in-one cuVSLAM launcher (container + setup + GUI terminals).")
    parser.add_argument("--container", default=None, help="Container name (if you know it). Otherwise auto-detect.")
    parser.add_argument("--single", action="store_true", help="Launch only one terminal (env0).")
    parser.add_argument("--ns0", default="unitree_go2_0", help="Namespace for env 0.")
    parser.add_argument("--ns1", default="unitree_go2_1", help="Namespace for env 1.")
    parser.add_argument("--base0", default="unitree_go2_0/base_footprint", help="base_frame for env 0.")
    parser.add_argument("--base1", default="unitree_go2_1/base_footprint", help="base_frame for env 1.")
    parser.add_argument("--ws", default=None, help="HOST path to isaac_ros-dev (for run_dev.sh). Defaults to <script_dir>/workspaces/isaac_ros-dev.")
    parser.add_argument("--cws", default="/workspaces/isaac_ros-dev", help="CONTAINER path to isaac_ros-dev (for cd/source/build).")
    args = parser.parse_args()

    # Auto-swap env0 to unitree_go2 when --single is used.
    DEFAULT_NS0 = "unitree_go2_0"
    DEFAULT_BASE0 = "unitree_go2_0/base_footprint"
    if args.single:
        if args.ns0 == DEFAULT_NS0:
            args.ns0 = "unitree_go2"
        if args.base0 == DEFAULT_BASE0:
            args.base0 = "unitree_go2/base_footprint"

    DOCKER = os.environ.get("DOCKER_BIN", None) or ensure_binary("docker", env_var="DOCKER_BIN", fallback="docker")
    TMUX = os.environ.get("TMUX_BIN", None) or ensure_binary("tmux", env_var="TMUX_BIN", fallback="tmux")

    script_dir = Path(__file__).resolve().parent
    default_ws = script_dir / "workspaces" / "isaac_ros-dev"
    HOST_WORKSPACE_DIR = Path(args.ws).expanduser().resolve() if args.ws else default_ws.resolve()
    RUN_DEV = HOST_WORKSPACE_DIR / "src/isaac_ros_common/scripts/run_dev.sh"

    CONTAINER_MATCHES = [r"isaac[_\-]?ros.*dev", r"ros.*humble.*dev"]
    APT_PACKAGES = ["ros-humble-isaac-ros-visual-slam", "ros-humble-isaac-ros-visual-slam-interfaces"]
    COLCON_PKGS_SELECT = "my_vslam_bringup"
    LAUNCH_PKG = "my_vslam_bringup"
    LAUNCH_FILE = "my_vslam.launch.py"
    CONTAINER_WS = args.cws.rstrip("/")

    # 1) Start / detect container.
    container = detect_container_name(DOCKER, args.container, CONTAINER_MATCHES)
    if not container:
        if not RUN_DEV.exists():
            raise RuntimeError(f"Couldn't find run_dev.sh at: {RUN_DEV}\n"
                               f"- pass --ws /path/to/workspaces/isaac_ros-dev OR start your container manually.")
        print(f"[info] No running container detected. Starting via: {RUN_DEV}")
        run(f'{TMUX} new-session -d -s cuvslam-setup "{shlex.quote(str(RUN_DEV))}"', check=True)
        print("[info] Waiting for container to start...")
        for _ in range(90):
            container = detect_container_name(DOCKER, args.container, CONTAINER_MATCHES)
            if container:
                print(f"[info] Detected container: {container}")
                break
            time.sleep(2)
        if not container:
            raise RuntimeError("Timed out waiting for the dev container to start.")
    else:
        print(f"[info] Detected running container: {container}")

    print(f"[info] Host workspace:      {HOST_WORKSPACE_DIR}")
    print(f"[info] Container workspace: {CONTAINER_WS}")

    # 2) Setup inside container (as "admin"): apt + build with ROS env.
    print("[info] Installing cuVSLAM packages inside container (as admin)...")
    apt_cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "sudo apt-get update && sudo apt-get install -y " + " ".join(APT_PACKAGES)
    )
    docker_exec(DOCKER, container, apt_cmd, tty=False, user=ADMIN_USER, env_extra={"HOME": ADMIN_HOME})

    print("[info] Building with colcon inside container (as admin)...")
    build_cmd = (
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "source /opt/ros/humble/setup.bash; "
        f"cd {shlex.quote(CONTAINER_WS)}; "
        f"colcon build --symlink-install --packages-select {COLCON_PKGS_SELECT}"
    )
    docker_exec(DOCKER, container, build_cmd, tty=False, user=ADMIN_USER, env_extra={"HOME": ADMIN_HOME})

    # 3) Open GUI terminal(s) attached to the container.
    term, term_path = pick_terminal()
    if not term:
        print("[error] No supported terminal emulator found (gnome-terminal/konsole/xfce4-terminal/xterm).", file=sys.stderr)
        print("Manual commands you can run in host terminals:", file=sys.stderr)
        base_launch = lambda ns, base: (
            f'{DOCKER} exec -it --user {ADMIN_USER} -e HOME={ADMIN_HOME} {container} '
            f'bash -lic "source ~/.bashrc >/dev/null 2>&1 || true; '
            f'source /opt/ros/humble/setup.bash; '
            f'cd {CONTAINER_WS}; '
            f'source install/setup.bash || true; '
            f'ros2 launch {LAUNCH_PKG} {LAUNCH_FILE} ns:={ns} base_frame:={base}"'
        )
        print("Terminal 0:", base_launch(args.ns0, args.base0), file=sys.stderr)
        if not args.single:
            print("Terminal 1:", base_launch(args.ns1, args.base1), file=sys.stderr)
        sys.exit(1)

    print(f"[info] Launching {'one' if args.single else 'two'} {term} window(s) (as admin, interactive login shells)...")

    def base_launch(ns, base):
        return (
            f'{DOCKER} exec -it --user {ADMIN_USER} -e HOME={ADMIN_HOME} {shlex.quote(container)} '
            f'bash -lic "source ~/.bashrc >/dev/null 2>&1 || true; '
            f'source /opt/ros/humble/setup.bash; '
            f'cd {shlex.quote(CONTAINER_WS)}; '
            f'source install/setup.bash || true; '
            f'ros2 launch {LAUNCH_PKG} {LAUNCH_FILE} ns:={ns} base_frame:={base}"'
        )

    # env0 (always)
    open_terminal(term, base_launch(args.ns0, args.base0))
    # env1 (only if not --single)
    if not args.single:
        open_terminal(term, base_launch(args.ns1, args.base1))

    print("[done] Launch complete.")
    if args.single:
        print("Opened one terminal (env0). Use --ns0/--base0 to customize.")
    else:
        print("Opened two terminals (env0 & env1). Use --single to launch only env0.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user.")
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

