#!/usr/bin/env python3
"""
Bi3D from Unitree GO2 visual_slam topics.

Inputs (from Unitree):
  /unitree_go2/visual_slam/image_rgb_0
  /unitree_go2/visual_slam/camera_info_0
  /unitree_go2/visual_slam/image_rgb_1
  /unitree_go2/visual_slam/camera_info_1

Remapped to what Bi3D expects:
  left_image_bi3d        -> /left/resize/image
  right_image_bi3d       -> /right/resize/image
  left_camera_info_bi3d  -> /left/resize/camera_info
  right_camera_info_bi3d -> /right/resize/camera_info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # --- CLI-configurable args ---
    featnet = LaunchConfiguration("featnet_engine_file_path")
    segnet = LaunchConfiguration("segnet_engine_file_path")
    disparity_yaml = LaunchConfiguration("disparity_values_yaml")  # YAML string, e.g. "[2,4,6,8]"
    use_freespace = LaunchConfiguration("use_freespace")

    left_w  = LaunchConfiguration("left_width")
    left_h  = LaunchConfiguration("left_height")
    right_w = LaunchConfiguration("right_width")
    right_h = LaunchConfiguration("right_height")

    declare_args = [
        DeclareLaunchArgument("featnet_engine_file_path"),
        DeclareLaunchArgument("segnet_engine_file_path"),
        DeclareLaunchArgument("disparity_values_yaml", default_value="[2,4,6,8]"),
        DeclareLaunchArgument("left_width",  default_value="256"),
        DeclareLaunchArgument("left_height", default_value="160"),
        DeclareLaunchArgument("right_width",  default_value="256"),
        DeclareLaunchArgument("right_height", default_value="160"),
        DeclareLaunchArgument("use_freespace", default_value="false"),
    ]

    def launch_setup(context, *args, **kwargs):
        # Left resize (subscribes to Unitree topics)
        left_resize = ComposableNode(
            package="isaac_ros_image_proc",
            plugin="nvidia::isaac_ros::image_proc::ResizeNode",
            name="left_resize",
            namespace="left",
            remappings=[
                ("image", "/unitree_go2/visual_slam/image_rgb_0"),
                ("camera_info", "/unitree_go2/visual_slam/camera_info_0"),
            ],
            parameters=[{
                # include both key variants for compatibility across releases
                "output_width":  ParameterValue(left_w,  value_type=int),
                "output_height": ParameterValue(left_h,  value_type=int),
                "width":         ParameterValue(left_w,  value_type=int),
                "height":        ParameterValue(left_h,  value_type=int),
            }],
        )

        # Right resize
        right_resize = ComposableNode(
            package="isaac_ros_image_proc",
            plugin="nvidia::isaac_ros::image_proc::ResizeNode",
            name="right_resize",
            namespace="right",
            remappings=[
                ("image", "/unitree_go2/visual_slam/image_rgb_1"),
                ("camera_info", "/unitree_go2/visual_slam/camera_info_1"),
            ],
            parameters=[{
                "output_width":  ParameterValue(right_w, value_type=int),
                "output_height": ParameterValue(right_h, value_type=int),
                "width":         ParameterValue(right_w, value_type=int),
                "height":        ParameterValue(right_h, value_type=int),
            }],
        )

        # Bi3D node
        bi3d_node = ComposableNode(
            package="isaac_ros_bi3d",
            plugin="nvidia::isaac_ros::bi3d::Bi3DNode",
            name="bi3d_node",
            remappings=[
                ("left_image_bi3d",        "/left/resize/image"),
                ("right_image_bi3d",       "/right/resize/image"),
                ("left_camera_info_bi3d",  "/left/resize/camera_info"),
                ("right_camera_info_bi3d", "/right/resize/camera_info"),
            ],
            parameters=[{
                "featnet_engine_file_path": featnet,
                "segnet_engine_file_path":  segnet,
                # YAML string avoids the integer/double array type-mismatch
                "disparity_values_yaml":    disparity_yaml,
            }],
        )

        nodes = [left_resize, right_resize, bi3d_node]

        # Optional freespace consumer (kept off unless you set use_freespace:=true)
        if str(use_freespace.perform(context)).lower() in ("1", "true", "yes"):
            freespace = ComposableNode(
                package="isaac_ros_bi3d",
                plugin="nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode",
                name="freespace_segmentation_node",
                remappings=[
                    ("disparity_image", "/bi3d_node/bi3d_output"),
                ],
                # If you add intrinsics, use doubles (e.g., 500.0 not 500) to avoid type errors.
                # parameters=[{"f_x": 500.0, "f_y": 500.0, "c_x": 320.0, "c_y": 240.0}],
            )
            nodes.append(freespace)

        return [
            ComposableNodeContainer(
                name="bi3d_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                emulate_tty=True,
                output="screen",
                composable_node_descriptions=nodes,
            )
        ]

    return LaunchDescription(declare_args + [OpaqueFunction(function=launch_setup)])

