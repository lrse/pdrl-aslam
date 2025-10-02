from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    ns      = LaunchConfiguration('ns')
    base    = LaunchConfiguration('base_frame')
    ncam    = LaunchConfiguration('num_cameras')
    nmin    = LaunchConfiguration('min_num_images')
    imu_on  = LaunchConfiguration('enable_imu_fusion')
    rect    = LaunchConfiguration('rectified_images')
    sync_ms = LaunchConfiguration('sync_matching_threshold_ms')
    
    use_sim = LaunchConfiguration('use_sim_time')
    map_fr  = LaunchConfiguration('map_frame')
    odom_fr = LaunchConfiguration('odom_frame')
    pub_tf_odom_base = LaunchConfiguration('publish_odom_to_base_tf')
    pub_tf_map_odom  = LaunchConfiguration('publish_map_to_odom_tf')

    # Static TFs publish.
    left_cam_static = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="left_cam_static_tf",
        arguments=[
            "0.20", "0.06", "0.20",
            "-1.57079632679", "0", "-1.57079632679",
            base, [ns, TextSubstitution(text="/left_camera_optical_frame")],
        ],
    )
    right_cam_static = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="right_cam_static_tf",
        arguments=[
            "0.20", "-0.06", "0.20",
            "-1.57079632679", "0", "-1.57079632679",
            base, [ns, TextSubstitution(text="/right_camera_optical_frame")],
        ],
    )

    # cuVSLAM.
    vslam = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam_node',
        namespace=ns,
        parameters=[{
            'use_sim_time':               ParameterValue(use_sim, value_type=bool),
            'base_frame':                 base,
            'map_frame':                  map_fr,
            'odom_frame':                 odom_fr,
            'publish_odom_to_base_tf':    ParameterValue(pub_tf_odom_base, value_type=bool),
            'publish_map_to_odom_tf':     ParameterValue(pub_tf_map_odom,  value_type=bool),

            'num_cameras':                ParameterValue(ncam,    value_type=int),
            'min_num_images':             ParameterValue(nmin,    value_type=int),
            'enable_imu_fusion':          ParameterValue(imu_on,  value_type=bool),
            'rectified_images':           ParameterValue(rect,    value_type=bool),
            'sync_matching_threshold_ms': ParameterValue(sync_ms, value_type=float),
            'image_qos':'SENSOR_DATA', 
            'image_buffer_size': 80, 

	    'enable_slam_visualization':  ParameterValue(
		                                LaunchConfiguration('enable_slam_visualization'),
		                                value_type=bool),
            'enable_landmarks_view':      ParameterValue(
		                                LaunchConfiguration('enable_landmarks_view'),
		                                value_type=bool),
	    'enable_observations_view':   ParameterValue(
		                                LaunchConfiguration('enable_observations_view'),
		                                value_type=bool),
		    
            
            
        }],
    )

    container = ComposableNodeContainer(
        name='vslam_container',
        namespace=ns,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[vslam],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('ns',                 default_value='unitree_go2'),
        DeclareLaunchArgument('base_frame',         default_value='unitree_go2/base_footprint'),
        DeclareLaunchArgument('num_cameras',        default_value='2'),
        DeclareLaunchArgument('min_num_images',     default_value='2'),
        DeclareLaunchArgument('enable_imu_fusion',  default_value='false'),
        DeclareLaunchArgument('rectified_images',   default_value='true'),
        DeclareLaunchArgument('sync_matching_threshold_ms', default_value='10.0'),

        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map_frame',    default_value=[ns, TextSubstitution(text='/map')]),
        DeclareLaunchArgument('odom_frame',   default_value=[ns, TextSubstitution(text='/odom')]),
        DeclareLaunchArgument('publish_odom_to_base_tf', default_value='true'),
        DeclareLaunchArgument('publish_map_to_odom_tf',  default_value='true'),
        
        DeclareLaunchArgument('enable_slam_visualization',  default_value='false'),
        DeclareLaunchArgument('enable_observations_view',  default_value='false'),
        DeclareLaunchArgument('enable_landmarks_view',  default_value='false'),


        # Start statics first, then cuVSLAM.
        left_cam_static,
        right_cam_static,
        container,
    ])

