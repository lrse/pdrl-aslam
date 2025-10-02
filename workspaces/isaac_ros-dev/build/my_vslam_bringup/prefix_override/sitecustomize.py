import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wakkow/workspaces/isaac_ros-dev/install/my_vslam_bringup'
