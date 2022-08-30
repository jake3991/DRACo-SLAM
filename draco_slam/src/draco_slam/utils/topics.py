"""
Topics for the draco_slam project
"""

#sensor topic
IMU_TOPIC = "/vn100/imu/raw"
DVL_TOPIC = "/rti/body_velocity/raw"
DEPTH_TOPIC = "/bar30/depth/raw"
SONAR_TOPIC = "/sonar_oculus_node/ping"

#inter robot comms topics
RING_KEY_TOPIC = "/common/ring_keys"
DATA_REQUEST_TOPIC = "/common/data_requests"
KEYFRAME_TOPIC = "/common/keyframes"
LOOP_CLOSURE_TOPIC = "/common/loop_closures"
STATE_UPDATE_TOPIC = "/common/state_updates"

#SLAM topics
SLAM_NS = ""
LOCALIZATION_ODOM_TOPIC = SLAM_NS + "localization/odom"
LOCALIZATION_TRAJ_TOPIC = SLAM_NS + "localization/traj"
SLAM_POSE_TOPIC = SLAM_NS + "slam/pose"
SLAM_ODOM_TOPIC = SLAM_NS + "slam/odom"
SLAM_TRAJ_TOPIC = SLAM_NS + "slam/traj"
SLAM_CLOUD_TOPIC = SLAM_NS + "slam/cloud"
SLAM_CONSTRAINT_TOPIC = SLAM_NS + "slam/constraint"
PARTNER_TRAJECTORY_TOPIC = SLAM_NS + "partner_traj"
SLAM_ISAM2_TOPIC = SLAM_NS + "slam/isam2"
SLAM_PREDICT_SLAM_UPDATE_SERVICE = SLAM_NS + "slam/predict_slam_update"
MAPPING_INTENSITY_TOPIC = SLAM_NS + "mapping/intensity"
MAPPING_OCCUPANCY_TOPIC = SLAM_NS + "mapping/occupancy"
MAPPING_GET_MAP_SERVICE = SLAM_NS + "mapping/get_map"
SONAR_FEATURE_TOPIC = SLAM_NS + "feature_extraction/feature"
SONAR_FEATURE_IMG_TOPIC = SLAM_NS + "feature_extraction/feature_img"
