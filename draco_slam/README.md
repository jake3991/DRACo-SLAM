# Overview
This readme covers the details of our multi-robot SLAM system. Consider this as a high-level user manual. A flow chart overview is shown below. 

<img src="images/flow_chart.png " width="500"/>

This system searches for inter-robot loop closures using only sensor data with NO initial conditions between robots. We first share scene descriptors, then if they are similar, the raw point clouds from the sonar. By first sharing descriptors, we manage the bandwidth usage. The system depicted in the flow chart above is instantiated on each vehicle, making this a fully distributed system. The entire goal is to find inter-robot loop closures and insert them into the existing pose graph. The pose graph is shown below. 

<img src="images/factor_graph.png " width="500"/>

This shows an example pose graph for a single robot with one other team member, note we can have N team members. Yellow circles indicate the robot's own poses. Grey circles show the team members' poses. Green lines are odometry factors between our own poses (from dead reckoning DVL+IMU). Red lines show intra-robot loop closures, the loops we find with ourselves. Blue lines show inter-robot loop closures, and purple lines show partner robot factors and the team members' trajectory. 

# Parameter overview
Here we provide documentation of each parameter in the multi-robot SLAM system. Once again, tune at your own risk. These parameters are in the config folder in multi_robot.yaml.

PCM 
- multi_robot_pcm_queue_size: the queue size for the PCM queue 
- multi_robot_min_pcm: the minimum set size for PCM

Trajectory updates
- resend_translation: resend if a pose has moved by this much
- resend_rotation: resend if a pose has rotated by this much

Point cloud compression
- point_compression_resolution: voxel size used in grid compression

System options for registration outlier rejection
- study:
  - overlap: Should we use point cloud overlap?
  - count: Should we use point cloud point count?
  - ratio: Should we compare the ratio of point clouds?
  - context: Should we use scene context images?
  - pcm: Should we use PCM?
  - point_compression: Should we use point cloud compression when passing clouds?

Registration params
- mrr: (multi-robot-registration)
  - min_points: the min points needed in a cloud to try registration
  - points_ratio: the max ratio between the number of points in a pair of clouds to try registration
  - min_overlap: the minimum overlap between two registered clouds after ICP
  - sampling_points: optimizer parameter for ICP
  - iterations: optimizer parameter for ICP
  - tolerance: optimizer parameter for ICP
  - max_translation: the max translation between a pair of clouds, optimizer parameter for ICP
  - max_rotation: the max rotation between a pair of clouds, optimizer parameter for ICP
  - k_neighbors: the number of neighbors to consider when doing a descriptor comparison
  - max_scan_context: the max context image difference to try registration
  - max_tree_cost: the max difference between descriptors

- scan context params
  - bearing_bins: the number of steps in the bearing axis
  - max_bearing: the size of the bearing axis in degrees
  - range_bins: the number of steps in the range axis
  - max_range: the size of the range axis in meters 

Search parameters for after we have some loop closures
- plcs: (partner loop closure search)
  - max_translation_search: the max difference between a pair of poses
  - max_rotation_search: the max rotation between a pair of poses
