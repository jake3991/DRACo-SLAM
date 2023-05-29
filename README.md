# DRACo-SLAM 
## Distributed Robust Acoustic Communication-efficient SLAM for Imaging Sonar Equipped Underwater Robot Teams

Welcome! This repo contains the code for DRACo-SLAM, a distributed multi-robot SLAM system for underwater robots using imaging sonar for perception. This system is based on our single agent slam system, extending it to estimate each robot's pose but also the poses of team members. Real data is used with simulated communications. For more information on the single agent SLAM system, see this [repo](https://github.com/jake3991/sonar-SLAM). Note that the single agent SLAM system uses DVL, IMU, and imaging sonar. 

Detailed documentation on the DRACo-SLAM system can be found in the DRACo-SLAM folder and paper. 

# Sensor overview

Our vehicle is documented in this repo https://github.com/jake3991/Argonaut.git. The highlights are the following pieces of hardware. 
- Occulus M750d imaging sonar
- Rowe SeaPilot DVL
- Vectornav 100 MEMS IMU
- Bar30 pressure sensor
- KVH-DSP-1760 fiber optic gyroscope (optional)

# Python Dependencies, note python-3

```
cv_bridge
gtsam
matplotlib
message_filters
numpy
opencv_python
rosbag
rospy
scikit_learn
scipy
sensor_msgs
Shapely
tf
tqdm
pyyaml
```

# ROS Dependencies
```
ROS-noetic
catkin-pybind11
catkin-tools
```

# Installation
- Ensure all python dependencies are installed
- Check ros distro; we use noetic
- clone this repo into your catkin workspace
- clone git clone https://github.com/ethz-asl/libnabo.git into your catkin workspace
- clone https://github.com/ethz-asl/libpointmatcher.git into your catkin workspace
- clone https://github.com/jake3991/Argonaut.git into your catkin workspace
- clone https://github.com/jake3991/sonar-SLAM into your catkin workspace
- build your workspace with catkin build NOT catkin_make

# Sample data
Link to sample data: https://drive.google.com/file/d/1FyQxVwVZbaGT_OypRuzX4nm1dPCfsJh5/view?usp=sharing

# Running "Online"
This will launch the SLAM system, and then we will playback the data as if it is happening now. Note the launch file slam.launch will launch a team of three vehicles. 
- source catkin_ws/devel/setup.bash
- roslaunch draco_slam slam.launch
- rosbag play your_data.bag

# Configuration
This multi-robot SLAM system has many parameters. Please review the readme in the draco_slam folder for an explanation of each parameter. However, we highly recommend using the default parameters in the config folder. Tune at your own risk. 

# Presentation
For a high-level overview of this systems functionality, see the presentation video here LINK NEEDED on my YouTube channel. 

# Citation
Paper arxiv link: https://arxiv.org/abs/2210.00867

```
@inproceedings{
  title={DRACo-SLAM: Distributed Robust Acoustic Communication-efficient SLAM for Imaging Sonar Equipped Underwater Robot Teams},
  author={John McConnell, Yewei Huang, Paul Szenher, Ivana Collado-Gonzalez and Brendan Englot},
  booktitle={International Conference on Intelligent Robots and Systems (IROS),
  year={2022},
  organization={IEEE/RSJ}
}
```






