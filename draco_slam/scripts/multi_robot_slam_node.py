#!/usr/bin/env python

import rospy
from bruce_slam.utils.io import *
from bruce_slam.slam_ros import SLAMNode
from draco_slam.multi_robot_slam import MultiRobotSLAM


if __name__ == "__main__":

    #init the node
    rospy.init_node("slam", log_level=rospy.INFO)

    #call the class constructor
    node = MultiRobotSLAM()
    node.init_node()
    node.init_multi_robot()

    #parse and start
    args, _ = common_parser().parse_known_args()

    if not args.file:
        loginfo("Start online slam...")
        rospy.spin()
    else:
        loginfo("Start offline slam...")
        offline(args)