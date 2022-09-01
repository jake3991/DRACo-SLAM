#!/usr/bin/env python

import rospy
from draco_slam.multi_robot_slam import MultiRobotSLAM


if __name__ == "__main__":

    #init the node
    rospy.init_node("slam", log_level=rospy.INFO)

    #call the class constructor
    node = MultiRobotSLAM()
    node.init_node()
    
    rospy.loginfo("Start online slam...")
    rospy.spin()