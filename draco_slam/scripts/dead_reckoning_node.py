#!/usr/bin/env python

# python imports
import rospy

# pull in the dead reckoning code
from draco_slam.dead_reckoning import DeadReckoningNodeMultiRobot


if __name__ == "__main__":
    rospy.init_node("localization", log_level=rospy.INFO)

    node = DeadReckoningNodeMultiRobot()
    node.init_node()
    rospy.spin()

