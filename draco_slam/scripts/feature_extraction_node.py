#!/usr/bin/env python
import rospy

from draco_slam.feature_extraction import FeatureExtractionMultiRobot

if __name__ == "__main__":

    #init the ros node
    rospy.init_node("feature_extraction_node", log_level=rospy.INFO)

    #call class constructor
    node = FeatureExtractionMultiRobot()
    node.init_node()

    rospy.spin()

