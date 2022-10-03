# python imports
import rospy
import cv_bridge
import numpy as np

# sensor message imports
from sensor_msgs.msg import PointCloud2, Image
from sonar_oculus.msg import OculusPing, OculusPingUncompressed

# bruce and draco imports
from bruce_slam.feature_extraction import FeatureExtraction
from bruce_slam.utils.conversions import *
from draco_slam.utils.topics import *


class FeatureExtractionMultiRobot(FeatureExtraction):
    def __init__(self) -> None:
        super().__init__()

    def init_node(self, ns="~"):

        # get the ROV ID
        self.rov_id = rospy.get_param(ns + "rov_number")

        # read in CFAR parameters
        self.Ntc = rospy.get_param(ns + "CFAR/Ntc")
        self.Ngc = rospy.get_param(ns + "CFAR/Ngc")
        self.Pfa = rospy.get_param(ns + "CFAR/Pfa")
        self.rank = rospy.get_param(ns + "CFAR/rank")
        self.alg = rospy.get_param(ns + "CFAR/alg", "SOCA")
        self.threshold = rospy.get_param(ns + "filter/threshold")

        # read in PCL downsampling parameters
        self.resolution = rospy.get_param(ns + "filter/resolution")
        self.outlier_filter_radius = rospy.get_param(ns + "filter/radius")
        self.outlier_filter_min_points = rospy.get_param(ns + "filter/min_points")

        # parameter to decide how often to skip a frame
        self.skip = rospy.get_param(ns + "filter/skip")

        # are the incoming images compressed?
        self.compressed_images = rospy.get_param(ns + "compressed_images")

        # cv bridge
        self.BridgeInstance = cv_bridge.CvBridge()

        # read in the format
        self.coordinates = rospy.get_param(
            ns + "visualization/coordinates", "cartesian"
        )

        # vis parameters
        self.radius = rospy.get_param(ns + "visualization/radius")
        self.color = rospy.get_param(ns + "visualization/color")

        # sonar subsciber
        if self.compressed_images:
            self.sonar_sub = rospy.Subscriber(
                SONAR_TOPIC, OculusPing, self.callback, queue_size=10
            )
        else:
            self.sonar_sub = rospy.Subscriber(
                SONAR_TOPIC_UNCOMPRESSED,
                OculusPingUncompressed,
                self.callback,
                queue_size=10,
            )

        # feature publish topic
        self.feature_pub = rospy.Publisher(
            SONAR_FEATURE_TOPIC, PointCloud2, queue_size=10
        )

        # vis publish topic
        self.feature_img_pub = rospy.Publisher(
            SONAR_FEATURE_IMG_TOPIC, Image, queue_size=10
        )

        self.configure()

    def publish_features(self, ping, points):
        """Publish the feature message using the provided parameters in an OculusPing message
        ping: OculusPing message
        points: points to be converted to a ros point cloud, in cartisian meters
        """

        # shift the axis
        points = np.c_[points[:, 0], np.zeros(len(points)), points[:, 1]]

        # convert to a pointcloud
        feature_msg = n2r(points, "PointCloudXYZ")

        # give the feature message the same time stamp as the source sonar image
        # this is CRITICAL to good time sync downstream
        feature_msg.header.stamp = ping.header.stamp
        feature_msg.header.frame_id = self.rov_id + "_base_link"

        # publish the point cloud, to be used by SLAM
        self.feature_pub.publish(feature_msg)
