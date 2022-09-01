# python imports
import tf
import rospy

# ros-python imports
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Imu
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber

# import custom messages
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth

# bruce imports
from bruce_slam.utils.conversions import *
from bruce_slam.dead_reckoning import DeadReckoningNode

# DRACo imports
from draco_slam.utils.topics import *


class DeadReckoningNodeMultiRobot(DeadReckoningNode):

    def __init__(self) -> None:
        super().__init__()

    def init_node(self, ns="~")->None:
        """Init the node, fetch all paramaters from ROS

        Args:
            ns (str, optional): The namespace of the node. Defaults to "~".
        """

        # Parameters for Node
        self.imu_pose = rospy.get_param(ns + "imu_pose")
        self.imu_pose = n2g(self.imu_pose, "Pose3")
        self.imu_rot = self.imu_pose.rotation()
        self.dvl_max_velocity = rospy.get_param(ns + "dvl_max_velocity")
        self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
        self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
        self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

        # Subscribers and caches
        self.dvl_sub = Subscriber(DVL_TOPIC, DVL)
        self.gyro_sub = Subscriber(GYRO_INTEGRATION_TOPIC, Odometry)
        self.depth_sub = Subscriber(DEPTH_TOPIC, Depth)
        self.depth_cache = Cache(self.depth_sub, 1)

        if rospy.get_param(ns + "imu_version") == 1:
            self.imu_sub = Subscriber(IMU_TOPIC, Imu)
        elif rospy.get_param(ns + "imu_version") == 2:
            self.imu_sub = Subscriber(IMU_TOPIC_MK_II, Imu)

        # Use point cloud for visualization
        self.traj_pub = rospy.Publisher(
            "traj_dead_reck", PointCloud2, queue_size=10)

        self.odom_pub = rospy.Publisher(
            LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=10)

        # are we using the FOG gyroscope?
        self.use_gyro = rospy.get_param(ns + "use_gyro")

        # define the callback, are we using the gyro or the VN100?
        if self.use_gyro:
            self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub, self.gyro_sub], 300, .1)
            self.ts.registerCallback(self.callback_with_gyro)
        else:
            self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub], 200, .1)
            self.ts.registerCallback(self.callback)

        self.tf = tf.TransformBroadcaster()

        print(self.pose)
        rospy.loginfo("Localization node is initialized")

    def publish_pose(self,flag)->None:
        """Publish the pose
        """
        if self.pose is None:
            return

        header = rospy.Header()
        header.stamp = self.prev_time
        header.frame_id = self.rov_id + "_odom"

        odom_msg = Odometry()
        odom_msg.header = header
        # pose in odom frame
        odom_msg.pose.pose = g2r(self.pose)
        # twist in local frame
        odom_msg.child_frame_id = self.rov_id + "_base_link"
        # Local planer behaves worse
        # odom_msg.twist.twist.linear.x = self.prev_vel[0]
        # odom_msg.twist.twist.linear.y = self.prev_vel[1]
        # odom_msg.twist.twist.linear.z = self.prev_vel[2]
        # odom_msg.twist.twist.angular.x = self.prev_omega[0]
        # odom_msg.twist.twist.angular.y = self.prev_omega[1]
        # odom_msg.twist.twist.angular.z = self.prev_omega[2]
        odom_msg.twist.twist.linear.x = 0
        odom_msg.twist.twist.linear.y = 0
        odom_msg.twist.twist.linear.z = 0
        odom_msg.twist.twist.angular.x = 0
        odom_msg.twist.twist.angular.y = 0
        odom_msg.twist.twist.angular.z = 0
        self.odom_pub.publish(odom_msg)

        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        self.tf.sendTransform(
            (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), header.stamp, self.rov_id + "_base_link", self.rov_id + "_odom"
        )

