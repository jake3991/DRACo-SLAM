# python imports
import threading
from typing import Tuple
import tf
import rospy
import gtsam
import cv_bridge
import ros_numpy
import numpy as np
from nav_msgs.msg import Odometry
from message_filters import  Subscriber
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from scipy.spatial.ckdtree import cKDTree as KDTree
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer


# Argonaut imports
from sonar_oculus.msg import OculusPing

# bruce imports
from bruce_slam.slam_ros import SLAMNode
from bruce_slam.utils.conversions import *
from bruce_slam.slam_objects import Keyframe
from bruce_slam.utils.visualization import *
from bruce_slam import pcl

# draco imports
from draco_slam.utils.topics import *
from draco_slam.utils.conversions import Y,Z
from draco_slam.msg import RingKey,DataRequest,KeyframeImage,LoopClosure,PoseHistory,Dummy
from draco_slam.multi_robot_registration import MultiRobotRegistration
from draco_slam.multi_robot_objects import ICPResultInterRobot

class MultiRobotSLAM(SLAMNode):

    def __init__(self) -> None:
        super(MultiRobotSLAM, self).__init__()

    def init_node(self, ns="~") -> None:

        #keyframe paramters, how often to add them
        self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
        self.keyframe_duration = rospy.Duration(self.keyframe_duration)
        self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
        self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

        #SLAM paramter, are we using SLAM or just dead reckoning
        # TODO remove this param
        self.enable_slam = rospy.get_param(ns + "enable_slam")

        #noise models
        self.prior_sigmas = rospy.get_param(ns + "prior_sigmas")
        self.odom_sigmas = rospy.get_param(ns + "odom_sigmas")
        self.icp_odom_sigmas = rospy.get_param(ns + "icp_odom_sigmas")
        self.inter_robot_sigmas = rospy.get_param(ns + "inter_robot_sigmas")
        self.partner_robot_sigmas = rospy.get_param(ns + "partner_robot_sigmas")

        #resultion for map downsampling
        self.point_resolution = rospy.get_param(ns + "point_resolution")

        #sequential scan matching parameters (SSM)
        self.ssm_params.enable = rospy.get_param(ns + "ssm/enable")
        self.ssm_params.min_points = rospy.get_param(ns + "ssm/min_points")
        self.ssm_params.max_translation = rospy.get_param(ns + "ssm/max_translation")
        self.ssm_params.max_rotation = rospy.get_param(ns + "ssm/max_rotation")
        self.ssm_params.target_frames = rospy.get_param(ns + "ssm/target_frames")

        #non sequential scan matching parameters (NSSM) aka loop closures
        self.nssm_params.enable = rospy.get_param(ns + "nssm/enable")
        self.nssm_params.min_st_sep = rospy.get_param(ns + "nssm/min_st_sep")
        self.nssm_params.min_points = rospy.get_param(ns + "nssm/min_points")
        self.nssm_params.max_translation = rospy.get_param(ns + "nssm/max_translation")
        self.nssm_params.max_rotation = rospy.get_param(ns + "nssm/max_rotation")
        self.nssm_params.source_frames = rospy.get_param(ns + "nssm/source_frames")
        self.nssm_params.cov_samples = rospy.get_param(ns + "nssm/cov_samples")

        #pairwise consistency maximization parameters for loop closure 
        #outliar rejection
        self.pcm_queue_size = rospy.get_param(ns + "pcm_queue_size")
        self.min_pcm = rospy.get_param(ns + "min_pcm")

        #mak delay between an incoming point cloud and dead reckoning
        self.feature_odom_sync_max_delay = 0.5

        #define the subsrcibing topics
        self.feature_sub = Subscriber(SONAR_FEATURE_TOPIC, PointCloud2)
        self.odom_sub = Subscriber(LOCALIZATION_ODOM_TOPIC, Odometry)

        #define the sync policy
        self.time_sync = ApproximateTimeSynchronizer(
            [self.feature_sub, self.odom_sub], 1000, 
            self.feature_odom_sync_max_delay, allow_headerless = False)

        #register the callback in the sync policy
        self.time_sync.registerCallback(self.SLAM_callback)

        # odom sub for republising the pose at a higher rate
        # self.odom_sub_repub = rospy.Subscriber(LOCALIZATION_ODOM_TOPIC_MKII, Odometry, callback=self.odom_callback,queue_size=200)

        #pose publisher
        self.pose_pub = rospy.Publisher(
            SLAM_POSE_TOPIC, PoseWithCovarianceStamped, queue_size=10)

        #dead reckoning topic
        self.odom_pub = rospy.Publisher(SLAM_ODOM_TOPIC, Odometry, queue_size=10)

        #SLAM trajectory topic
        self.traj_pub = rospy.Publisher(
            SLAM_TRAJ_TOPIC, PointCloud2, queue_size=1, latch=True)

        #constraints between poses
        self.constraint_pub = rospy.Publisher(
            SLAM_CONSTRAINT_TOPIC, Marker, queue_size=1, latch=True)

        #point cloud publisher topic
        self.cloud_pub = rospy.Publisher(
            SLAM_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)

        #tf broadcaster to show pose
        self.tf = tf.TransformBroadcaster()

        #cv bridge object
        self.CVbridge = cv_bridge.CvBridge()

        #get the ICP configuration from the yaml fukle
        self.icp_config = rospy.get_param(ns + "icp_config")
        self.icp.loadFromYaml(self.icp_config)

        #call the configure function
        self.configure()
        
        # call the multi-robot init function
        self.init_multi_robot(ns)
        self.init_message_pool()


    def init_multi_robot(self,ns) -> None:

        # get the rov ID and the number of robots in the system
        self.rov_id = rospy.get_param(ns + "rov_number")
        self.number_of_robots = rospy.get_param(ns + "number_of_robots")

        # set the numerical robot ID
        if self.rov_id == "rov_one":
            self.vin = 1
        elif self.rov_id == "rov_two":
            self.vin = 2
        elif self.rov_id == "rov_three":
            self.vin = 3

        #pull the ablation study params
        self.case_type = rospy.get_param(ns + "study/case")
        self.use_count = rospy.get_param(ns + "study/count")
        self.use_ratio = rospy.get_param(ns + "study/ratio")
        self.use_overlap = rospy.get_param(ns + "study/overlap")
        self.use_context = rospy.get_param(ns + "study/context")
        self.use_pcm = rospy.get_param(ns + "study/pcm")
        self.point_compression = rospy.get_param(ns + "study/point_compression")
        self.use_multi_robot = rospy.get_param(ns + "study/use_multi_robot")
        self.add_multi_robot = rospy.get_param(ns + "study/add_multi_robot")
        self.play_bag = rospy.get_param(ns + "study/play_bag")
        self.share_loops = rospy.get_param(ns + "study/share_loops")

        #point cloud compression
        self.point_compression_resolution = rospy.get_param(ns + "point_compression_resolution")
        
        #multi-robot PCM
        self.multi_robot_pcm_queue_size = rospy.get_param(ns + "multi_robot_pcm_queue_size")
        self.multi_robot_min_pcm = rospy.get_param(ns + "multi_robot_min_pcm")

        #define the noise models
        self.inter_robot_model = self.create_noise_model(self.inter_robot_sigmas)
        self.partner_robot_model = self.create_noise_model(self.partner_robot_sigmas)

        #multi-robot-registration
        self.mrr_max_tree_cost = rospy.get_param(ns + "mrr/max_tree_cost")
        self.mrr_min_points = rospy.get_param(ns + "mrr/min_points")
        self.mrr_points_ratio = rospy.get_param(ns + "mrr/points_ratio")
        self.mrr_min_overlap = rospy.get_param(ns + "mrr/min_overlap")
        self.mrr_sampling_points = rospy.get_param(ns + "mrr/sampling_points")
        self.mrr_iterations = rospy.get_param(ns + "mrr/iterations")
        self.mrr_tolerance = rospy.get_param(ns + "mrr/tolerance")
        self.mrr_max_translation = rospy.get_param(ns + "mrr/max_translation")
        self.mrr_max_rotation = rospy.get_param(ns + "mrr/max_rotation")
        self.mrr_max_scan_context = rospy.get_param(ns + "mrr/max_scan_context")
        self.mrr_k_neighbors = rospy.get_param(ns + "mrr/k_neighbors")
        self.mrr_max_translation_search = rospy.get_param(ns + "plcs/max_translation_search")
        self.mrr_max_rotation_search = rospy.get_param(ns + "plcs/max_rotation_search")
        self.mrr_resend_translation = rospy.get_param(ns + "resend_translation")
        self.mrr_resend_rotation = rospy.get_param(ns + "resend_rotation")

        #scan context
        self.sc_number_of_scans = rospy.get_param(ns + "sc/number_of_scans")
        self.sc_bearing_bins = rospy.get_param(ns + "sc/bearing_bins")
        self.sc_max_bearing = rospy.get_param(ns + "sc/max_bearing")
        self.sc_range_bins = rospy.get_param(ns + "sc/range_bins")
        self.sc_max_range = rospy.get_param(ns + "sc/max_range")

        #active alcs closure search
        self.alcs_enable = rospy.get_param(ns + "alcs/enable")
        self.alcs_passive_count = rospy.get_param(ns + "alcs/passive_count")
        self.alcs_uncertainty = rospy.get_param(ns + "alcs/uncertainty")
        self.alcs_point_count = rospy.get_param(ns + "alcs/point_count")
        self.alcs_uncertainty_max = rospy.get_param(ns + "alcs/uncertainty_max")
        self.alcs_bits_max = rospy.get_param(ns + "alcs/bits_max")
        self.alcs_roi = float(rospy.get_param(ns + "alcs/roi"))
        self.alcs_covariance_prediction_samples = rospy.get_param(ns + "alcs/covariance_prediction_samples")


        #define all the publishers and subscribers for the multi-robot system

        # ring key exchange
        self.ring_key_pub = rospy.Publisher(RING_KEY_TOPIC,RingKey,queue_size=5)
        self.ring_key_sub = rospy.Subscriber(RING_KEY_TOPIC,RingKey,callback=self.ring_key_callback, queue_size=50)

        # point cloud request exchange
        self.request_pub = rospy.Publisher(DATA_REQUEST_TOPIC,DataRequest,queue_size=5)
        self.request_sub = rospy.Subscriber(DATA_REQUEST_TOPIC,DataRequest,callback=self.request_callback,queue_size=50) 

        # actual point cloud data exchange
        self.key_frame_pub = rospy.Publisher(KEYFRAME_TOPIC, KeyframeImage, queue_size=5)
        self.key_frame_sub = rospy.Subscriber(KEYFRAME_TOPIC, KeyframeImage, callback=self.keyframe_callback, queue_size=50)
        
        # loop closure exchange
        self.loop_closure_pub = rospy.Publisher(LOOP_CLOSURE_TOPIC,LoopClosure,queue_size=5)
        self.loop_closure_sub = rospy.Subscriber(LOOP_CLOSURE_TOPIC,LoopClosure,callback=self.loop_closure_callback,queue_size=20)

        # state update exchange
        self.state_pub = rospy.Publisher(STATE_UPDATE_TOPIC,PoseHistory,queue_size=5)
        self.state_sub = rospy.Subscriber(STATE_UPDATE_TOPIC,PoseHistory,callback=self.state_update_callback,queue_size=5)

        # dummy topic to make global registration run
        self.dummy_pub = rospy.Publisher("/" + self.rov_id + "/slam/global_reg_run", Dummy, queue_size=5)
        self.dummy_sub = rospy.Subscriber("/" + self.rov_id + "/slam/global_reg_run", Dummy, callback=self.global_registration_callback,
                                                                                queue_size=50)
        
        # shutdown sub
        self.shutdown_sub = rospy.Subscriber("/" + self.rov_id + "/shutdown", Dummy, callback=self.shutdown_callback,
                                                                                queue_size=50)

        #multi-robot registration results
        self.registration_pub = rospy.Publisher(
            "registration_results", PointCloud2, queue_size=5, latch=True)

        self.merged_pub = rospy.Publisher(
            "merged", PointCloud2, queue_size=5)

        #constraints between poses
        self.inter_robot_constraint_pub = rospy.Publisher(
            "inter_robot_constraint", Marker, queue_size=1, latch=True)

        self.partner_traj_pub = rospy.Publisher(
            PARTNER_TRAJECTORY_TOPIC, Marker, queue_size=1, latch=True)

        # define the central registration job queue
        self.frames_to_check = []

    def init_message_pool(self) -> None:
        """Build the multi robot registration message pool. This includes the objects required to manage
        the multi-robot system.
        """

        #get a list of all the other robots
        robots = []
        for i in range(self.number_of_robots):
            if i+1 != self.vin: # exclude myself
                robots.append(i+1)

        #build the message pool
        self.send_state = False
        self.keyframes_multi_robot = []
        self.robots = robots # keep a list of the robots
        self.message_pool = {} # 
        self.outside_frames = {}
        self.home = {}
        self.home_arr = {}
        self.multi_robot_queue = {}
        self.multi_robot_symbols = {}
        self.merged_robots = {}
        self.partner_trajectory = {}
        self.partner_covariance = {}
        self.partner_covariance_log = {}
        self.tested_jobs = {}
        self.outside_frames_added = {}
        self.loops_added = {}
        keys = ["y","z"]
        for robot,k in zip(robots,keys):
            self.partner_trajectory[robot] = []
            self.partner_covariance[robot] = []
            self.partner_covariance_log[robot] = []
            self.merged_robots[robot] = False
            self.multi_robot_symbols[robot] = k
            self.home[robot] = None
            self.home_arr[robot] = []
            self.multi_robot_queue[robot] = []
            self.outside_frames[robot] = {}
            self.message_pool[robot] = MultiRobotRegistration(vin = self.vin,
                                                            number_of_scans = self.sc_number_of_scans,
                                                            k_neighbors = self.mrr_k_neighbors,
                                                            bearing_bins = self.sc_bearing_bins,
                                                            max_bearing = self.sc_max_bearing,
                                                            range_bins = self.sc_range_bins,
                                                            max_range = self.sc_max_range,
                                                            max_translation = self.mrr_max_translation,
                                                            max_rotation = self.mrr_max_rotation,
                                                            min_overlap = self.mrr_min_overlap,
                                                            min_points = self.mrr_min_points,
                                                            points_ratio = self.mrr_points_ratio,
                                                            sampling_points = self.mrr_sampling_points,
                                                            iterations = self.mrr_iterations,
                                                            tolerance = self.mrr_tolerance,
                                                            max_scan_context = self.mrr_max_scan_context,
                                                            use_count = self.use_count,
                                                            use_ratio = self.use_ratio,
                                                            use_overlap = self.use_overlap,
                                                            use_context = self.use_context,
                                                            icp_path = self.icp_config)

    def SLAM_callback(self, feature_msg:PointCloud2, odom_msg:Odometry)->None:
        """SLAM call back. Subscibes to the feature msg point cloud and odom msg
            Handles the whole SLAM system and publishes map, poses and constraints

        Args:
            feature_msg (PointCloud2): the incoming sonar point cloud
            odom_msg (Odometry): the incoming DVL/IMU state estimate
        """

        #aquire the lock 
        self.lock.acquire()

        #get rostime from the point cloud
        time = feature_msg.header.stamp

        #get the dead reckoning pose from the odom msg, GTSAM pose object
        dr_pose3 = r2g(odom_msg.pose.pose)

        #init a new key frame
        frame = Keyframe(False, time, dr_pose3)

        #convert the point cloud message to a numpy array of 2D
        points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(feature_msg)
        points = np.c_[points[:,0] , -1 *  points[:,2]]

        # In case feature extraction is skipped in this frame
        if len(points) and np.isnan(points[0, 0]):
            frame.status = False
        else:
            frame.status = self.is_keyframe(frame)

        #set the frames twist
        frame.twist = odom_msg.twist.twist

        #update the keyframe with pose information from dead reckoning
        if self.keyframes:
            dr_odom = self.current_keyframe.dr_pose.between(frame.dr_pose)
            pose = self.current_keyframe.pose.compose(dr_odom)
            frame.update(pose)

        #check frame staus, are we actually adding a keyframe? This is determined based on distance 
        #traveled according to dead reckoning
        if frame.status:

            #add the point cloud to the frame
            frame.points = points

            #perform seqential scan matching
            #if this is the first frame do not
            if not self.keyframes:
                self.add_prior(frame)
            else:
                self.add_sequential_scan_matching(frame)

            #update the factor graph with the new frame
            self.update_factor_graph(frame)

            #if loop closures are enabled
            #nonsequential scan matching is True (a loop closure occured) update graph again
            if self.nssm_params.enable and self.add_nonsequential_scan_matching():
                self.update_factor_graph()

            self.build_ring_key() # build the ring key for the most recent cloud
            self.update_submaps() # check if we need to update any submaps
            self.publish_newest_ring_key() # send the most recent ring key out to the team
            self.query_tree() # do a kd tree search here using our most recent ring key
            
        #update current time step and publish the topics
        self.current_frame = frame
        self.publish_all()
        self.lock.release()
        
    def ring_key_callback(self, msg : RingKey) -> None:
        """This is a the first step in the multi-robot SLAM system. 
        Here we will perform the following tasks in the order below. 
        1. Recive the ring key message and log it 
        (ring key scene descriptor, pose in the sending robots frame and keyframe index)
        2. Compare this newest ring key to all of MY ring keys
        3. If there are any good matches based on the comparison in step 2, log those
        to the job queue (self.frames_to_check) for furthur inspection

        Args:
            msg (RingKey): A ring key message, ring key scene descriptor, 
                pose in the sending robots frame and keyframe index
        """

        # we don't need to look at out own ring keys
        if msg.vin == self.vin:
            return

        # instaciate a keyframe
        frame = Keyframe(True,
                        None, 
                        pose223(gtsam.Pose2(msg.pose[0],msg.pose[1],msg.pose[3])),
                        source_pose=gtsam.Pose2(msg.pose[0],msg.pose[1],msg.pose[3]),
                        index=msg.keyframe_id)
        frame.ring_key = np.array(list(msg.data)) #send through the ring key

        # aquire the lock log the keyframe in       
        self.lock.acquire()
        self.message_pool[msg.vin].add_keyframe_ring_key(frame) # log the keyframe into our message pool
        keyframes_ = list(self.keyframes) # grab our set of keyframes
        home = dict(self.home) # get any robot frames we have solved for
        self.lock.release() # release the lock as we no longer need it
        
        # query the tree in feature space
        tree = self.vectorize(keyframes_) #pull the Kdtree for our own keyframes
        distances, indexes = tree.query(frame.ring_key,k=self.mrr_k_neighbors,distance_upper_bound=self.mrr_max_tree_cost)
        indexes = indexes[distances < float('inf')] #drop any inf queries
        distances = distances[distances < float('inf')]

        # gate this on eclidian distance if we are merged
        if home[msg.vin] is not None and len(distances) != 0:
            frame_pose = home[msg.vin].compose(frame.source_pose) # get this pose in our reference frame
            indexes, distances = self.filter_by_distance(frame_pose,indexes,distances,keyframes_)

        # check the distances array, see if we need to file a request for this point cloud
        if len(distances) != 0: # are there any valid neighbors
            self.lock.acquire() # we need to touch shared memory
            vals_to_request = []
            for i in range(msg.keyframe_id - 1,msg.keyframe_id + 2): #loop over the ids in the submap
                if i not in self.outside_frames[msg.vin]: #check if we have this point cloud aleady
                    vals_to_request.append(i) # add it to the list of point clouds we need
            
            # assemble and publish a data request message
            if len(vals_to_request) != 0: # if the list of frames we need is not zero
                request_msg = DataRequest()
                request_msg.data = vals_to_request #the list of clouds we want
                request_msg.requester_vin = self.vin # my OWN robot ID
                request_msg.target_vin = msg.vin # the robot we need the data from
                self.request_pub.publish(request_msg)

            # log which frames need to be furthur inspected, possible loop closures
            if len(indexes) != 0:
                # registration jobs are logged as (my frame indexes, their frame indexes, their vin)
                self.frames_to_check.append((indexes,[msg.keyframe_id],[msg.vin]))
                self.dummy_pub.publish(Dummy()) # this kicks the registration callback 
 
            self.lock.release()

    def query_tree(self) -> None:
        """Similar to the ring key callback above except here we compare the most recent
        SLAM keyframe from MY system to all the recived ring keys so far. 
        1. pull the most recent ring key
        2. compare it to the recived ring keys
        3. log anything we need to inspect furthur in self.frames_to_check
        """

        if len(self.keyframes) < 3:
            return

        # combine all the ring keys from all the robots, track their offests
        keys = []
        vin_list = []
        real_indexes = []
        for vin in self.robots: # loop over the other robots in the system
            keys_i = self.message_pool[vin].get_ring_keys() #get the ring keys from this robot
            keys += keys_i #add these keys to the list
            for i in range(len(keys_i)):
                vin_list.append(vin) # create an array of vin numbers to go along with each key
                real_indexes.append(i) # index tracker

        if len(keys) > 0: # if we have any ring keys at all

            # build a query a KD tree, we are searching for matches with OUR most recent SLAM frame
            tree = KDTree(np.array(keys))
            distances, indexes = tree.query(self.keyframes[-2].ring_key, k=self.mrr_k_neighbors, distance_upper_bound=self.mrr_max_tree_cost)
            indexes = indexes[distances < float('inf')] #filter out inf values
            distances = distances[distances < float('inf')]

            # get the vin for each nearest neighbor
            # here we also clean up the indexes getting the index in the keyframe list
            vin_for_jobs = []
            job_indexes = []
            for id in indexes:
                vin_for_jobs.append(vin_list[id]) # get vin
                job_indexes.append(real_indexes[id]) # get the index in the vin.keyframes

            # euclidan space filtering
            if len(job_indexes) != 0:
                vin_for_jobs, job_indexes, distances = self.filter_by_distance_2(self.keyframes[-2].pose,
                                                                                np.array(vin_for_jobs),
                                                                                np.array(job_indexes),
                                                                                np.array(distances))

            # log the job to the job quene to be run by the global reg callback
            if len(job_indexes) != 0:
                self.frames_to_check.append(([len(self.keyframes)-2], job_indexes, vin_for_jobs))
                self.dummy_pub.publish(Dummy()) # send a message to trip the callback
                
    def request_callback(self, msg : DataRequest) -> None:
        """This callback handles the requests for point clouds.
        If we recive a request we need to fulfill it by simply sending
        the requesting robot the pointcloud. 

        Args:
            msg (DataRequest): the data request message
        """

        #we only need to listen to requests that are targeted at us 
        if msg.target_vin != self.vin:
            return
        
        '''self.lock.acquire()
        keyframes_ = list(self.keyframes) # copy the keyframes and dump the lock
        self.lock.release()'''

        requester_vin = msg.requester_vin

        # loop over the list of requested keyframes
        for i in list(msg.data):

            # fetch the frame
            frame = self.keyframes[i]
            pose = g2n(frame.pose3) #get the 3D pose as a numpy array
            
            #prep the message
            msg = KeyframeImage()
            msg.cloud = n2r(np.c_[frame.points, np.zeros_like(frame.points[:,0])], "PointCloudXYZ") # push the point cloud
            msg.pose = [pose[0], pose[1], pose[2], pose[5]] # push the pose
            msg.vin = self.vin # push our own vehicle ID 
            msg.requester_vin = requester_vin
            msg.keyframe_id = i
            self.key_frame_pub.publish(msg)

    def keyframe_callback(self, msg : KeyframeImage) -> None:
        """manage the incoming keyframes, these keyframes have been requested after Kdtree search

        Args:
            msg (KeyframeImage): the incoming keyframe
        """
        
        #if we did not request this, we do not need it 
        if msg.requester_vin != self.vin:
            return

        #if we sent this, we do not need it
        if msg.vin == self.vin:
            return

        #parse out the message
        points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg.cloud)[:,:2] #decode the cloud
        
        #if we are using point cloud compression
        #@if self.point_compression and points is not None and len(points) > 0:
        #    pass
            #points_zipped,xmin,ymin = self.zip_points(points,self.point_compression_resolution)
            #points = self.unzip_points(points_zipped,self.point_compression_resolution,xmin,ymin)

        pose_their_frame = msg.pose #get the pose
        pose_their_frame = gtsam.Pose2(pose_their_frame[0],pose_their_frame[1],pose_their_frame[3]) #make it a gtsam object
        keyframe_id = msg.keyframe_id
        vin = msg.vin
        
        #perform the updates
        self.lock.acquire()
        self.message_pool[vin].update_keyframe(pose_their_frame,points,keyframe_id) #update the regisgtration system with this cloud
        self.outside_frames[vin][keyframe_id] = keyframe_id #update the hash table to indicate we have recived this message, do not request it again
        self.lock.release()

        # send a message to trip the global registration callback
        self.dummy_pub.publish(Dummy())

    def global_registration_callback(self, msg : Dummy) -> None:
        """A callback to manage the registration between robot point clouds

        Args:
            msg (Dummy): a dummy message to make this callback run
        """

        # we need to touch shared memory, pick up the lock
        self.lock.acquire()

        #check if we have any jobs to run
        if len(self.frames_to_check) == 0:
            self.lock.release()
            return

        #parse the job queue, get the oldest job
        my_keys, their_keys, vin_list = self.frames_to_check[0]

        #check if we have the required data for this job
        if self.check_job(their_keys,vin_list) == False:
            self.dummy_pub.publish(Dummy())
            self.lock.release()
            return

        #copy the keyframe lists so we can release the lock
        my_keyframes = list(self.keyframes)
        message_pool_copy = dict(self.message_pool)
        self.lock.release() #we release here so this thread can run without holding up the SLAM thread

        #case 1: [unkown number inside keyframes], [a single outside keyframe], [a single vin]
        #case 2: [a single inside keyframe], [unkown number of outside keyframes], [unkown number of vins,len is same as before]
        job_status, loop  = self.run_global_registration(my_keys,
                                                            their_keys,
                                                            vin_list,
                                                            my_keyframes,
                                                            message_pool_copy
                                                            )

        #now we need to touch shared memory, pick up a lock
        self.lock.acquire()
        self.frames_to_check.pop(0) #update the job tracker, this job is complete

        #if we got nothing from the above search, go ahead and return
        if job_status == False:
            if len(self.frames_to_check) != 0: #if there are more jobs, send a message to run this callback again
                self.dummy_pub.publish(Dummy())
            self.lock.release()
            return

        #log the loop to the appropriate data structure
        self.multi_robot_queue[loop.vin].append(loop) 

        #maintain the multi-robot PCM queue
        if self.use_pcm:
            while (self.multi_robot_queue[loop.vin] and loop.target_key - self.multi_robot_queue[loop.vin][0].target_key > self.multi_robot_pcm_queue_size):
                self.multi_robot_queue[loop.vin].pop(0)
        queue = list(self.multi_robot_queue[loop.vin]) #copy the queue before we release the lock
        self.lock.release() #release the lock so we can do PCM without holding up other threads

        #call PCM
        if self.use_pcm:
            pcm = self.verify_pcm(queue, self.multi_robot_min_pcm)
        else:
            pcm = [len(queue)-1]

        if len(pcm) > 0:
            self.lock.acquire()
            self.publish_multi_robot_registration(self.vis_pcm_result(pcm,loop.vin)) #vis pcm results
            self.merge_queue(pcm,loop.vin) #merge the recent loop closures
            self.update_factor_graph() #update the factor graph
            self.publish_all(False)
            self.lock.release()

        self.dummy_pub.publish(Dummy())
        
    def loop_closure_callback(self, msg : LoopClosure) -> None:
        """Handle a inter-robot loop closure found by another robot and sent to us. 

        Args:
            msg (LoopClosure): the loop closure message
        """

        # if this loop closure does not concern me, I do not care about it
        if msg.loop_vin != self.vin:
            return

        self.lock.acquire()

        # check if we aleady have this loop closure
        if (msg.vin, msg.target_key) not in self.outside_frames_added: 
            self.outside_frames_added[(msg.vin,msg.target_key)] = True

            # add the factor to the SLAM solution
            icp_transform = n2g(np.array(list(msg.data)),"Pose2").inverse() #make sure to get the inverse
            factor = gtsam.BetweenFactorPose2(
                            X(msg.source_key),
                            self.to_symbol(msg.target_key,msg.vin),
                            icp_transform,
                            self.inter_robot_model)
            self.graph.add(factor)

            # if not we need to get an initial guess and build a keyframe
            target_pose = self.keyframes[msg.source_key].pose.compose(icp_transform) #get the pose in this robots frame
            self.keyframes_multi_robot.append(Keyframe(
                        True,
                        None,
                        pose223(target_pose),
                        ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg.cloud)[:,:2], 
                        source_pose=self.message_pool[msg.vin].keyframes[msg.target_key].source_pose,
                        index=msg.target_key,
                        vin=msg.vin,
                        index_kf=msg.source_key))
            
            # add an initial guess
            self.values.insert(self.to_symbol(msg.target_key,msg.vin), target_pose)

            # mark that this robot has been merged
            self.merged_robots[msg.vin] = True
            
        self.update_factor_graph() #update the factor graph
        self.publish_all(False)
        self.lock.release()
        
    def state_update_callback(self, msg : PoseHistory) -> None:
        """Handle an incoming state update message

        Args:
            msg (PoseHistory): the incoming message, the whole time history of poses from 
            a robot
        """

        #we do not care about our own state updates
        if msg.vin == self.vin:
            return

        self.lock.acquire() #shared memory, pick up the lock
        data = np.array(list(msg.data))
        data = np.reshape(data, (len(data)//3,3)) #reshape the message into a Nx3 trajectory [x,y,theta]
        iter = np.min([len(self.message_pool[msg.vin].keyframes), len(data)]) #loop over whatever is shorter, the state update or keyframes
        for i in range(iter): #pass the new state vector of to the registration system
            self.message_pool[msg.vin].keyframes[i].source_pose = gtsam.Pose2(data[i][0],data[i][1],data[i][2])
        self.update_factor_graph() #update the factor graph
        self.publish_all(False)
        self.lock.release()

    def shutdown_callback(self, msg : Dummy) -> None:
        """Handle a shutdown message.

        Args:
            msg (Dummy): the message used to trip this callback
        """

        # shutdown is easy, just get the lock and never let it go
        self.lock.acquire()
        print(self.rov_id,"shutdown")

    def get_scan_context(self,points : np.array) -> np.array:
        """Perform scan context for an aggragated point cloud

        Args:
            points (np.array): the input point cloud, a submap

        Returns:
            np.array: the scan context image and ring key
        """

        #instanciate the image
        polar_image = np.zeros((self.sc_bearing_bins,self.sc_range_bins))

        #convert to discrete polar coords
        r_cont = np.sqrt(np.square(points[:,0]) + np.square(points[:,1])) #first contiuous polar coords
        b_cont = abs(np.degrees(np.arctan2(points[:,0] , points[:,1]))) #* np.sign(points[:,0])
        r_dis = np.array((r_cont / self.sc_max_range) * self.sc_range_bins).astype(np.uint16) #discret coords
        b_dis = np.array((((b_cont / self.sc_max_bearing) + 1) / 2) * self.sc_bearing_bins).astype(np.uint16)

        #clip the vales
        r_dis = np.clip(r_dis, 0, self.sc_range_bins-1)
        b_dis = np.clip(b_dis, 0, self.sc_bearing_bins-1)

        #populate the image
        for i, j in zip(b_dis, r_dis):
            polar_image[i][j] = 1

        #build the ring key
        ring_key = np.sum(polar_image, axis = 0)

        return polar_image, ring_key

    def update_submaps(self) -> None:
        """Update the keyframes submaps and scan context ring_keys
        """

        #loop over the keyframes
        if len(self.keyframes) >=3: #only do so if we have enough keyframes to build submaps
            for i in range(1,len(self.keyframes)-2): 
                if self.keyframes[i].redo_submap: #check if the keyframe needs to be rebuilt
                    self.keyframes[i].submap = self.get_points([i-1,i,i+1], i)  #push the submap 
                    _, self.keyframes[i].ring_key = self.get_scan_context(self.keyframes[i].submap) #push the ring_key

    def build_ring_key(self) -> None:
        """Build the most recent ring key
        """

        # check if we have more than 3 ring frames, the submap size
        if len(self.keyframes) >= 3:

            # first get the submap, aggragate the points of three keyframes
            # we want the previous, the current and the next
            self.keyframes[-2].submap = self.get_points(
                            [len(self.keyframes)-3,len(self.keyframes)-2,len(self.keyframes)-1], len(self.keyframes)-2)

            #get the ring key descriptor for this newest frame
            self.keyframes[-2].context, self.keyframes[-2].ring_key = self.get_scan_context(
                            self.keyframes[len(self.keyframes)-2].submap) 

        # we are not far enough along, push in a fake polar image and ring key
        else:
            polar_image = np.ones((self.sc_bearing_bins,self.sc_range_bins)) * float('inf')
            self.keyframes[-1].ring_key = np.sum(polar_image, axis = 0)

    def publish_newest_ring_key(self) -> None:
        """Send the most recent ring key to the team
        """

        #check if we have enough keyframes to have built a ring key
        if len(self.keyframes) >= 3:
            pose = g2n(self.keyframes[-2].pose3) # package the pose of this ring key
            msg = RingKey()
            msg.data = list(np.array(self.keyframes[-2].ring_key).astype(np.uint8)) # the ring key
            assert(msg.data is not None)
            msg.keyframe_id = len(self.keyframes) - 2 # the keyframe number
            msg.vin = self.vin # the vehicle id number
            msg.pose = [pose[0], pose[1], pose[2], pose[5]] #[x,y,theta,depth]
            msg.count = len(self.keyframes[-2].submap) # the size of the submap
            self.ring_key_pub.publish(msg) 

    def vectorize(self,keyframes : list) -> KDTree:
        """Convert a keyframe list to a KDtree of the ring keys in each keyframe

        Args:
            keyframes_ (list): a list of slam Keyframes

        Returns:
            KDTree: tree of ring keys for search
        """

        keys = [] 
        for frame in keyframes:
            if frame.ring_key is not None:
                keys.append(frame.ring_key)
        return KDTree(keys)

    def check_job(self,their_keys:list,vin_list:list)->bool:
        """Checks if the job can be completed, check if the data required for the job is here

        Args:
            their_keys (list): list of a single element of an outside keyframe
            vin_list (list): list of a single element of the outside vin

        Returns:
            bool: if the job can run, if the point clouds are present
        """

        #check if we have the required data to run the job
        for their_key,search_vin in zip(their_keys,vin_list):
            if self.message_pool[search_vin].keyframes[their_key].context is None: #do we have context yet?
                for i in range(their_key-1,their_key+2): #loop over the frames needed for context
                    if (i >= len(self.message_pool[search_vin].keyframes) or 
                                    self.message_pool[search_vin].keyframes[i].points is None):
                                    return False # we do not have the required data for this job
                self.message_pool[search_vin].update_scan_context(their_key) #update the scan context image


    def run_global_registration(self,my_keys:list,their_keys:list,vin_list:list,
                                                my_keyframes:list,message_pool_copy:dict)->Tuple[bool,np.array]:
        """Perform a global registration run

        Args:
            my_keys (list): the keyframe indexes from my own SLAM solution
            their_keys (list): the keyframe indexes from the outside robots
            vin_list (list): the vin for each of the above outside robot keyframes
            my_keyframes (list): a copy of my keyframes
            message_pool_copy (dict): a copy of the message pool

        Returns:
            Tuple[bool,np.array]: [status flag, results if there are any, run time on a step wise basis]
        """

        #containers for loop closures
        cost = [] 
        loops = []

        #loop over the job, this loop is NOT N^2 as one of the lists will always have exactly 1 item
        for my_key in my_keys: #loop over each of my keys
            for their_key,search_vin in zip(their_keys,vin_list): #loop over the outside key and the vin associated with that key

                # Check if we have run this job before, no repeats
                if (search_vin,their_key,my_key) not in self.tested_jobs:
                    self.tested_jobs[(search_vin,their_key,my_key)] = True # log that we have run this job
                
                    result, status, message = message_pool_copy[search_vin].compare_frames(my_keyframes[my_key],
                                                            message_pool_copy[search_vin].keyframes[their_key]) 

                    # if the above registration method returned a good result package it up as a loop closure
                    if status:
                        # overlap, fit_score, pose_between_frames, pose_global, None
                        overlap, fit_score, _, pose_global, covariance = result #parse out the results
                        cost.append(overlap) #log the cost, in this case overlap
                        loop = ICPResultInterRobot(my_key,  #push the results into a ICP results object
                                                    their_key,
                                                    my_keyframes[my_key].pose,
                                                    pose_global,
                                                    fit_score,
                                                    overlap,
                                                    search_vin,
                                                    covariance,
                                                    )
                        loops.append(loop)

        if len(loops) != 0:
            return True, loops[np.argmax(cost)] #only return the best one
        else:
            return False, None #we got nothing, indicate that

    def publish_multi_robot_registration(self, points_for_publish:np.array)->None:
        """Publish points from good registration runs inside the global registration callback

        Args:
            points_for_publish (np.array): the points we want to publish
        """

        #publish the registration results
        if points_for_publish is not None:
            points_for_publish = np.column_stack((points_for_publish[:,0], 
                                                    points_for_publish[:,1], 
                                                    np.zeros(len(points_for_publish)),
                                                    points_for_publish[:,2]))
            cloud_msg = n2r(points_for_publish, "PointCloudXYZI")
            cloud_msg.header.stamp = self.current_keyframe.time
            cloud_msg.header.frame_id = self.rov_id + "_map"
            self.registration_pub.publish(cloud_msg)
        else:
            points_for_publish = np.array([[np.nan, np.nan, np.nan, np.nan]])
            cloud_msg = n2r(points_for_publish, "PointCloudXYZI")
            cloud_msg.header.stamp = self.current_keyframe.time
            cloud_msg.header.frame_id = self.rov_id + "_map"
            self.registration_pub.publish(cloud_msg)

    def vis_pcm_result(self,pcm:list,vin:int)->np.array:
        """Gather the point clouds for a PCM result

        Args:
            pcm (list): pcm results, the indexes of the queue that are approved
            vin (int): the vin for the robot we are vis 

        Returns:
            np.array: point cloud
        """

        #vis if PCM found some good loop closures
        out = np.zeros((1,3))
        for i in pcm:
            if self.multi_robot_queue[vin][i].inserted == False: # only vis a result once
                pts = self.message_pool[vin].keyframes[self.multi_robot_queue[vin][i].target_key].points
                ps = self.multi_robot_queue[vin][i].target_pose
                pts = pts.dot(ps.matrix()[:2, :2].T) + np.array([ps.x(),ps.y()])
                pts = np.column_stack(((pts, np.zeros(len(pts)))))
                out = np.row_stack((out,pts))
        return out

    def merge_queue(self,pcm:list,vin:int)->None:
        """Merge the multi-robot measurnments into the SLAM graph, this is after a PCM has been approved

        Args:
            pcm (list): indexes in the PCM queue that have been approved
            vin (int): the vehicle ID number for this PCM run
        """

        self.merged_robots[vin] = True # log that we have merged this robot into our system

        #add the factors that connect to this SLAM graph
        for i in pcm:

            source_key = self.multi_robot_queue[vin][i].source_key
            target_key = self.multi_robot_queue[vin][i].target_key
            
            source = self.keyframes[source_key]
            target = self.message_pool[vin].keyframes[target_key]

            source_pose = source.pose
            target_pose = self.multi_robot_queue[vin][i].target_pose

            pose_between = source_pose.between(target_pose)
            self.message_pool[vin].keyframes[target_key].pose = target_pose

            #each loop closure only inserted once
            if self.multi_robot_queue[vin][i].inserted == False:
                self.multi_robot_queue[vin][i].inserted = True

                #send the loop out to the team
                self.publish_loop_closure(source_key,target_key,pose_between,vin)

                #build the factor between the poses
                factor = gtsam.BetweenFactorPose2(
                        X(source_key),
                        self.to_symbol(target_key,vin),
                        pose_between,
                        self.inter_robot_model)
                self.graph.add(factor)

                # check if we have this loop closure yet
                if (vin,target_key) not in self.outside_frames_added:
                    self.outside_frames_added[(vin,target_key)] = True 
                    self.keyframes_multi_robot.append(Keyframe(
                                            True,
                                            None,
                                            pose223(target_pose),
                                            self.message_pool[vin].keyframes[target_key].points, 
                                            source_pose=self.message_pool[vin].keyframes[target_key].source_pose,
                                            index=target_key,
                                            vin = vin,
                                            index_kf=source_key))
                    self.values.insert(self.to_symbol(target_key,vin), target_pose) # insert the intial guess

    def publish_loop_closure(self,source_key:int,target_key:int,pose_between:gtsam.Pose2,vin:int)->None:
        """Publish a inter-robot loop closure that we found

        Args:
            source_key (int): the key for the source frame
            target_key (int): the key for the target frame
            pose_between (gtsam.Pose2): gtsam pose between the two frames from GO-ICP
            vin (int): the vin for the vehicle that this loop closure is WITH
        """

        msg = LoopClosure()
        msg.source_key = target_key #flip these on purpose for other robots pose graph
        msg.target_key = source_key
        msg.data = list(g2n(pose_between))
        msg.vin = self.vin #my own vin 
        msg.loop_vin = vin #the vin of the robot this loop closure is with
        msg.cloud = n2r(np.c_[self.keyframes[source_key].points, np.zeros_like(self.keyframes[source_key].points[:,0])], "PointCloudXYZ")
        self.loop_closure_pub.publish(msg)

    def to_symbol(self,key:int,vin:int)->gtsam.symbol:
        """Given a key (index of keyframes i.e. keyframes[key]), encode it in the proper symbol type
        Using the vin number. 

        Args:
            key (int): the index of the keyframe
            vin (int): vehicle ID number

        Returns:
            gtsam.symbol: the encoded key as a gtsam symbol
        """

        if self.multi_robot_symbols[vin] == "y":
            return Y(key)
        elif self.multi_robot_symbols[vin] == "z":
            return Z(key)

    def merge_trajectory(self,isam: gtsam.ISAM2, vin: int) -> gtsam.ISAM2:
        """Add the robot trajectory to a factor graph to esimate the whole merged system

        Args:
            isam (gtsam.ISAM2): the instance of SLAM we want to add this trajectory to
            vin (int): the vin for the vehicle we want to merge

        Returns:
            gtsam.ISAM2: return the populated ISAM2 instance
        """

        # make some empty structures
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        # if there are not enough keyframes, don't do anything
        if len(self.message_pool[vin].keyframes) <= 3:
            return isam

        # loop over all the keyframes
        run = False
        for i in range(len(self.message_pool[vin].keyframes)-1):

            # TODO do we need this?
            run = True

            # make sure we have a source pose for this frame
            if self.message_pool[vin].keyframes[i].source_pose is None:
                continue
            
            # if there is no intial guess, build one
            if self.message_pool[vin].keyframes[i].guess_pose is None:
                self.message_pool[vin].keyframes[i].guess_pose = self.home[vin].compose(
                                                        self.message_pool[vin].keyframes[i].source_pose)

            # get the pose between sequential frames
            pose_between = self.message_pool[vin].keyframes[i].source_pose.between(
                                            self.message_pool[vin].keyframes[i+1].source_pose)

            factor = gtsam.BetweenFactorPose2( # build it as a factor
                        self.to_symbol(i,vin),
                        self.to_symbol(i+1,vin),
                        pose_between,
                        self.partner_robot_model)
            graph.add(factor) # add the factor

            # if this frame is not a multi-robot keyframe via ICP, then we need an intitial guess
            if (vin,i) not in self.outside_frames_added:
                values.insert(self.to_symbol(i,vin),self.message_pool[vin].keyframes[i].guess_pose) # add the initial guess
    
        # handle the last frame, we just need an initial guess here
        if run == True and (vin,i+1) not in self.outside_frames_added: # check if this frame is added via ICP
            self.message_pool[vin].keyframes[i+1].guess_pose = self.home[vin].compose(
                                            self.message_pool[vin].keyframes[i+1].source_pose)
            values.insert(self.to_symbol(i+1,vin),self.message_pool[vin].keyframes[i+1].guess_pose) # add the initial guess for the last frame

        # update the isam object that was passed into this function
        isam.update(graph, values)

        return isam

    def update_factor_graph(self, keyframe: Keyframe=None):
        """Update the factor graph, this is called after we have inserted new factors.

        Args:
            keyframe (Keyframe, optional): A keyframe that needs to be added. Defaults to None.
        """

        # log if we need to
        if keyframe:
            self.keyframes.append(keyframe)

        # update the ISAM2 instance and clear out the graph/values objects
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)
        self.values.clear()

        # pull the slam result
        values = self.isam.calculateEstimate()
        isam = gtsam.ISAM2(self.isam) # make a copy of the ISAM2 instance

        # check to see if we have any partner robot frames in our factor graph
        if len(self.keyframes_multi_robot) != 0:
            # if we have any merged frames, we need to see which robots have been merged
            # and add their trajectories to our factor graph
            for robot in self.robots: # loop over the robots in the system
                if self.home[robot] is not None: # check if this robot has been merged
                    isam = self.merge_trajectory(isam,robot) # merge the robots trajectory into my graph
            values = isam.calculateEstimate() # get the pose graph estimate using the combined graph

            # update our multi-robot keyframes
            for y in range(len(self.keyframes_multi_robot)):
                if self.home[self.keyframes_multi_robot[y].vin] is not None: # check for merge
                    # grab and update the pose
                    pose = values.atPose2(self.to_symbol(self.keyframes_multi_robot[y].index, self.keyframes_multi_robot[y].vin))
                    self.keyframes_multi_robot[y].update(pose)

            # update the trajectory estimate for each robot in the system
            for robot in self.robots:
                if self.home[robot] is not None: # check if we have a reference frame for the other robot
                    partner_trajectory = []
                    for y in range(0, len(self.message_pool[robot].keyframes)): # loop over all the keyframes for this robot
                        if values.exists(self.to_symbol(y,robot)):
                            pose = values.atPose2(self.to_symbol(y,robot)) # get the pose
                            self.message_pool[robot].keyframes[y].guess_pose = pose # set this pose as the new intitial guess
                            partner_trajectory.append(g2n(pose)) # log it to the vis trajectory
                    self.partner_trajectory[robot] = np.array(partner_trajectory) # push it to the table that stores all the traj
                
                # we do NOT have a reference frame, we need to get one
                # only try to get a reference frame if we have the data to do so
                elif self.merged_robots[robot] == True:   
                    home_arr = dict(self.home_arr) # make a copy of the empty structure
                    for y in range(len(self.keyframes_multi_robot)): # loop over all the robot keyframes
                        # check if this keyframe is from the robot we care about
                        if self.keyframes_multi_robot[y].vin == robot: 
                            pose = self.keyframes_multi_robot[y].pose
                            between = self.keyframes_multi_robot[y].source_pose.between(gtsam.Pose2(0,0,0)) #estimate of robots own frame
                            home_arr[robot].append(g2n(pose.compose(between)))
                    if len(home_arr[robot]) != 0:
                        self.home[robot] = gtsam.Pose2(np.mean(np.array(home_arr[robot]),axis=0))

        # Update my own SLAM trajectory
        errors = []
        for x in range(len(self.keyframes)):
            pose = values.atPose2(X(x))
            errors.append(g2n(self.keyframes[x].pose.between(pose))) # log the change in our pose estimates
            self.keyframes[x].update(pose)
        errors = np.array(errors)

        # Only update latest cov
        cov = self.isam.marginalCovariance(X(len(self.keyframes)- 1))
        self.keyframes[-1].update(pose, cov)

        #check the error if we need to send out our state vector
        if len(errors) > 0:
            euclidan_change = abs(np.max(np.sqrt(errors[:,0]**2 + errors[:,1]**2)))
            rotation_change = np.degrees(abs(np.max(errors[:,2])))
            #set the flag
            if self.mrr_resend_translation != -1:
                if euclidan_change >= self.mrr_resend_translation or rotation_change >= self.mrr_resend_rotation: 
                    self.send_state = True

        #update the pose estimates in the PCM queue, this gives the best estimate of what is pairwise consistent
        for ret in self.nssm_queue:
            ret.source_pose = self.keyframes[ret.source_key].pose
            ret.target_pose = self.keyframes[ret.target_key].pose
            if ret.inserted:
                ret.estimated_transform = ret.target_pose.between(ret.source_pose)

    def publish_partner_trajectory(self) -> None:
        """Publish out estimate of the partners trajectory in our own frame
        """

        # define the vis colors
        colors = {1:"red", 2:"light_blue", 3:"yellow"}

        # container for the trajectory
        traj = []

        # loop over all the robots, get their trajectories
        for robot in self.robots:
            for x, kf in enumerate(self.partner_trajectory[robot][1:], 1):
                p1 = self.partner_trajectory[robot][x - 1][0], self.partner_trajectory[robot][x - 1][1], 0.
                p2 = self.partner_trajectory[robot][x][0], self.partner_trajectory[robot][x][1], 0.
                traj.append((p1, p2, colors[robot]))

        # if nothing, do nothing
        if len(traj) > 0:
            # convert this list to a series of multi-colored lines and publish
            link_msg = ros_constraints(traj)
            link_msg.header.stamp = self.current_keyframe.time
            link_msg.header.frame_id = self.rov_id + "_map"
            self.partner_traj_pub.publish(link_msg)

    def publish_slam_update(self) -> None:
        """Send an update to our SLAM solution if it has encoutered a large change
        """

        if self.send_state: #if we need to send the state
            data = [] 
            for frame in self.keyframes:
                data.append(g2n(frame.pose))
            data = np.array(data)
            msg = PoseHistory() #build a message 
            msg.vin = self.vin
            msg.data = list(np.ravel(data))
            self.state_pub.publish(msg)
            self.send_state = False # reset the flag

    def publish_point_cloud_merged(self) -> None:
        """Publish the point clouds we have merged into our own graph.
        """

        #define the vis colors
        colors = {1:[255,0,0], 2:[112,158,206], 3:[255,255,0]}

        #define an empty array
        all_points = [np.zeros((0, 2), np.float32)]

        #list of keyframe ids
        all_keys = []

        if len(self.keyframes_multi_robot) > 0:
            for key in range(len(self.keyframes_multi_robot)):

                #parse the pose
                pose = self.keyframes_multi_robot[key].pose

                #get the resgistered point cloud
                transf_points = self.keyframes_multi_robot[key].transf_points
                if transf_points is not None:
                    all_points.append(transf_points)
                    all_keys.append(np.ones((len(transf_points),3)) * colors[self.keyframes_multi_robot[key].vin])

            if len(all_keys) == 0:
                return
                
            all_points = np.concatenate(all_points)
            all_keys = np.concatenate(all_keys)

            #use PCL to downsample this point cloud
            sampled_points, sampled_keys = pcl.downsample(
                all_points, all_keys, self.point_resolution
            )

            #parse the downsampled cloud into the ros xyzi format
            sampled_xyzi = np.c_[sampled_points, sampled_keys]
            
            #if there are no points return and do nothing
            if len(sampled_xyzi) == 0:
                return

            #convert the point cloud to a ros message and publish
            cloud_msg = n2r(sampled_xyzi, "PointCloudXYZRGB")
            cloud_msg.header.stamp = self.current_keyframe.time
            cloud_msg.header.frame_id = self.rov_id + "_map"
            self.merged_pub.publish(cloud_msg)
    
    def publish_multi_robot_constraints(self) -> None:
        """Publish the multi-robot factors as blue lines.
        """

        links = []

        # get the interrobot loops
        for frame in self.keyframes_multi_robot:
            p1 = frame.pose3.x(), frame.pose3.y(), frame.dr_pose3.z()
            p2 = self.keyframes[frame.index_kf].pose3.x(), self.keyframes[frame.index_kf].pose3.y(), self.keyframes[frame.index_kf].dr_pose3.z()
            links.append((p1, p2, "blue"))

        # convert this list to a series of multi-colored lines and publish
        if links:
            link_msg = ros_constraints(links)
            link_msg.header.stamp = self.current_keyframe.time
            link_msg.header.frame_id = self.rov_id + "_map"
            self.inter_robot_constraint_pub.publish(link_msg)

    def publish_all(self,main: bool=True) -> None:
        """Publish to all ouput topics

        Args:
            main (bool, optional): If we are calling this from the SLAM_callback. Defaults to False.
        """

        # make sure we have some keyframes
        if not self.keyframes:
            return

        # don't publish the pose from outside the SLAM callback for timing reasons
        if main:
            self.publish_pose()

        # publish all the output topics if this is a new SLAM keyframe in the SLAM callback
        # or we are running from outside the slam callback
        if self.current_frame.status or main != False:
            self.publish_trajectory()
            self.publish_constraint()
            self.publish_point_cloud()
            self.publish_partner_trajectory()
            self.publish_point_cloud_merged()
            self.publish_slam_update()
            self.publish_multi_robot_constraints()

    def filter_by_distance(self,pose_my_frame: gtsam.Pose2,indexes: np.array,distances: np.array,keyframes: list) -> np.array:
        """Filter the results of a Kd-tree search by the distance between query and neighbors

        Args:
            pose_my_frame (gtsam.Pose2): the pose of query frame in my own reference frame
            indexes (np.array): kdtree search results, the indexes of the array
            distances (np.array): kdtree distance for the above
            keyframes (list): the keyframes from our own SLAM solution

        Returns:
            np.array: filtered indexes and distances according to the max rotation and translation 
        """

        distances_euclidian = []
        for idx in indexes: #loop over the Kd-tree nearest neighbors
            #get the pose btween the nearest neighbor and the query frame
            dist = g2n(keyframes[idx].pose.between(pose_my_frame))
            distances_euclidian.append(dist) 
        distances_euclidian = np.array(distances_euclidian) #cast to array
        rotations = distances_euclidian[:,2] #parse out rotation
        distances_euclidian = np.linalg.norm(distances_euclidian[:,:2],axis=1) #Parse out ecuclidan distance
        #filter based on max distance and rotation
        indexes = indexes[(distances_euclidian<self.mrr_max_translation_search)&(rotations<self.mrr_max_rotation_search)]
        distances = distances[(distances_euclidian<self.mrr_max_translation_search)&(rotations<self.mrr_max_rotation_search)]
        return np.array(indexes), np.array(distances)

    def filter_by_distance_2(self,my_pose: gtsam.Pose2,vins: np.array,indexes: np.array,distances: np.array)-> np.array:
        """Filter the indexes and distances by their ecldian distance

        Args:
            my_pose (gtsam.Pose2): my current pose estimate
            vins (np.array): the vins for each index
            indexes (np.array): the index in the keyframes list 
            distances (np.array): the Kd-tree distance with each index

        Returns:
            vins (np.array): the filtered vins
            indexes (np.array): the filtered indexes
            distances (np.array): the filtered distances
        """
        
        distances_euclidian = []
        for idx,vin in zip(indexes,vins): #loop over the Kd-tree nearest neighbors
            
            #check if we have an initial frame and a source pose for this index
            if self.home[vin] is not None and self.message_pool[vin].keyframes[idx].source_pose is not None:
                pose_2 = self.home[vin].compose(self.message_pool[vin].keyframes[idx].source_pose) #their pose in my frame
                distances_euclidian.append(g2n(my_pose.between(pose_2))) #pose between these

            #we have not merged this robot yet, but we do have a source pose for it, do not filter
            elif self.home[vin] is None and self.message_pool[vin].keyframes[idx].source_pose is not None:
                distances_euclidian.append(g2n(gtsam.Pose2(0,0,0))) #push through zeros so this does not get removed

            #not enough information
            else:
                distances_euclidian.append(g2n(gtsam.Pose2(500000,500000,500000))) #push a big number so it does get removed

        distances_euclidian = np.array(distances_euclidian) #cast to array
        rotations = distances_euclidian[:,2] #parse out rotation
        distances_euclidian = np.linalg.norm(distances_euclidian[:,:2],axis=1) #Parse out ecuclidan distance
        #filter based on max distance and rotation
        indexes = indexes[(distances_euclidian<self.mrr_max_translation_search)&(rotations<self.mrr_max_rotation_search)]
        distances = distances[(distances_euclidian<self.mrr_max_translation_search)&(rotations<self.mrr_max_rotation_search)]
        vins = vins[(distances_euclidian<self.mrr_max_translation_search)&(rotations<self.mrr_max_rotation_search)]
        return np.array(vins), np.array(indexes), np.array(distances)
            


    
    
                
    

        

        