# python imports
import threading
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

# draco imports
from draco_slam.utils.topics import *
from draco_slam.msg import RingKey,DataRequest,KeyframeImage,LoopClosure,PoseHistory,Dummy
from draco_slam.multi_robot_registration import MultiRobotRegistration

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
            [self.feature_sub, self.odom_sub], 20, 
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
        else:
            self.vin = None

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
        self.state_sub = rospy.Subscriber(STATE_UPDATE_TOPIC,PoseHistory,callback=self.listen_for_state,queue_size=5)

        # dummy topic to make global registration run
        self.dummy_pub = rospy.Publisher("/" + self.rov_id + "/slam/global_reg_run", Dummy, queue_size=5)
        self.dummy_sub = rospy.Subscriber("/" + self.rov_id + "/slam/global_reg_run", Dummy, callback=self.global_registration_callback,
                                                                                queue_size=50)
        
        # shutdown sub
        self.shutdown_sub = rospy.Subscriber("/" + self.rov_id + "/slam/shutdown", Dummy, callback=self.shutdown_callback,
                                                                                queue_size=50)

        # dummy topic to make the search thread run
        self.search_pub = rospy.Publisher("/" + self.rov_id + "/slam/search", Dummy, queue_size=5)
        self.search_sub = rospy.Subscriber("/" + self.rov_id + "/slam/search", Dummy, callback=self.search_callback,
                                                                                queue_size=50)

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
            if self.nssm_params.enable  and self.add_nonsequential_scan_matching():
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
            '''if len(job_indexes) != 0:
                vin_for_jobs, job_indexes, distances = self.filter_by_distance2(self.keyframes[-2].pose,
                                                                                np.array(vin_for_jobs),
                                                                                np.array(job_indexes),
                                                                                np.array(distances))'''

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
        
        self.lock.acquire()
        keyframes_ = list(self.keyframes) # copy the keyframes and dump the lock
        self.lock.release()

        # loop over the list of requested keyframes
        for i in list(msg.data):

            # fetch the frame
            frame = keyframes_[i] 
            pose = g2n(frame.pose3) #get the 3D pose as a numpy array
            
            #prep the message
            msg = KeyframeImage()
            msg.cloud = n2r(np.c_[frame.points, np.zeros_like(frame.points[:,0])], "PointCloudXYZ") # push the point cloud
            msg.pose = [pose[0], pose[1], pose[2], pose[5]] # push the pose
            msg.vin = self.vin # push our own vehicle ID 
            msg.requester_vin = msg.requester_vin
            msg.keyframe_id = i
            self.key_frame_pub.publish(msg)

    def keyframe_callback(self, msg : KeyframeImage) -> None:
        pass

    def loop_closure_callback(self, msg : LoopClosure) -> None:
        pass

    def listen_for_state(self, msg : PoseHistory) -> None:
        pass

    def global_registration_callback(self, msg : Dummy) -> None:
        pass

    def shutdown_callback(self, msg : Dummy) -> None:
        pass

    def search_callback(self, msg : Dummy) -> None:
        pass

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

    
    
                
    

        

        