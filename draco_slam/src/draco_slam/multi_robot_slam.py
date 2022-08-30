from logging import raiseExceptions
import rospy

from bruce_slam.slam_ros import SLAMNode
from draco_slam.utils.topics import *
from draco_slam.msg import RingKey,DataRequest,KeyframeImage,LoopClosure,PoseHistory,Dummy

class MultiRobotSLAM(SLAMNode):

    def __init__(self) -> None:
        super().__init__()

    def init_multi_robot(self, ns="~") -> None:


        # get the rov ID and the number of robots in the system
        self.rov_id = rospy.get_param(ns + "rov_number")
        self.number_of_robots = rospy.get_param(ns + "number_of_robots")

        # set the numerical robot ID
        if self.rov_id == "rov_one":
            self.id_code = 1
        elif self.rov_id == "rov_two":
            self.id_code = 2
        elif self.rov_id == "rov_three":
            self.id_code = 3
        else:
            self.id_code = None


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
        size = 50

        # ring key exchange
        self.ring_key_pub = rospy.Publisher(RING_KEY_TOPIC,RingKey,queue_size=5)
        self.ring_key_sub = rospy.Subscriber(RING_KEY_TOPIC,RingKey,callback=self.ring_key_callback, queue_size=size)

        # point cloud request exchange
        self.request_pub = rospy.Publisher(DATA_REQUEST_TOPIC,DataRequest,queue_size=5)
        self.request_sub = rospy.Subscriber(DATA_REQUEST_TOPIC,DataRequest,callback=self.request_callback,queue_size=size) 

        # actual point cloud data exchange
        self.key_frame_pub = rospy.Publisher(KEYFRAME_TOPIC, KeyframeImage, queue_size=5)
        self.key_frame_sub = rospy.Subscriber(KEYFRAME_TOPIC, KeyframeImage, callback=self.keyframe_callback, queue_size=size)
        
        # loop closure exchange
        self.loop_closure_pub = rospy.Publisher(LOOP_CLOSURE_TOPIC,LoopClosure,queue_size=5)
        self.loop_closure_sub = rospy.Subscriber(LOOP_CLOSURE_TOPIC,LoopClosure,callback=self.loop_closure_callback,queue_size=20)

        # state update exchange
        self.state_pub = rospy.Publisher(STATE_UPDATE_TOPIC,PoseHistory,queue_size=5)
        self.state_sub = rospy.Subscriber(STATE_UPDATE_TOPIC,PoseHistory,callback=self.listen_for_state,queue_size=5)

        # dummy topic to make global registration run
        self.dummy_pub = rospy.Publisher("/" + self.rov_id + "/slam/global_reg_run", Dummy, queue_size=5)
        self.dummy_sub = rospy.Subscriber("/" + self.rov_id + "/slam/global_reg_run", Dummy, callback=self.global_registration_callback,
                                                                                queue_size=size)
        
        # shutdown sub
        self.shutdown_sub = rospy.Subscriber("/" + self.rov_id + "/slam/shutdown", Dummy, callback=self.shutdown_callback,
                                                                                queue_size=size)

        # dummy topic to make the search thread run
        self.search_pub = rospy.Publisher("/" + self.rov_id + "/slam/search", Dummy, queue_size=5)
        self.search_sub = rospy.Subscriber("/" + self.rov_id + "/slam/search", Dummy, callback=self.search_callback,
                                                                                queue_size=size)



    def ring_key_callback(self, msg : RingKey) -> None:
        pass

    def request_callback(self, msg : DataRequest) -> None:
        pass

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

        

        