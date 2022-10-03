# python imports
import gtsam
import numpy as np
from scipy.optimize import shgo, differential_evolution
from scipy.spatial.transform import Rotation

# bruce imports
from bruce_slam.CFAR import *
from bruce_slam import pcl
from bruce_slam.utils.conversions import *
from bruce_slam.slam_objects import Keyframe


class MultiRobotRegistration:
    """A class to integrate register multi-robot observations"""

    def __init__(
        self,
        vin: int,
        number_of_scans: int,
        k_neighbors: int,
        bearing_bins: int,
        max_bearing: float,
        range_bins: int,
        max_range: float,
        max_translation: float,
        max_rotation: float,
        min_overlap: float,
        min_points: int,
        points_ratio: float,
        sampling_points: int,
        iterations: int,
        tolerance: float,
        max_scan_context: float,
        use_count: bool,
        use_ratio: bool,
        use_overlap: bool,
        use_context: bool,
        icp_path: str,
    ) -> None:

        """Class constructor

        Args:
            vin (int): our own robot vin
            number_of_scans (int): the number of scans before and after the scan of interest in a point cloud
            k_neighbors (int): the number of neighbors to query when searching
            bearing_bins (int): number of bearing bins in scan context
            max_bearing (float): the max bearing in scan context
            range_bins (int): the number of range bins in scan context
            max_range (float): the max range in scan context
            max_translation (float): max translation in the global registation solution (meters)
            max_rotation (float): max rotation in the global registration solution (radians)
            min_overlap (float): min overlap between two clouds to be considers valid (bounded between 0 and 1)
            min_points (int): the min number of points in EACH cloud required to send to registration
            points_ratio (float): the ratio of points between two clouds required
            sampling_points (int): the number of sampling points for the SHGO optimizer
            iterations (int): number of iterations for the SHGO optimizer
            tolerance (float): tolerance for SHGO solution
            max_scan_context (float): the max differnce between two scan context images
            use_count (bool): are we using the point cloud count requirnment?
            use_ratio (bool): are we using the point cloud ratio?
            use_overlap (bool): are we using point cloud overlap?
            use_context (bool): are we using context comparison?
            icp_path (str): the path to the ICP config
        """

        # get our own robot vin
        self.robot_id = vin

        # instanciate the class data fields
        self.number_of_scans = number_of_scans
        self.k_neighbors = k_neighbors
        self.bearing_bins = bearing_bins
        self.max_bearing = max_bearing
        self.range_bins = range_bins
        self.max_range = max_range

        # define some paramters for the global optimizer
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.pose_bounds = np.array(
            [
                [-self.max_translation, self.max_translation],
                [-self.max_translation, self.max_translation],
                [-self.max_rotation, self.max_rotation],
            ]
        )
        self.sampling_points = sampling_points
        self.iterations = iterations
        self.tolerance = tolerance

        # define the outliar rejection parameters
        self.min_overlap = min_overlap
        self.min_points = min_points
        self.points_ratio = points_ratio
        self.max_scan_context = max_scan_context

        # define the ablation study parameters
        self.use_count = use_count
        self.use_ratio = use_ratio
        self.use_overlap = use_overlap
        self.use_context = use_context

        # define the optimizer we want to use
        self.optimizer_mode = 1  # 1 is shgo 2 is differential

        # setup ICP
        self.icp = pcl.ICP()
        self.icp.loadFromYaml(icp_path)

        # define some objects to track our slam solution
        self.keyframes = []  # this will be a list of keyframes

        # we need three dummy frames to start
        for i in range(3):
            frame = Keyframe(True, None, pose223(gtsam.Pose2(0, 0, 0)), index=i)
            frame.ring_key = np.sum(
                np.ones((self.bearing_bins, self.range_bins)) * float("inf"), axis=0
            )
            self.keyframes.append(frame)

        # keep a list of the context images and ring keys for quick access
        # self.scan_context_list = [] #list of scan context images
        # self.ring_key_list = [] #list of scan context ring keys

    def get_ring_keys(self) -> list:
        """Return all the ring keys in the system

        Returns:
            list: a list of the systems ring keys.
        """

        keys = []
        for frame in self.keyframes:
            if frame.ring_key is not None:
                keys.append(frame.ring_key)
        return keys

    def add_keyframe_ring_key(self, keyframe: Keyframe) -> None:
        """Add a keyframe that only has a ring key

        Args:
            keyframe (Keyframe): a partially populated keyframe, pose and ring_key only
        """

        if keyframe.index >= len(self.keyframes):
            self.keyframes.append(keyframe)
        else:
            self.keyframes[keyframe.index].ring_key = keyframe.ring_key
            self.keyframes[keyframe.index].bits = keyframe.bits

    def update_keyframe(self, pose: gtsam.Pose2, points: np.array, index: int) -> None:
        """Update a keyframe as we recive it's point cloud

        Args:
            pose (gtsam.Pose2): the pose in the keyframes home ref frame
            points (np.array): point cloud in 2D
            index (int): keyframe index
        """

        frame = Keyframe(
            True, None, pose223(pose), points, source_pose=pose, index=index
        )

        if len(self.keyframes) > index:
            frame.ring_key = self.keyframes[index].ring_key
            frame.bits = self.keyframes[index].bits
            assert self.keyframes[index].index == index
            self.keyframes[index] = frame
        else:
            frame.ring_key = np.sum(
                np.ones((self.bearing_bins, self.range_bins)) * float("inf"), axis=0
            )
            self.keyframes.append(frame)

    def update_scan_context(self, keyframe_id: int) -> None:
        """Get the scan context image for self.keyframes[keyframe_id]

        Args:
            keyframe_id (int): the index of the keyframe we want a scan context image for
        """

        assert keyframe_id > 0  # catch some silly cases
        self.keyframes[keyframe_id].submap = self.get_points(
            [keyframe_id - 1, keyframe_id, keyframe_id + 1], keyframe_id
        )
        context, ring_key = self.get_scan_context_aggragated(
            self.keyframes[keyframe_id].submap
        )
        self.keyframes[keyframe_id].ring_key = ring_key
        self.keyframes[keyframe_id].context = context

    def get_points(self, frames=None, ref_frame=None, return_keys=False):
        """
        - Accumulate points in frames
        - Transform them to reference frame
        - Downsample points
        - Return the corresponding keys for every point

        """
        if frames is None:
            frames = range(self.current_key)
        if ref_frame is not None:
            if isinstance(ref_frame, gtsam.Pose2):
                ref_pose = ref_frame
            else:
                ref_pose = self.keyframes[ref_frame].pose

        # Add empty point in case all points are empty
        if return_keys:
            all_points = [np.zeros((0, 3), np.float32)]
        else:
            all_points = [np.zeros((0, 2), np.float32)]
        for key in frames:
            if ref_frame is not None:
                points = self.keyframes[key].points
                pose = self.keyframes[key].pose
                transf = ref_pose.between(pose)
                transf_points = Keyframe.transform_points(points, transf)
            else:
                transf_points = self.keyframes[key].transf_points

            if return_keys:
                transf_points = np.c_[
                    transf_points, key * np.ones((len(transf_points), 1))
                ]
            all_points.append(transf_points)

        all_points = np.concatenate(all_points)
        if return_keys:
            return pcl.downsample(all_points[:, :2], all_points[:, (2,)], 0.5)
        else:
            return pcl.downsample(all_points, 0.5)

    def get_scan_context_aggragated(self, points: np.array) -> np.array:
        """Perform scan context for an aggragated point cloud

        Args:
            points (np.array): the point cloud we want converted to a ring key and context image

        Returns:
            np.array: the ring key and scan context image
        """

        # instanciate the image
        polar_image = np.zeros((self.bearing_bins, self.range_bins))

        # convert to discrete polar coords
        r_cont = np.sqrt(
            np.square(points[:, 0]) + np.square(points[:, 1])
        )  # first contiuous polar coords
        b_cont = abs(
            np.degrees(np.arctan2(points[:, 0], points[:, 1]))
        )  # * np.sign(points[:,0])
        r_dis = np.array((r_cont / self.max_range) * self.range_bins).astype(
            np.uint16
        )  # discret coords
        b_dis = np.array(
            (((b_cont / self.max_bearing) + 1) / 2) * self.bearing_bins
        ).astype(np.uint16)

        # clip the vales
        r_dis = np.clip(r_dis, 0, self.range_bins - 1)
        b_dis = np.clip(b_dis, 0, self.bearing_bins - 1)

        # populate the image
        for i, j in zip(b_dis, r_dis):
            polar_image[i][j] = 1

        # build the ring key
        ring_key = np.sum(polar_image, axis=0)

        return polar_image, ring_key

    def global_pose_optimization_routine(
        self, source_points, source_pose, target_points, target_pose, point_noise=0.5
    ):
        """Build a routine for SHGO to optimize
        source_points: the source point cloud
        source_pose: the frame for the source points
        target_points: the target point cloud
        target_pose: target points pose initial guess
        returns: a functon to be called and optimized by SHGO
        """

        # a container for the poses tested by the optimizer
        pose_samples = []

        # Build a grid that will fit the target points
        xmin, ymin = np.min(target_points, axis=0) - 2 * point_noise
        xmax, ymax = np.max(target_points, axis=0) + 2 * point_noise
        resolution = point_noise / 10.0
        xs = np.arange(xmin, xmax, resolution)
        ys = np.arange(ymin, ymax, resolution)
        target_grids = np.zeros((len(ys), len(xs)), np.uint8)

        # conver the target points to a grid
        r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
        c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
        r = np.clip(r, 0, target_grids.shape[0] - 1)
        c = np.clip(c, 0, target_grids.shape[1] - 1)
        target_grids[r, c] = 255

        # dilate this grid
        dilate_hs = int(np.ceil(point_noise / resolution))
        dilate_size = 2 * dilate_hs + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs)
        )
        target_grids = cv2.dilate(target_grids, kernel)

        def subroutine(x):
            """The subroutine to be run at every step by the scipy optimizer
            x: the pose of the source points, [x, y, theta]
            """

            # conver this pose to a gtsam object
            delta = n2g(x, "Pose2")
            sample_source_pose = source_pose.compose(delta)
            sample_transform = target_pose.between(sample_source_pose)

            # transform the source points using sample_transform
            points = Keyframe.transform_points(source_points, sample_transform)

            # convert the points into discrete grid coords
            r = np.int32(np.round((points[:, 1] - ymin) / resolution))
            c = np.int32(np.round((points[:, 0] - xmin) / resolution))

            # get the points that are in an occupied grid cell
            inside = (
                (0 <= r)
                & (r < target_grids.shape[0])
                & (0 <= c)
                & (c < target_grids.shape[1])
            )

            # tabulate cost, the number of source points inside an occupied cell
            cost = -np.sum(target_grids[r[inside], c[inside]] > 0)

            # log the pose sample
            pose_samples.append(np.r_[g2n(sample_source_pose), cost])

            return cost

        return subroutine, pose_samples

    def global_pose_optimization_execute(self, subroutine, bounds=None) -> np.array:
        """Using the provided subroutine, use scipy SHGO to find a pose

        Args:
            subroutine (function): the function to be minimized
            bounds (_type_, optional): _description_. Defaults to None.

        Returns:
            np.array: the minimizer result
        """

        if bounds is None:
            bounds = self.pose_bounds

        if self.optimizer_mode == 1:
            return shgo(
                func=subroutine,
                bounds=bounds,
                n=self.sampling_points,
                iters=self.iterations,
                sampling_method="sobol",
                minimizer_kwargs={"options": {"ftol": self.tolerance}},
            )

        elif self.optimizer_mode == 2:
            return differential_evolution(func=subroutine, bounds=bounds, seed=10)

    def compare_frames(self, frame_self: Keyframe, frame_multi_robot: Keyframe):
        """manage the search for loop closures given two keyframes

        Args:
            frame_self (Keyframe): a keyframe from the robots own SLAM solution
            frame_multi_robot (Keyframe): a keyframe that has been passed to this robot from outside
        """

        # pull the point cloud data needed for this job
        multi_robot_points = (
            frame_multi_robot.submap
        )  # get the points of the outside robot
        my_robot_points = frame_self.submap  # get the points from myself
        slam_pose = g2n(frame_self.pose)  # get my pose as a numpy array

        # catch if the points are none
        if multi_robot_points is None or my_robot_points is None:
            return None, False, "None Points"

        # catch if there are enough points in the clouds
        if self.use_count:
            if (
                len(multi_robot_points) <= self.min_points
                or len(my_robot_points) <= self.min_points
            ):
                return (
                    None,
                    False,
                    "Not enough points "
                    + str(len(multi_robot_points))
                    + " "
                    + str(len(my_robot_points)),
                )

        # catch the ratio of the point clouds
        if self.use_ratio:
            if (len(multi_robot_points) / len(my_robot_points)) < (
                1 / self.points_ratio
            ) or (len(my_robot_points) / len(multi_robot_points)) > self.points_ratio:
                return (
                    None,
                    False,
                    "Ratio Wrong "
                    + str((len(multi_robot_points) / len(my_robot_points)))
                    + " "
                    + str(len(my_robot_points) / len(multi_robot_points)),
                )

        # gate the registration based on scan context results
        if self.use_context:
            if (
                np.sum(abs(frame_self.context - frame_multi_robot.context))
                > self.max_scan_context
            ):
                return (
                    None,
                    False,
                    "Context High "
                    + str(np.sum(abs(frame_self.context - frame_multi_robot.context))),
                )

        # build the subroutine and run the global optimizer to perform global ICP
        subroutine, pose_samples = self.global_pose_optimization_routine(
            multi_robot_points,
            gtsam.Pose2(0, 0, 0),
            my_robot_points,
            gtsam.Pose2(0, 0, 0),
        )
        global_result = self.global_pose_optimization_execute(subroutine)

        # gate based on the outcome of the global optimization
        if global_result.success == False:
            return None, False, "Bad optimizer run"

        # apply the global registration transformation
        r_mtx = Rotation.from_euler("xyz", [0, 0, -global_result.x[2]]).as_matrix()[
            :2, :2
        ]
        reg_points = multi_robot_points.dot(r_mtx) + np.array(
            [global_result.x[0], global_result.x[1]]
        )

        # now get the overlap after the GO-ICP call
        idx, distances = pcl.match(my_robot_points, reg_points, 1, 0.5)
        overlap = len(idx[idx != -1]) / len(idx[0])
        fit_score = np.mean(
            distances[distances < float("inf")]
        )  # we need fit score to derive a covariance

        # gate the overlap
        if self.use_overlap:
            if overlap < self.min_overlap:
                return None, False, "Low Overlap " + str(overlap)

        # refine with standard ICP
        icp_x_form = self.icp.compute(reg_points, my_robot_points, np.eye(3))
        reg_points = reg_points.dot(icp_x_form[1][:2, :2].T) + np.array(
            [icp_x_form[1][0][2], icp_x_form[1][1][2]]
        )

        # get the pose of the SLAM frame of my_slam_points
        pose_slam = gtsam.Pose2(slam_pose[0], slam_pose[1], -slam_pose[2])

        # register the multi-robot points into my own global frame
        r_mtx = Rotation.from_euler("xyz", [0, 0, -slam_pose[2]]).as_matrix()[:2, :2]

        # get the theta rotation angle for ICP
        r_mtx_icp = np.row_stack(
            (np.column_stack((icp_x_form[1][:2, :2].T, [0, 0])), [0, 0, 1])
        )
        theta_icp = Rotation.from_matrix(r_mtx_icp).as_euler("xyz")[2]

        # build the pose chain between frames
        global_opti = gtsam.Pose2(
            global_result.x[0], global_result.x[1], -global_result.x[2]
        )
        icp_opti = gtsam.Pose2(icp_x_form[1][0][2], icp_x_form[1][1][2], theta_icp)
        pose_between_frames = gtsam.Pose2.compose(global_opti, icp_opti)

        # build the pose chain in the global frame
        pose_global = np.zeros((1, 2))
        pose_global = pose_global.dot(pose_between_frames.matrix()[:2, :2])
        pose_global = pose_global + np.array(
            [pose_between_frames.x(), pose_between_frames.y()]
        )
        pose_global = pose_global.dot(pose_slam.matrix()[:2, :2])
        pose_global = pose_global + np.array([pose_slam.x(), pose_slam.y()])
        pose_global = gtsam.Pose2(
            pose_global[0][0],
            pose_global[0][1],
            -(pose_between_frames.theta() + pose_slam.theta()),
        )

        # log the outcomes
        return (
            [overlap, fit_score, pose_between_frames, pose_global, None],
            True,
            "Good",
        )
