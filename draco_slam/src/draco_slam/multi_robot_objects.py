import gtsam
import numpy as np


class ICPResultInterRobot(object):
    """Stores the results of an inter robot loop closure"""

    def __init__(
        self,
        source_key: int,
        target_key: int,
        source_pose: gtsam.Pose2,
        target_pose: gtsam.Pose2,
        fit_score: float,
        context_error: float,
        vin: int,
        covariance=None,
        dcs=False,
    ) -> None:
        """Class constructor

        Args:
            source_key (int): the ID of the frame in the outside SLAM solution
            target_key (int): the ID of the frame in my own slam solution
            source_pose (gtsam.Pose2): the pose of the ouside keyframe, in my own global frame
            target_pose (gtsam.Pose2): the pose of the inside keyframe, obvi in my frame
            fit_score (float): the ICP fit score
            context_error (float): the difference in scan context images between the two point clouds
            vin (int): the vehicle ID number we have closed the loop with
            covariance (_type_, optional): ICP covariance matrix. Defaults to None.
            dcs (bool, optional): Are we using DCS factors for this loop?. Defaults to False.
        """

        # source is the robots own slam solution
        # target is the outside robot
        self.source_key = source_key
        self.target_key = target_key
        self.source_pose = source_pose
        self.target_pose = target_pose
        if covariance is None:
            self.cov = np.eye(3) * fit_score * 20.0
        else:
            self.cov = np.array(covariance)

        self.estimated_transform = self.source_pose.between(
            self.target_pose
        )  # get the transform between source-target
        self.context_error = context_error
        self.inserted = False
        self.vin = vin
        self.dcs = dcs
