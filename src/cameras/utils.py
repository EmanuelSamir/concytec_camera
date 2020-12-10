#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import rospy, tf, math, copy, sys
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from apriltag_ros.msg import AprilTagDetectionArray


def pose2mat(pose):
	"""
	Converts pose np.array([[x],[y],[qw],[qx],[qy],[qz]]) into matrix.
	Could work also with 1-D array or list.
	"""
	T = quaternion_matrix(pose[3:])
	T[:3,3] = np.array(pose[:3])
	return np.array(T)

def mat2pose(m):
	"""
	Converts numpy matrix (4x4) into pose np.array([[x],[y],[qw],[qx],[qy],[qz]])
	"""
	q = np.expand_dims(quaternion_from_matrix(m), axis = 1)
	p = np.expand_dims(m[:3,3], axis = 1)
	pose = np.vstack((p,q))
	return pose

def homogenous_transform_inverse(T_in):
	"""
	Calculates the inverse of a Transformation Matrix (4x4).
	"""
	R_in = T_in[:3,:3]
	t_in = T_in[:3,[-1]]
	R_out = R_in.T
	t_out = -np.matmul(R_out,t_in)
	return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))


class KalmanConstantEstimator:
	"""
	Class to estimate a constant value from noisy signals. Made adhoc for a (7,1) pose vector
	Kalman Estimator based on
	http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/WELCH/kalman.3.html
	"""
	def __init__(self, x0 ):
		n = 7
		self.Q = 1e-5*np.eye(n) # Process variance
		self.R = 0.2*np.eye(n) # Measurement variance

		self.x_t = x0
		self.x_t_1 = np.zeros((n,1))

		self.K = 0.1 * np.ones((n,n))
		self.P_t = 1. * np.ones((n,n))
		self.P_t_1 = np.zeros((n,n))

		self.train = []
		
	def prediction(self ):
		self.x_t_1 = copy.copy(self.x_t)
		self.P_t_1 = copy.copy(self.P_t) + self.Q
		
	def correction(self, z):
		self.K = self.P_t_1 * np.linalg.pinv(self.P_t_1 + self.R)
		self.x_t = self.x_t_1 + np.dot(self.K, (z - self.x_t_1))
		self.P_t = np.dot((np.eye(7) - self.K), self.P_t_1)

	def flag_stable(self, N_train = 50, e = 1e-3):
		"""
		Additional function that is constantly checking that state x_t holds
		as a constant for a defined N_train length using standard desviation.
		Used for calibration. Smaller the e, the more precision of the calibrator.
		"""
		flag = False

		if len(self.train) < N_train:
			self.train.append(np.linalg.norm(self.x_t))
		else:
			self.train = self.train[1:]
			self.train.append(np.linalg.norm(self.x_t))
			x = np.std(self.train)
			if x < e:
				flag = True

		return flag

class ArTagTracker:
	def __init__(self, camera_ns):
		"""
		Class to get artag poses for a camera.
		Args: camera_ns:= Camera namespace group (e.g. /camera_1)
		"""
		artag_pose_topic = camera_ns + '/tag_detections'
		rospy.Subscriber(artag_pose_topic, AprilTagDetectionArray, self.cb)
		
		self.poses = dict()

	def cb(self, msg):
		tag_ids = []
		poses = []
		for marker in msg.detections:
			tag_ids.append(marker.id[0]) 
			o = marker.pose.pose.pose.orientation
			p = marker.pose.pose.pose.position
			poses.append(np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w]))
		self.poses = dict(zip(tag_ids, poses))


def euler_to_se3(rpy):
    """Converts Euler angles to an SE3 matrix.
    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.
    Returns:
        numpy.matrixlib.defmatrix.matrix: 4x4 SE3 matrix
    Raises:
        ValueError: if `len(rpy) != 3`.
    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    T_x = np.matrix([
					 [1, 0, 0, 0],
                     [0, np.cos(rpy[0]), -np.sin(rpy[0]), 0],
                     [0, np.sin(rpy[0]), np.cos(rpy[0]), 0],
					 [0, 0, 0, 1]
					 ])
    T_y = np.matrix([
					 [np.cos(rpy[1]), 0, np.sin(rpy[1]), 0],
                     [0, 1, 0, 0],
                     [-np.sin(rpy[1]), 0, np.cos(rpy[1]), 0],
					 [0, 0, 0, 1]
					 ])
    T_z = np.matrix([
					 [np.cos(rpy[2]), -np.sin(rpy[2]), 0, 0],
                     [np.sin(rpy[2]), np.cos(rpy[2]), 0, 0],
                     [0, 0, 1, 0],
					 [0, 0, 0, 1]
					 ])
    T_zyx = T_z * T_y * T_x
    return T_zyx



def calibration_between_two_nodes(camera1_topic, camera2_topic, tag_id, rate, N_train = 50, e = 1e-3):
	tracker_camera1 = ArTagTracker(camera1_topic)
	tracker_camera2 = ArTagTracker(camera2_topic)

	# Calibration T_camera1, T_camera2
	rospy.loginfo("Colocar tag {} visible para {} y {}".format(tag_id, camera1_topic, camera2_topic))
	_ = raw_input("Press Enter to continue...")

	# Check if tag is seen from both cameras
	while True:
		rate.sleep()
		if (tag_id in tracker_camera1.poses) and (tag_id in tracker_camera2.poses):
			break
		if rospy.is_shutdown():
			rospy.loginfo("Out because of ctrl + c.")
			sys.exit(0)

		if not (tag_id in tracker_camera1.poses):
			rospy.loginfo("Tag {} not found in {}.".format(tag_id, camera1_topic) )

		if not (tag_id in tracker_camera2.poses):
			rospy.loginfo("Tag {} not found in {}.".format(tag_id, camera2_topic) )

	rospy.loginfo("Tag {} encontrado. Inicio de calibracion {} -> {}.".format(tag_id, camera1_topic, camera2_topic))

	
	# Initialize pose vector for estimation of p_1_2 (pose of camera 1 with respect to camera 2)
	p_1_2 = np.zeros((7,1))
	filter_1_2 = KalmanConstantEstimator(p_1_2)

	while True:	
		if not (tag_id in tracker_camera1.poses) or not (tag_id in tracker_camera2.poses) or rospy.is_shutdown() :
			rospy.loginfo("Out because of ctrl + c or cameras were not located correctly.")
			sys.exit(0)

		# p^{1}_tag and p^{2}_tag
		p_1_tag = tracker_camera1.poses[tag_id]
		p_2_tag = tracker_camera2.poses[tag_id]

		# T^{1}_tag and T^{2}_tag
		T_1_tag = pose2mat(p_1_tag) 
		T_2_tag = pose2mat(p_2_tag)
		
		# T^{2}_tag -> T^{tag}_2
		T_tag_2 = homogenous_transform_inverse(T_2_tag)

		# T^{1}_tag x T^{tag}_2 -> T^{1}_2
		T_1_2 = np.dot(T_1_tag,T_tag_2)

		# p^{1}_2
		p_1_2 = mat2pose(T_1_2)

		# Using p_1_2 to update kalman filter.		
		filter_1_2.prediction()
		filter_1_2.correction(p_1_2)

		# Read flag. If Kalman filter is enough constant, it will return True.
		flag = filter_1_2.flag_stable(N_train = 50, e = 1e-3)

		if flag:
			rospy.loginfo("Calibration was correct for {} -> {}".format(camera1_topic, camera2_topic))
			break
		rate.sleep()

	# Return constant pose
	return pose2mat(np.squeeze(filter_1_2.x_t))

def calibration_first_node(camera_topic, tag_id, rate, N_train = 50, e = 1e-3):
	# Calibration world -> camera
	tracker_camera = ArTagTracker(camera_topic)
	rospy.loginfo("Colocar tag {} visible para {}".format(tag_id, camera_topic))
	_ = raw_input("Press Enter to continue...")

	while True:
		rate.sleep()
		if (tag_id in tracker_camera.poses):
			break
		if rospy.is_shutdown():
			rospy.loginfo("Out because of ctrl + c.")
			sys.exit(0)
		rospy.loginfo("Tag {} not found in {}.".format(tag_id, camera_topic) )

	rospy.loginfo("Tag {} encontrado. Inicio de calibracion world -> {}.".format(tag_id, camera_topic))
	
	# Initialize pose vector of p_tag_c (pose of world (tag) with respect to camera_topic)
	p_tag_c = np.zeros((7,1))
	filter_tag_c = KalmanConstantEstimator(p_tag_c)

	while True:	
		if not (tag_id in tracker_camera.poses) or rospy.is_shutdown() :
			rospy.loginfo("Out because of ctrl + c or cameras were not located correctly.")
			sys.exit(0)
		# p^{c}_tag
		p_c_tag = tracker_camera.poses[tag_id]
		
		# T^{c}_tag
		T_c_tag = pose2mat(p_c_tag)

		# T^{c}_tag -> T^{tag}_c
		T_tag_c = homogenous_transform_inverse(T_c_tag)

		# p^{tag}_c
		p_tag_c = mat2pose(T_tag_c)

		# Using p_tag_c to update kalman filter.		
		filter_tag_c.prediction()
		filter_tag_c.correction(p_tag_c)

		# Read flag. If Kalman filter is enough constant, it will return True.
		flag = filter_tag_c.flag_stable(N_train = 50, e = 1e-3)

		if flag:
			rospy.loginfo("Calibration was correct for world -> {}".format(camera_topic))
			rospy.loginfo("World is now pose of tag {}".format(tag_id))
			break
		rate.sleep()

	# Return constant pose
	return pose2mat(np.squeeze(filter_tag_c.x_t))
