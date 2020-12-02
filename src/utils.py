#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, WrenchStamped

class Omni(object):
    def __init__(self, joint_topic='/Touch3D/joint_states', pose_topic='/Touch3D/pose'):
        self.state = JointState()
        self.pose  = JointState()
        rospy.Subscriber(joint_topic, JointState, self.callback_state, queue_size=1)
        rospy.Subscriber( pose_topic, JointState,  self.callback_pose, queue_size=1)

    def callback_state(self, msg):
        self.state = msg

    def callback_pose(self, msg):
        self.pose = msg

    def get_state(self):
        return self.state.position

    def get_pose(self):
        return self.pose.position

class UR5(object):
    def __init__(self, topic='/arm_controller/state'):
        self.state = JointTrajectoryControllerState()
        rospy.Subscriber(topic, JointTrajectoryControllerState, self.callback, queue_size=1)

    def callback(self, msg):
        self.state = msg

    def get_pos(self):
        return self.state.actual.positions

class FTSensor(object):
    def __init__(self, topic='/wrench'):
        self.wrench = WrenchStamped()
        rospy.Subscriber(topic, WrenchStamped, self.callback, queue_size=1)

    def callback(self, msg):
        self.wrench = msg

    def getWrench(self):
        return self.wrench

def PS(origin, position=[0,0,0], orientation=[0,0,0,1]):
    h = Header()
    h.frame_id = origin
    h.stamp = rospy.Time().now()
    
    p=Pose()

    if type(position) == Point:
        p.position = position
    else:
        p.position=Point(*position)
    
    if type(orientation) == Quaternion:
        p.orientation = orientation 
    elif len(orientation) == 4:
        p.orientation=Quaternion(*orientation)
    elif len(orientation) == 3:
        p.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*orientation))
    else:
        p.orientation = Quaternion(0,0,0,1)
    
    return PoseStamped(h,p)