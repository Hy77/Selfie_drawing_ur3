#!/usr/bin/env python3
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def move_robot():
    rospy.init_node('ur3_simple_move', anonymous=True)

    # Connect to the action server
    client = actionlib.SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                                          FollowJointTrajectoryAction)
    rospy.loginfo("Waiting for joint trajectory action server...")
    client.wait_for_server()

    # Define a single joint trajectory point
    point = JointTrajectoryPoint()
    point.positions = [0.0, -1.57, 1.57, -1.57, 0.0, 0.0]  # Example positions for each joint
    point.time_from_start = rospy.Duration(5.0)

    # Define the joint trajectory
    trajectory = JointTrajectory()
    trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                              'wrist_2_joint', 'wrist_3_joint']
    trajectory.points.append(point)

    # Define the goal
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = trajectory

    # Send the goal
    rospy.loginfo("Sending goal...")
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))

    # Check the result
    if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Action succeeded!")
    else:
        rospy.loginfo("Action failed.")


if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
