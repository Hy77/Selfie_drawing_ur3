#!/usr/bin/env python3
import sys
import tf
import math
import rospy
import actionlib
import moveit_commander
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped


class UR3Control:
    def __init__(self):
        # Initialize moveit_commander and a rospy node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur3_control', anonymous=True)

        # Initialize MoveIt! commander for the 'manipulator' group.
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        # Initialize action client
        self.client = actionlib.SimpleActionClient('/eff_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for follow_joint_trajectory server...")
        # self.client.wait_for_server()

        # subscribe joint state
        self.joint_states = []
        self.subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        # Initialize the IK service
        rospy.wait_for_service('compute_ik')
        self.compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

        # Get the current pose of the end effector
        current_ee_pose = self.get_current_ee_pose()
        rospy.loginfo("Current EE pose: " + str(pose_to_list(current_ee_pose.pose)))

    def get_current_ee_pose(self):
        # Get the current pose and return it
        return self.group.get_current_pose()

    def joint_state_callback(self, data):
        # Add time and joint velocities to the joint_states list
        self.joint_states.append((rospy.get_time(), list(data.position), list(data.velocity)))

    def plot_velocities_accelerations(self):
        times, positions, velocities = zip(*self.joint_states)

        velocities = np.array([gaussian_filter1d(v, 5) for v in np.array(velocities).T]).T
        accelerations = np.gradient(velocities, axis=0, edge_order=2)
        jerks = np.gradient(accelerations, axis=0, edge_order=2)

        # Plot smoothed velocities
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        for i in range(velocities.shape[1]):
            plt.plot(times, velocities[:, i], label=f'Velocity Joint {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (rad/s)')
        plt.title('Joint Velocities Over Time')
        plt.legend()

        # Plot smoothed accelerations
        plt.subplot(1, 2, 2)
        for i in range(accelerations.shape[1]):
            plt.plot(times, accelerations[:, i], label=f'Acceleration Joint {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (rad/s^2)')
        plt.title('Joint Accelerations Over Time')

        # Plot smoothed jerks
        plt.figure(figsize=(10, 5))
        for i in range(jerks.shape[1]):
            plt.plot(times, jerks[:, i], label=f'Jerk Joint {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Jerk (rad/s^3)')
        plt.title('Joint Jerks Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compute_ik_pose(self, pose):
        ik_request = PositionIKRequest()
        ik_request.group_name = self.group.get_name()
        ik_request.robot_state = self.robot.get_current_state()
        ik_request.pose_stamped = pose
        ik_request.timeout.secs = 2
        ik_request.avoid_collisions = False

        try:
            ik_response = self.compute_ik(ik_request)
            if ik_response.error_code.val == ik_response.error_code.SUCCESS:
                # Extract the joint positions from the response
                return ik_response.solution.joint_state.position
            else:
                rospy.logerr("IK request failed")
                return None
        except rospy.ServiceException as exc:
            rospy.logerr("Service did not process request: " + str(exc))
            return None

    def move_to_pose(self, pose):
        # Compute the joint positions from the desired pose using IK
        joint_positions = self.compute_ik_pose(pose)

        if joint_positions:
            # Create a trajectory point
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start = rospy.Duration(5.0)

            # Create a trajectory message
            trajectory = JointTrajectory()
            trajectory.joint_names = self.group.get_active_joints()
            trajectory.points.append(point)

            # Create and send the FollowJointTrajectoryGoal
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = trajectory

            rospy.loginfo("Sending goal to the robot...")
            self.client.send_goal(goal)
            self.client.wait_for_result()
            rospy.loginfo("Goal execution completed.")
            return self.client.get_result()
        else:
            rospy.logerr("No valid joint positions found for the given pose")
            return None

    def process_goal(self, goal, pen_length):
        # Always let ur3 facing down
        quaternion = tf.transformations.quaternion_from_euler(math.pi, 0, 0)  # roll=π, pitch=0, yaw=0

        # Define a new desired pose
        desired_pose = PoseStamped()
        desired_pose.header.frame_id = 'base_link'
        desired_pose.pose.position.x = goal[0]
        desired_pose.pose.position.y = goal[1]
        desired_pose.pose.position.z = goal[2] + pen_length
        desired_pose.pose.orientation.x = quaternion[0]
        desired_pose.pose.orientation.y = quaternion[1]
        desired_pose.pose.orientation.z = quaternion[2]
        desired_pose.pose.orientation.w = quaternion[3]

        # Move the robot to the new pose
        result = self.move_to_pose(desired_pose)
        if result:
            rospy.loginfo("Robot moved to new pose")
        else:
            rospy.loginfo("Failed to move the robot to new pose")

        # Get the current pose of the end effector
        current_ee_pose = self.get_current_ee_pose()
        rospy.loginfo("Current ur3 EE pose: " + str(pose_to_list(current_ee_pose.pose)))

        if pen_length != 0:
            # Update pen EE to reflect z-offset
            pen_ee = pose_to_list(current_ee_pose.pose)
            pen_ee[2] -= pen_length
            rospy.loginfo("Current pen EE pose: " + str(pen_ee))

    @staticmethod
    def define_cam_ee_offset():
        # Define camera's offset
        return {'x': 0.02806, 'z': -0.05358}  # x forward 0.02806m，z down 0.05358m

    def transform_to_global(self, local_corners, distance):
        # convert local coord to global coord
        current_ee_global = self.get_current_ee_pose()
        cam_ee_offset = self.define_cam_ee_offset()

        global_corners = []
        for (x_local, y_local, _) in local_corners:
            # Convert local coordinates to global coordinates using camera offsets
            global_x = current_ee_global.position.x + x_local + cam_ee_offset['x']
            global_y = current_ee_global.position.y + y_local
            global_z = current_ee_global.position.z + distance + cam_ee_offset['z']
            global_corners.append((global_x, global_y, global_z))

        return global_corners

    def define_paper_global_coord(self, paper_local_info):
        # convert paper info into coordinates -> convert to global from local
        local_corners = paper_local_info['corners']
        distance = paper_local_info['distance']  # dist between paper & cam

        return self.transform_to_global(local_corners, distance)

    # control ur3 move to the goal
    def run(self, goals, pen_length):
        # Process all goals
        for goal in goals:
            self.process_goal(goal, pen_length)
            # Wait for the result to make sure we don't send the next goal too early
            self.client.wait_for_result()

        # After all movements are done, plot the joint states
        self.plot_velocities_accelerations()


if __name__ == '__main__':
    try:
        # Initialize the UR3Control class
        ur_control = UR3Control()
        ur_control.run([(0.3, -0.1, 0.47)], 0)

    except rospy.ROSInterruptException:
        pass
