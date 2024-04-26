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
        # self.client = actionlib.SimpleActionClient('/eff_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.client = actionlib.SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                                                   FollowJointTrajectoryAction)

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

    def get_current_camera_ee_xyz(self):
        # 获取当前UR3末端执行器的位姿
        current_ee_pose_stamped = self.get_current_ee_pose()
        current_ee_pose = current_ee_pose_stamped.pose

        # 提取末端执行器的x, y, z坐标
        current_ee_x = current_ee_pose.position.x
        current_ee_y = current_ee_pose.position.y
        current_ee_z = current_ee_pose.position.z

        # 获取摄像头相对于末端执行器的偏移量
        cam_ee_offset = self.define_cam_ee_offset()

        # 根据偏移量计算摄像头的全局坐标
        camera_x = current_ee_x - cam_ee_offset['x']
        camera_y = current_ee_y + cam_ee_offset['y']
        camera_z = current_ee_z - cam_ee_offset['z']

        return (camera_x, camera_y, camera_z)

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
            point.time_from_start = rospy.Duration(5)

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
        # 确定摄像头与机器人基座的偏移量，单位为米
        return {'x': 0.02806, 'y': 0, 'z': 0.05358}

    def transform_to_global(self, local_corners):
        # 获取当前末端执行器的位置和姿态
        current_ee_pose_stamped = self.get_current_ee_pose()
        current_ee_pose = current_ee_pose_stamped.pose

        cam_ee_offset = self.define_cam_ee_offset()

        global_corners = []
        for (x_local, y_local, local_z) in local_corners:
            # 计算全局坐标，其中考虑了摄像头到UR3基座的偏移量和笔的长度
            global_x = current_ee_pose.position.x + y_local + cam_ee_offset['x']
            global_y = current_ee_pose.position.y + x_local + cam_ee_offset['y']
            global_z = current_ee_pose.position.z - local_z - cam_ee_offset['z']  # keep it as same
            global_corners.append((global_x, global_y, global_z))

        return global_corners

    """
    (-0.274, 0.11, 0.485)]
    
    0.438，0.124，0.246
    0.5，-0.097，0.246
    0.18，-0.11，0.246
    0.188，0.095，0.246
    
    corners': [(-0.44664529411008586, 0.14662582645840025, 0.022435507736270677), 
    (-0.44664529411008586, -0.07523352309518085, 0.022435507736270677), 
    (-0.13908039009649906, -0.07523352309518085, 0.022435507736270677), 
    (-0.13908039009649906, 0.14662582645840025, 0.022435507736270677)]}

    glb [(-0.45429802319083784, 0.2987982699088576, 0.02242037856100726), 
    (-0.45429802319083784, 0.0724513905804163, 0.02242037856100726), 
    (-0.14582890205379445, 0.0724513905804163, 0.02242037856100726), 
    (-0.14582890205379445, 0.2987982699088576, 0.02242037856100726)]
    
    
    'corners': [(-0.14811856527496634, 0.10722420683833081, 0.02614328094207563), 
    (-0.4440109794789362, 0.10722420683833081, 0.02614328094207563), 
    (-0.4440109794789362, -0.10196553523907485, 0.02614328094207563), 
    (-0.14811856527496634, -0.10196553523907485, 0.02614328094207563)]
    
    
    [(-0.13983940849368168, -0.3011882869050817, 0.009183161616729552), (-0.4504397775026059, -0.3011882869050817, 0.009183161616729552), (-0.4504397775026059, -0.5320967592256654, 0.009183161616729552), (-0.13983940849368168, -0.5320967592256654, 0.009183161616729552)]
glb [(-0.13983940849368168, -0.3011882869050817, 0.009183161616729552), (-0.4504397775026059, -0.3011882869050817, 0.009183161616729552), (-0.4504397775026059, -0.5320967592256654, 0.009183161616729552), (-0.13983940849368168, -0.5320967592256654, 0.009183161616729552)]
Final detected paper info: {'corners': 
[(-0.13983940849368168, -0.3011882869050817, 0.009183161616729552), 
(-0.4504397775026059, -0.3011882869050817, 0.009183161616729552), 
(-0.4504397775026059, -0.5320967592256654, 0.009183161616729552), 
(-0.13983940849368168, -0.5320967592256654, 0.009183161616729552)]}
    """

    def define_paper_global_coord(self, paper_local_info):
        # convert paper info into coordinates -> convert to global from local
        local_corners = paper_local_info['corners']

        # return self.transform_to_global(local_corners)
        return local_corners

    # control ur3 move to the goal
    def run(self, goals, pen_length):
        # Process all goals
        for goal in goals:
            print(goal)
            self.process_goal(goal, pen_length)
            # Wait for the result to make sure we don't send the next goal too early
            # self.client.wait_for_result()

        # After all movements are done, plot the joint states
        # self.plot_velocities_accelerations()


if __name__ == '__main__':
    try:
        # Initialize the UR3Control class
        ur_control = UR3Control()
        # goals = [(-0.14819689315343929, 0.05677191725504173, 0.026212555279538863),
        #          (-0.44357801592252305, 0.05677191725504173, 0.026212555279538863),
        #          (-0.44357801592252305, -0.15015018571761504, 0.026212555279538863),
        #          (-0.14819689315343929, -0.15015018571761504, 0.026212555279538863)]
        goals = [(-0.14068087322490452, 0.0989002099246277, 0.010020238866979836), (-0.4497068571711998, 0.0989002099246277, 0.010020238866979836), (-0.4497068571711998, -0.11905675615524158, 0.010020238866979836), (-0.14068087322490452, -0.11905675615524158, 0.010020238866979836)]

        ur_control.run([(-0.274, 0.11, 0.485)], 0)
        # ur_control.client.wait_for_result()
        # ur_control.run(goals[0:], 0.25)
        # ur_control.client.wait_for_result()
    except rospy.ROSInterruptException:
        pass
