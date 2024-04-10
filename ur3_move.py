#!/usr/bin/env python3
import sys
import rospy
import actionlib
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg


class UR3_Control:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur3_control', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_planning_time(5)
        print(self.move_group.get_current_pose().pose)
        self.client = actionlib.SimpleActionClient('move_group', moveit_msgs.msg.MoveGroupAction)
        self.client.wait_for_server()

        self.paper_local_info = None
        self.paper_global_info = None
        self.cam_ee_offset = self.define_cam_ee_offset()
        self.pen_ee_offset = None  # Will be set by user input

    @staticmethod
    def define_cam_ee_offset():
        cam_ee_offset = geometry_msgs.msg.Pose()
        cam_ee_offset.position.x = 0.02806  # 28.06 mm camera facing down to the paper
        cam_ee_offset.position.z = 0.05358  # 53.58 mm
        return cam_ee_offset

    def define_pen_ee_offset(self, pen_length):
        # usr need to input pen's length to define the offset
        self.pen_ee_offset = geometry_msgs.msg.Pose()
        self.pen_ee_offset.position.z = pen_length

    def move_to_goals(self, pen_goals):

        # Get initial pose
        initial_pose = self.move_group.get_current_pose().pose
        print("Initial Pose:")
        print(initial_pose)

        for pen_goal in pen_goals:
            arm_goal = geometry_msgs.msg.Pose()
            arm_goal.position.x = pen_goal.position.x
            arm_goal.position.y = pen_goal.position.y
            arm_goal.position.z = pen_goal.position.z - self.pen_ee_offset.position.z

            # 使用转换后的arm_goal作为移动的目标
            self.move_group.set_pose_target(arm_goal)
            arm_goal.pose.orientation.x = 1.0
            plan = self.move_group.plan()

            if plan[0]:
                print(f"Plan found for target position: {pen_goal}")
                self.move_group.go(wait=True)
            else:
                print(f"Failed to find a plan for target position: {pen_goal}")

        print(self.move_group.get_current_pose().pose)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def transform_to_global(self, local_corners, distance):
        # get current ur3's ee coord
        current_ee_global = self.move_group.get_current_pose().pose
        # define camera's ee offset
        cam_ee_offset = self.define_cam_ee_offset()

        global_corners = []
        for (x_local, y_local, _) in local_corners:
            # convert local to global
            global_x = current_ee_global.position.x + x_local - cam_ee_offset.position.x
            global_y = current_ee_global.position.y + y_local
            global_z = current_ee_global.position.z + distance + cam_ee_offset.position.z
            global_corners.append((global_x, global_y, global_z))

        return global_corners

    def define_paper_global_coord(self, paper_local_info):
        local_corners = paper_local_info['corners']
        distance = paper_local_info['distance']  # same z (distance) val for all corners

        # convert all local points to global points
        self.paper_global_info = self.transform_to_global(local_corners, distance)
        return self.paper_global_info

    def run(self, pen_goals):
        self.define_pen_ee_offset(0.15)  # define pen's length
        self.move_to_goals(pen_goals)
