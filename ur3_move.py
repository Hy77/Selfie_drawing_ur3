#!/usr/bin/env python3
import sys
import rospy
import actionlib
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

def move_robot():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur3_move', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Set planning time and goal time tolerance
    move_group.set_planning_time(5)  # Increase planning time to 10 seconds

    # Set tolerance
    # move_group.set_goal_position_tolerance(0.01)  # Position tolerance in meters
    # move_group.set_goal_orientation_tolerance(0.1)  # Orientation tolerance in radians

    # Get initial pose
    initial_pose = move_group.get_current_pose().pose
    print("Initial Pose:")
    print(initial_pose)

    # Create an action client
    client = actionlib.SimpleActionClient('move_group', moveit_msgs.msg.MoveGroupAction)
    client.wait_for_server()

    # Get user input for target positions
    targets = []
    while True:
        user_input = input("Enter a target position (x, y, z) or 'done' to finish: ")
        if user_input.lower() == 'done':
            break
        try:
            position = [float(coord) for coord in user_input.split(',')]
            if len(position) == 3:
                targets.append(position)
            else:
                print("Invalid input. Please enter three coordinates separated by commas.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    for target in targets:
        pose_goal = geometry_msgs.msg.PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = target[0]
        pose_goal.pose.position.y = target[1]
        pose_goal.pose.position.z = target[2]
        pose_goal.pose.orientation.x = 1.0  # Set a default orientation

        # Set the goal as a position-only goal
        move_group.set_pose_target(pose_goal)

        # Plan the motion
        plan = move_group.plan()

        # Check if the plan was successful
        if plan[0]:
            print(f"Plan found for target position: {target}")

            # Execute the motion
            move_group.go(wait=True)
        else:
            print(f"Failed to find a plan for target position: {target}")

    print(move_group.get_current_pose().pose)
    move_group.stop()
    move_group.clear_pose_targets()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
