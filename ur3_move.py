#!/usr/bin/env python3
import sys
import rospy
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
    move_group.set_planning_time(10)  # Increase planning time to 10 seconds

    # Set tolerance
    move_group.set_goal_position_tolerance(0.01)  # Position tolerance in meters
    move_group.set_goal_orientation_tolerance(0.1)  # Orientation tolerance in radians

    # Get initial pose
    initial_pose = move_group.get_current_pose().pose
    print("Initial Pose:")
    print(initial_pose)

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

    # Move to each target position
    for target in targets:
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 0.0
        pose_goal.position.x = target[0]
        pose_goal.position.y = target[1]
        pose_goal.position.z = target[2]

        move_group.set_pose_target(pose_goal)

        # Check if the position is reachable
        if move_group.plan():
            print(f"Moving to target position: {target}")
            move_group.go(wait=True)
        else:
            print(f"Target position {target} is not reachable.")

        # Print out mission success msg
        print(f"{target} success.")

    # Return to initial pose
    # move_group.set_pose_target(initial_pose)
    # move_group.go(wait=True)

    move_group.stop()
    move_group.clear_pose_targets()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
