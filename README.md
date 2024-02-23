# Selfie_Drawing_UR3
This project presents a robotic system capable of transforming a webcam-captured image of a person into a single-color marker drawing within a maximum time frame of three minutes. The system comprises three main subsystems: image processing, tool path planning, and control with easel localization and mechanical design.

In the image processing subsystem, the system employs various levels of sophistication, ranging from simple raster images to smooth bicubic vectorization with optional background removal for enhanced focus on the subject. The tool path planning subsystem evolves from basic, unstructured approaches to advanced, time-optimal strategies that solve the Traveling Salesman Problem (TSP) to determine the most efficient drawing order.

The control, easel localization, and mechanical subsystem ensure precise execution of the drawing, with advancements from basic pen holding and easel positioning to the incorporation of compliance mechanisms and the use of brush pens for improved drawing quality.

Overall, this robotic system demonstrates the integration of computer vision, path optimization, and precise control to achieve a creative and automated artistic process.


# UR3 Simulator Setup

This guide will help you set up a UR3 robot simulator using Gazebo, MoveIt, and ROS Noetic on Ubuntu 20.04.

## Prerequisites

- Ubuntu 20.04
- ROS Noetic
- Gazebo (usually installed with ROS Noetic)

## Installation

1. Update your package list:

   ```bash
   sudo apt-get update
   ```

2. Install the necessary ROS packages for the UR3 robot, Gazebo, and MoveIt:

   ```bash
   sudo apt-get install ros-noetic-ur-gazebo ros-noetic-ur3-moveit-config ros-noetic-moveit
   ```

3. (Optional/tbh idk/why not?) Install the universal-robots package if needed:

   ```bash
   sudo apt-get install ros-$ROS_DISTRO-universal-robots
   ```

## Running the Simulator

1. **Load UR3 Description:**
   Launch the UR3 robot description to set up the URDF model in ROS:

   ```bash
   roslaunch ur_description load_ur3.launch
   ```

2. **Start Gazebo:**
   Launch the Gazebo simulation environment with the UR3 robot:

   ```bash
   roslaunch ur_gazebo ur3_bringup.launch
   ```

3. **Start MoveIt for Motion Planning:**
   Launch MoveIt for motion planning with the UR3 robot in the simulation:

   ```bash
   roslaunch ur3_moveit_config moveit_planning_execution.launch sim:=true
   ```

4. **Start RViz for Visualization:**
   Launch RViz with the MoveIt configuration for the UR3 robot:

   ```bash
   roslaunch ur3_moveit_config moveit_rviz.launch config:=true
   ```

5. **Control the Robot (Optional):**
   If you have a custom Python script to control the robot (e.g., `ur3_move.py`), you can run it:

   ```bash
   ./ur3_move.py
   ```

## Notes

- Make sure to source your ROS environment before running any ROS commands:

  ```bash
  source /opt/ros/noetic/setup.bash
  ```

- You may need to adjust permissions for your script (`ur3_move.py`) to make it executable:

  ```bash
  chmod +x ur3_move.py
  ```
