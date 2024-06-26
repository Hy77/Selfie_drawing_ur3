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

2. Install the necessary packages for visualization and solvers:

   ```bash
   pip install simplification

   pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'

   pip install matplotlib

   ```
   
3. Install the RealSense camera package for ROS:

   ```bash
   sudo apt-get install ros-noetic-realsense2-camera
   ```
   
## Running on the Simulator

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

5. **Run the Rospkg & codes:** (paper detection may not be able to work due to it need real UR3.)
   ```commandline
   rosrun selfie_drawer_pkg main.py
   ```
   
## Running on the real UR3

1. Connect to the UR3
   ```bash
   roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.0.250
   ```
2. Turn on MoveIt
   ```commandline
   roslaunch ur3_moveit_config moveit_planning_execution.launch 
   ```
3. Turn on Rviz
   ```commandline
   roslaunch ur3_moveit_config moveit_rviz.launch config:=true
   ```

4. Launch the RealSense camera node:

   ```commandline
   roslaunch realsense2_camera rs_camera.launch align_depth:=true
   ```
   
5. Run the Rospkg & codes
   ```commandline
   rosrun selfie_drawer_pkg main.py
   ```
   
## Notes

- Make sure to source your ROS environment before running any ROS commands:

  ```bash
  source /opt/ros/noetic/setup.bash
  ```
