# Data-Efficinet Decentralized Visual SLAM Project

This is a python implementation of a data-efficient approach of Decentralized Visual SLAM. This was implemented by Team 13 (Aishwarya Unnikrishnan, Devesha Tewari, Lu Wen, and Haonan Chang) for the course EECS 568: Mobile Robotics.

!! TODO INSERT FINAL RESULT GIF !!

Please refer to our [paper](!! TODO HERE!!). 

Please refer to our [website](https://decentr-vslam.github.io/Team13_Decentralized-Visual-SLAM/)

This implementation is based on the paper [Data-Efficient Decentralized Visual SLAM](https://arxiv.org/pdf/1710.05772.pdf) by Titus Cieslewski, Siddharth Choudhary and Davide Scaramuzza.

## Getting Started

You may want to reference our paper before running this code. Please ensure that you have all the listed prerequisites before proceeeding with installation and running the system.

### Prerequisites:


#### ORB-SLAM2
We use [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) for Visual Odometry. Ensure that you have its dependencies (C++11, Pangolin, OpenCV, Eigen3) before installing based on their directions.

#### ROS

Install the [ROS ecosystem](http://wiki.ros.org/ROS/Installation). 
We have tested our system on **Ubuntu 16.04** and **ROS Kinetic**.
Additionally, install [catkin tools](http://catkin-tools.readthedocs.org/en/latest/installing.html), [vcstool](https://github.com/dirk-thomas/vcstool), OpenCV-nonfree dev, autoconf and libglew-dev.

```
sudo add-apt-repository --yes ppa:xqms/opencv-nonfree
sudo apt-get update
sudo apt-get install python-catkin-tools python-vcstool libopencv-nonfree-dev autoconf libglew-dev
```
#### ROS dslam package
This implementation uses the ROS services to be running with our code. After installing ROS, execute the following: 

```
# Create a new catkin workspace if needed:
mkdir -p my_ws/src
cd my_ws
catkin config --init --mkdirs --extend /opt/ros/<YOUR VERSION> --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release

# Clone dslam:
cd src
git clone git@github.com:uzh-rpg/dslam_open.git

# Clone dependencies:
vcs-import < dslam_open/dependencies.yaml

# Build:
catkin build
```
#### NetVLAD

!! TODO !!

#### Distributed Trajectory Estimation

Please follow the provided instruction to build the [distributed-mapper 'feature/logging' branch ](https://github.com/CogRob/distributed-mapper/tree/feature/logging) in an arbitrary location. The unit tests are not necessary and they may require extra dependencies. You will require the 'runDistributedMapper' executable, and take note of its path location to reference in our system. 

### Installing

!! TODO !!

## Running Sample

!! TODO !!
```
Example !! TODO !!
```


## Acknowledgments
* Thank you to Maani Ghaffari Jadidi, our course instructor for his support.
* Thank you to Titus Cieslewski, Siddharth Chourdhary and Davide Scaramuzza who authored the paper ["Data-Efficient Decentralized Visual SLAM"](https://arxiv.org/pdf/1710.05772.pdf), which our work was based on. 
* Thank you to the contributors of the folowing repositories which we utilized in our project:
    * [distributed-mapper 'feature/logging' branch ] by Siddharth Choudhary, Luca Carlone, Carlos Nieto and John Rogers
