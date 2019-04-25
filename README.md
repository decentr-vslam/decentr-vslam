# Data-Efficient Decentralized Visual SLAM Project

This is a Python implementation of the paper [Data-Efficient Decentralized Visual SLAM](https://arxiv.org/pdf/1710.05772.pdf) by Titus Cieslewski, Siddharth Choudhary and Davide Scaramuzza.

This was implemented by Team 13 (Aishwarya Unnikrishnan, Devesha Tewari, Lu Wen, and Haonan Chang) for the course EECS 568: Mobile Robotics at University of Michigan. 

Kindly refer to our [report](https://github.com/decentr-vslam/Team13_Decentralized-Visual-SLAM/blob/master/Paper.pdf) and our [website](https://decentr-vslam.github.io/Team13_Decentralized-Visual-SLAM/). See our [wiki](https://github.com/decentr-vslam/decentr-vslam/wiki) for further technical troubleshooting issues.

# 1. Getting Started

You may want to reference our report before running the code and ensure that you installed all the listed prerequisites beforehand.

## Prerequisites:


### ORB-SLAM2
Install [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) for Visual Odometry. You must have the following dependencies: C++11, Pangolin, OpenCV, Eigen3.

### ROS

Install the [ROS ecosystem](http://wiki.ros.org/ROS/Installation). 
We tested on **Ubuntu 16.04** and **ROS Kinetic**.
Additionally, install [catkin tools](http://catkin-tools.readthedocs.org/en/latest/installing.html), [vcstool](https://github.com/dirk-thomas/vcstool), OpenCV-nonfree dev, autoconf and libglew-dev.

```
sudo add-apt-repository --yes ppa:xqms/opencv-nonfree
sudo apt-get update
sudo apt-get install python-catkin-tools python-vcstool libopencv-nonfree-dev autoconf libglew-dev
```
### ROS DSLAM package
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

### Distributed Trajectory Estimation

Build the [distributed-mapper 'feature/logging' branch ](https://github.com/CogRob/distributed-mapper/tree/feature/logging) in the folder you've cloned the code. The unit tests are not necessary and they may require extra dependencies. You will require the 'runDistributedMapper' executable to run decentralized optimization.


### Our Decentralized Visual SLAM System Repository

Clone the repository:

```
git clone https://github.com/decentr-vslam/decentr-vslam
``` 
Download the [data](https://drive.google.com/drive/folders/13FMYv0bRFj3eGPpzbnmsDGsUBE_OyeG9) to run the experiments. Extract the kitti/ and robotcar_netvlad_feats/ folders inside clone of the repository.


## Generate NetVLAD descriptors

You can simply download the [NetVLAD descriptors](https://drive.google.com/open?id=1_JrwkJS9EcG4KOkXQfEB_Zzcyfrj3x0o) on KITTI dataset and use it for simulation.

If you want to generate NetVLAD descriptors for other dataset, we also provide with the codes. 

First, download the well-trained [checkpoint](https://drive.google.com/file/d/1ynkQKhzvgK5pkyjwjkwUE44dvUq4B1EQ/view?usp=sharing). Put the ```checkpoint``` folder under the same folder of NetVLAD. Then run the file ```getNetVLAD.py``` to generate a ```.json``` file of netvlad descriptors.

## Configure Parameters

Due to the amount of data generated, we have provided parameters which dictate if data should be generated or reloaded. Once the system was run once, you can change the following parameters to load the previously generated data. This is found in main.py.
```
# Configure params for generation/loading of data
# (1) - Generate data (default)
# (0) - Load previously generated data
regen_data = 1
regen_robots = 1
regen_stream = 1
```

# 2. Running the System 

Run a process of the verification_request_server from the ROS package in the same folder you're executing the main function from.
```
../../<insert your ROS workspace>/devel/lib/dslam/verification_request_server temp_request.txt temp_result.txt temp_lock.txt
```

Run the system
```
python3 main.py

```
# 3. Make Statics

If you need to watch the average static error (ATE):
```
Use evalAccuracy in static_tool moudule.
```

# 4. Checkpoints for Running the Code

## Data is loaded:
![GenData](/gifs/gendata.gif)


## Decentralized Visual Place Recognition (DVPR):
![DVPR](/gifs/DVPR.gif)

## Relative Pose Estimation (RelPose):
![RelPose](/gifs/RelPose.gif)

## Decentralized Optimization (DOpt):
![DOpt](/gifs/dOpt.gif)

# Acknowledgments
* Thank you to Maani Ghaffari Jadidi, our course instructor for his support.
* Thank you to Titus Cieslewski, Siddharth Chourdhary and Davide Scaramuzza who authored the paper ["Data-Efficient Decentralized Visual SLAM"](https://arxiv.org/pdf/1710.05772.pdf), which our work was based on. 
* Thank you to the contributors of the folowing repositories which we utilized in our project:
    * [distributed-mapper 'feature/logging' branch ](https://github.com/CogRob/distributed-mapper/tree/feature/logging) by Siddharth Choudhary, Luca Carlone, Carlos Nieto and John Rogers
