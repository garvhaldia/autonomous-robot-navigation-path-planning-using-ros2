# autonomous robot navigation system
This project implements a Deep Reinforcement Learning (DRL) approach for autonomous robot navigation in complex environments using ROS2 (Robot Operating System 2) and Gazebo simulation. The implementation leverages the TD3 algorithm to enable robots to navigate to target positions while avoiding obstacles using LiDAR sensor data.

## Features

- Autonomous point-to-point navigation in unknown environments
- Obstacle avoidance using LiDAR sensor data
- Deep Reinforcement Learning with TD3 algorithm
- Customizable reward functions and network architectures
- Training and testing in Gazebo simulation environment
- ROS2 integration for realistic robot control
- Visualization tools for tracking learning progress

## Technologies Used

- *ROS2*: Framework for robot software development
- *Gazebo*: 3D robotics simulator
- *PyTorch*: Deep learning framework
- *Python 3*: Programming language
- *TensorBoard*: Visualization tool for training metrics
- *Velodyne LiDAR*: Sensor simulation package
- *RViz*: Visualization tool for robot state and sensor data



### Prerequisites

- Ubuntu 20.04 or higher
- ROS2 Foxy or newer
- Python 3.8 or higher
- PyTorch 1.7 or higher
- CUDA (optional, for GPU acceleration)

### Setup Instructions

1. *Install ROS2*

   Follow the [official ROS2 installation guide](https://docs.ros.org/en/foxy/Installation.html).

2. *Create a workspace*

   bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   

3. *Clone this repository*

   bash
   git clone https://github.com/yourusername/DRL_robot_navigation_ros2.git
   

4. *Install dependencies*

   bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   pip install torch numpy tensorboard
   

5. *Build the workspace*

   bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   

## Usage

### Training

1. *Start the training environment*

   bash
   source ~/ros2_ws/install/setup.bash
   ros2 launch td3 training_simulation.launch.py
   

   The training process will start automatically. The robot will learn to navigate towards randomly generated goals while avoiding obstacles.

2. *Monitor training progress*

   bash
   tensorboard --logdir ~/ros2_ws/src/td3/scripts/runs
   

   Open your browser and navigate to http://localhost:6006 to view training metrics.

### Testing

1. *Test a trained model*

   bash
   source ~/ros2_ws/install/setup.bash
   ros2 launch td3 test_simulation.launch.py
   

   This will load a pre-trained model and run it in the simulation environment.

## Algorithm Description

### TD3 (Twin Delayed Deep Deterministic Policy Gradient)

The TD3 algorithm is an advanced actor-critic reinforcement learning approach that improves upon DDPG by addressing function approximation errors. Key features:

- *Twin Critics*: Uses two Q-functions to reduce overestimation bias
- *Delayed Policy Updates*: Updates policy less frequently than critics to reduce variance
- *Target Policy Smoothing*: Adds noise to the target action to make the policy robust to noise

### Neural Network Architecture

- *Actor Network*: Maps states to actions
  - Input layer → 800 neurons → 600 neurons → Output layer
  - ReLU activation for hidden layers, Tanh for output

- *Critic Networks*: Evaluates state-action pairs
  - Two identical networks for TD3's twin critics
  - Separate input pathways for state and action, merged into shared layers

### Training Process

1. The robot explores the environment by taking actions based on the current policy plus exploration noise
2. Experiences (state, action, reward, next_state, done) are stored in a replay buffer
3. Batches of experiences are randomly sampled for training
4. Critics are updated to minimize TD error
5. Actor is updated to maximize expected Q-value (less frequently than critics)
6. Target networks are updated via soft updates to stabilize training

### Reward Function

- Positive reward for reaching the goal (+100)
- Negative reward for collisions (-100)
- Small negative reward per step to encourage efficiency (-0.1)
- Distance-based component to guide towards goal

### State and Action Space

- *State Space*: 
  - LiDAR scan data (20 distance measurements around the robot)
  - Relative position to goal
  - Robot orientation

- *Action Space*:
  - Linear velocity (forward/backward movement)
  - Angular velocity (rotation)

## Results

The trained model demonstrates:
- Successful navigation to goal positions
- Effective obstacle avoidance
- Generalization to different goal positions not seen during training
- Smooth and efficient trajectories

## Customization

You can customize various aspects of the implementation:

- *Neural Network Architecture*: Modify network sizes in train_velodyne_node.py
- *Reward Function*: Adjust the reward components in the get_reward method
- *Hyperparameters*: Tune learning rates, discount factors, etc.
- *Environment*: Modify the world file in worlds/ directory

## Key Parameters

- *GOAL_REACHED_DIST*: 0.3 (meters)
- *COLLISION_DIST*: 0.35 (meters)
- *TIME_DELTA*: 0.2 (seconds)
- *Environment Dimension*: 20 (number of laser samples)
- *Replay Buffer Size*: 1,000,000 experiences
- *Batch Size*: 100
- *Discount Factor*: 1.0
- *Policy Noise*: 0.2
- *Noise Clip*: 0.5
- *Policy Update Frequency*: Every 2 critic updates

## Running on Different Hardware

For systems without a GPU:
- The code will automatically use CPU if CUDA is not available
- Training will be significantly slower on CPU-only systems
- Consider reducing the neural network size for faster training

## Troubleshooting

Common issues and solutions:
- *Gazebo crashes*: Ensure you have sufficient RAM (8GB minimum recommended)
- *ROS2 communication issues*: Check that all ROS2 topics are correctly configured
- *Training instability*: Try adjusting the learning rates or using a smaller network

