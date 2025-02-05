#!/usr/bin/env python

import wandb
import rospy
import math
import jax
import jax.numpy as jnp
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

# Import model utilities
from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.runners.eval_runner import EvalSampledRunner
from jaxmarl.environments.jaxnav import JaxNav

### **Global Variables**
robot_x, robot_y, robot_yaw = -2.5, 2.5, 0.0  # Initial spawn position
goal_x, goal_y = -2.5, 9.5  # Goal position
goal_tolerance = 0.2  # Stop when within 20 cm of the goal

# Initialize ROS node
rospy.init_node("robot_controller", anonymous=True)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

# Visualisation for RViz
marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

### **Function: Visualise Robot Position in RViz**
def visualize_robot_position():
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "robot_path"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = robot_x
    marker.pose.position.y = robot_y
    marker.pose.position.z = 0.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker_pub.publish(marker)

### **Function: Odometry Callback**
def odometry_callback(msg):
    """ Update the robot's position from odometry data. """
    global robot_x, robot_y, robot_yaw
    robot_x = msg.pose.pose.position.x
    robot_y = msg.pose.pose.position.y

    # Convert quaternion to Euler angles (yaw, pitch, roll)
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (_, _, robot_yaw) = euler_from_quaternion(orientation_list)

    rospy.loginfo(f"Robot position: x={robot_x:.2f}, y={robot_y:.2f}, yaw={robot_yaw:.2f}")
    visualize_robot_position()

### **Function: Load Trained Model**
def load_trained_model():
    """ Load the trained model for controlling the robot. """
    run_id = "evelinawg-university-of-oxford/jaxnav-barn/sx0ptcp5"
    api = wandb.Api()
    run = api.run(run_id)
    config = run.config
    run_name = run.name
    proj = run.project
    print('run name', run_name)

    # Load environment
    env = JaxNav(num_agents=1, do_sep_reward=False, **config["env"]["env_params"])
    model_artificat = api.artifact(f"{run.entity}/{run.project}/{run.name}-checkpoint:latest") # NOTE hardcoded
    name = model_artificat.download()
    # Load model parameters
    network = ActorCriticRNN(
        action_dim=env.action_space(agent='agent_0').shape[0], 
        config=config["learning"],)

    network_params = load_params(name + "/model.safetensors")

    return network, network_params, env, config

network, network_params, env, config = load_trained_model()
rng = jax.random.PRNGKey(10)
batch_size = 1  # Adjust value if needed
hidden_size = config["learning"]["HIDDEN_SIZE"] 
# Ensure `carry_state` is properly initialised
print(f"batch_size type: {type(batch_size)}, value: {batch_size}")
print(f"hidden_size type: {type(hidden_size)}, value: {hidden_size}")

carry_state = ScannedRNN.initialize_carry(batch_size, hidden_size)  # Set correct hidden size

### **JIT-Compile Model Inference**
@jax.jit
def policy_inference(observation, hidden_state):
    """ Run trained policy on an observation. """
    action, hidden_state = network.apply(network_params, hidden_state, observation)
    return action, hidden_state

### **Function: LIDAR Callback & Movement Controller**
def lidar_callback(data):
    rospy.loginfo("Received LIDAR data!")
    global carry_state, robot_x, robot_y, robot_yaw, goal_x, goal_y

    # Compute Euclidean distance to goal
    distance_to_goal = math.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
    rospy.loginfo(f"Distance to goal: {distance_to_goal:.2f}m")

    # If the robot is close to the goal, stop
    if distance_to_goal < goal_tolerance:
        rospy.loginfo("Goal reached! Stopping the robot.")
        stop_robot()
        return

    # Convert LIDAR scan data into JAX tensor
    input_data = jnp.array(data.ranges, dtype=jnp.float32).reshape(1, -1)
    
    rospy.loginfo(f"LIDAR input shape: {input_data.shape}")

    # Ensure input is correctly formatted
    dones = jnp.zeros((1,), dtype=jnp.bool_)

    # Run trained policy inference
    action, carry_state = policy_inference((input_data, dones), carry_state)

    # Extract action outputs (linear & angular velocity)
    model_linear_velocity = float(action[0])
    model_angular_velocity = float(action[1])

    rospy.loginfo(f"Action output - Linear: {model_linear_velocity}, Angular: {model_angular_velocity}")

    # Publish velocity commands
    cmd = Twist()
    cmd.linear.x = linear_velocity
    cmd.angular.z = angular_velocity
    pub.publish(cmd)

    rospy.loginfo("Published velocity command!")


### **Function: Stop Robot**
def stop_robot():
    """ Stop the robot. """
    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    pub.publish(cmd)

### **ROS Subscribers**
rospy.Subscriber("/scan", LaserScan, lidar_callback)  # LIDAR sensor data
rospy.Subscriber("/odom", Odometry, odometry_callback)  # Robot position tracking

### **ROS Spin (Keep Node Running)**
rospy.spin()







#!/usr/bin/env python

import rospy
import pickle
import jax
import jax.numpy as jnp
import wandb
import numpy as np
import tqdm
import os
import math
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Import model utilities
from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.runners.eval_runner import EvalSampledRunner
from jaxmarl.environments.jaxnav import JaxNav, EnvInstance

### **Global Variables**
robot_x, robot_y, robot_yaw = -2.5, 2.5, 0.0  # Initial position (Modify as needed)
goal_x, goal_y = -2.5, 9.5  # Goal position
goal_tolerance = 0.2  # Stop when within 20 cm of the goal
carry_state = None

# Initialize ROS node
rospy.init_node("robot_controller", anonymous=True)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

### ** Function: Odometry Callback**
def odometry_callback(msg):
    """ Update the robot's position from odometry data. """
    global robot_x, robot_y, robot_yaw
    robot_x = msg.pose.pose.position.x
    robot_y = msg.pose.pose.position.y

    # Convert quaternion to Euler angles (yaw, pitch, roll)
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (_, _, robot_yaw) = euler_from_quaternion(orientation_list)

    rospy.loginfo(f"Robot position: x={robot_x:.2f}, y={robot_y:.2f}, yaw={robot_yaw:.2f}")

### ** Function: Load Trained Model from W&B**
def load_trained_model(run_name):
    """ Load the trained model from Weights & Biases (W&B). """
    api = wandb.Api()
    run = api.run(run_name)
    config = run.config

    rospy.loginfo(f"Loading trained model from W&B: {run_name}")

    # Load JAXNav environment
    env = JaxNav(num_agents=config["env"]["num_agents"], **config["env"]["env_params"])

    config["learning"]["LOG_DORMANCY"] = True
    config["learning"]["USE_LAYER_NORM"] = False

    # Load model parameters
    model_artifact = api.artifact(f"{run.entity}/{run.project}/{run.name}-checkpoint:latest")
    artifact_dir = model_artifact.download()
    network_params = load_params(artifact_dir + "/model.safetensors")

    # Initialize neural network model
    network = ActorCriticRNN(
        action_dim=env.action_space(agent="agent_0").shape[0], 
        config=config["learning"],
    )
    
    rng = jax.random.PRNGKey(10)
    rng, _rng = jax.random.split(rng)

    # Initialize evaluation runner
    runner = EvalSampledRunner(
        _rng,
        env,
        network,
        ScannedRNN.initialize_carry,
        hidden_size=config["learning"]["HIDDEN_SIZE"],
        greedy=False,
        env_init_states=None,
        n_episodes=1,
    )

    return network, network_params, env, config


### ** Function: Visualize Environment**
def visualize_env(env: JaxNav, env_instances: EnvInstance):
    """ Visualize the robot's environment. """
    num_envs = env_instances.map_data.shape[0]
    _, states = jax.vmap(env.set_env_instance, in_axes=(0))(env_instances)
    fig, ax = plt.subplots()
    images = []
    
    for i in tqdm.tqdm(range(num_envs), desc="Rendering maps"):
        state = jax.tree_map(lambda x: x[i], states)
        ax.clear()
        env.init_render(ax, state, lidar=True)
        fig.canvas.draw()
        
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        images.append(wandb.Image(image))
    
    table = wandb.Table(columns=["id", "maps"], data=list(zip(range(num_envs), images)))
    wandb.log({"environment_maps": table})

### ** Load the Model (Ensure the Run ID is Correct)**
run_id = "evelinawg-university-of-oxford/jaxnav-barn/sx0ptcp5"
network, network_params, env, config = load_trained_model(run_id)

# Initialize RNN hidden state
rng = jax.random.PRNGKey(10)
hidden_size = config["learning"]["HIDDEN_SIZE"]
batch_size = 1
carry_state = ScannedRNN.initialize_carry(, hidden_size)


# JIT-compile model inference for real-time execution
@jax.jit
def policy_inference(observation, hidden_state):
    """ Run trained policy on an observation. """
    action, hidden_state = network.apply(network_params, hidden_state, observation)
    return action, hidden_state

### ** Function: LIDAR Callback & Movement Controller**
def lidar_callback(data):
    """ Process LIDAR sensor data and publish velocity commands. """
    global carry_state, robot_x, robot_y, robot_yaw, goal_x, goal_y

    # Compute Euclidean distance to goal
    distance_to_goal = math.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
    rospy.loginfo(f"Distance to goal: {distance_to_goal:.2f}m")

    # If the robot is close to the goal, stop
    if distance_to_goal < goal_tolerance:
        rospy.loginfo("Goal reached! Stopping the robot.")
        stop_robot()
        return

    # Convert LIDAR scan data into JAX tensor
    input_data = jnp.array(data.ranges, dtype=jnp.float32).reshape(1, -1)

    # Run trained policy inference
    action, carry_state = policy_inference(input_data, carry_state)

    # Extract action outputs (linear & angular velocity)
    model_linear_velocity = float(action[0])
    model_angular_velocity = float(action[1])

    # Compute angle to goal
    angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)

    # Compute heading error (difference between robot orientation & goal direction)
    heading_error = angle_to_goal - robot_yaw
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # Normalize to [-pi, pi]

    # Proportional control correction for turning toward the goal
    angular_correction = 1.0 * heading_error  # Adjust gain as needed

    # Combine model output with goal-based correction
    linear_velocity = min(0.5, model_linear_velocity)  # Limit max speed
    angular_velocity = model_angular_velocity + angular_correction

    # Publish velocity commands
    cmd = Twist()
    cmd.linear.x = linear_velocity
    cmd.angular.z = angular_velocity
    pub.publish(cmd)

    # Log data to W&B
    wandb.log({"linear_velocity": linear_velocity, "angular_velocity": angular_velocity})

### ** Function: Stop Robot**
def stop_robot():
    """ Stop the robot. """
    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    pub.publish(cmd)
    rospy.loginfo("Robot Stopped")

### ** ROS Subscribers**
rospy.Subscriber("/scan", LaserScan, lidar_callback)  # LIDAR sensor data
rospy.Subscriber("/odom", Odometry, odometry_callback)  # Robot position tracking

### ** ROS Spin (Keep Node Running)**
rospy.spin()



