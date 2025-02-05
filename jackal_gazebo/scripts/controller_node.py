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

carry_state = ScannedRNN.initialize_carry(batch_size, hidden_size) # Set correct hidden size


def policy_inference(observation, hidden_state):
    """ Run trained policy on an observation. """

    # Ensure `observation` is a JAX array (convert if needed)
    if isinstance(observation, tuple):
        rospy.loginfo("DEBUG: `observation` is a tuple. Extracting first element.")

    obs, dones = observation  # Unpack tuple
     # Ensure they are JAX arrays
    obs = jnp.array(obs, dtype=jnp.float32)
    dones = jnp.array(dones, dtype=jnp.bool_)
    

    # Debugging logs
    rospy.loginfo(f"Policy Inference - Obs shape: {obs.shape}, Dones shape: {dones.shape}, Hidden state shape: {hidden_state.shape}")

    # Log type information
    rospy.loginfo(f"DEBUG: network type: {type(network)}")
    rospy.loginfo(f"DEBUG: network params type: {type(network_params)}")
    rospy.loginfo(f"DEBUG: network params keys: {network_params.keys() if isinstance(network_params, dict) else 'Unknown'}")
    rospy.loginfo(f"DEBUG: hidden_state type: {type(hidden_state)}")
    rospy.loginfo(f"DEBUG: observation type: {type(observation)}")


    try:
        rospy.loginfo("Running network.apply() for policy inference...")
        # action, hidden_state = network.apply(network_params, hidden_state, (obs, dones))
        
        # Ensure obs is at least 2D (1, feature_size)
        if obs.ndim == 1:
            rospy.logwarn("WARNING: obs is 1D. Reshaping to (1, -1).")
            obs = obs.reshape(1, -1)
        elif obs.ndim == 0:
            rospy.logwarn("WARNING: obs is scalar. Reshaping to (1, 1).")
            obs = obs.reshape(1, 1)

        result = network.apply(hidden_state, observation)
        rospy.loginfo(f"DEBUG: network.apply() returned: {type(result)}")

        # Sample an action from the policy
        action = pi.sample(seed=jax.random.PRNGKey(0))

        rospy.loginfo(f"Policy Inference - Action: {action}")

        return action, hidden_state
    except Exception as e:
        rospy.logerr(f"Policy inference failed: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())  # Print full traceback
        return None, hidden_state  # Handle failure

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

    input_data = np.array(data.ranges, dtype=np.float32) 

    # Convert LIDAR scan data into JAX tensor
    rospy.loginfo(f"LIDAR input shape: {input_data.shape}")

    # expected_size = 205  # Model expects 205 features

    # if input_data.ndim > 1:  # Ensure input is 1D
    #     input_data = input_data.flatten()

    # if input_data.shape[0] != expected_size:
    #     input_data = np.interp(
    #         np.linspace(0, 1, expected_size),  # New target size (205)
    #         np.linspace(0, 1, len(input_data)),  # Original size (720)
    #         input_data  # Original LIDAR data
    #     )

    input_data = jnp.array(input_data).reshape(1, -1)

    # Ensure input is correctly formatted
    dones = jnp.zeros((1,), dtype=jnp.bool_)

    # Run trained policy inference
    rospy.loginfo("DEBUG: Calling policy_inference()...")
    try:
        action, carry_state = policy_inference((input_data, dones), carry_state)
    except Exception as e:
        rospy.logerr(f"Policy inference failed: {e}")
        return

    # Ensure policy inference was successful
    if action is None:
        rospy.logwarn("Policy inference returned None. Skipping this cycle.")
        return

    # Extract action outputs (linear & angular velocity)
    try:
        model_linear_velocity = float(action[0])
        model_angular_velocity = float(action[1])
    except (IndexError, TypeError) as e:
        rospy.logerr(f"Invalid action output: {e}")
        return

    rospy.loginfo(f"DEBUG: Action output - Linear: {model_linear_velocity}, Angular: {model_angular_velocity}")

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
rospy.Subscriber("/front/scan", LaserScan, lidar_callback)  # LIDAR sensor data
rospy.Subscriber("/odom", Odometry, odometry_callback)  # Robot position tracking

### **ROS Spin (Keep Node Running)**
rospy.spin()





