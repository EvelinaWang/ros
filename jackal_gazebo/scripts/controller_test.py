#!/usr/bin/env python
import rospy
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from jaxmarl.environments.jaxnav import JaxNav


from geometry_msgs.msg import PoseStamped

def send_goal(x, y):
    """Send a goal position to the move_base navigation system."""
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

    rospy.sleep(1)  # Give time for the publisher to connect

    goal = PoseStamped()
    goal.header.frame_id = "map"  # Change to "odom" if needed
    goal.header.stamp = rospy.Time.now()

    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0

    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0
    goal.pose.orientation.w = 1.0  # Facing forward

    pub.publish(goal)
    rospy.loginfo(f"🚀 Goal sent to: ({x}, {y})")

# Load Trained Model from WandB
def load_wandb_model():
    rospy.loginfo("Loading trained model from WandB...")
    run_id = "evelinawg-university-of-oxford/jaxnav-barn/sx0ptcp5"
    api = wandb.Api()
    run = api.run(run_id)
    config = run.config

    # Load Environment
    env = JaxNav(num_agents=1, do_sep_reward=False, **config["env"]["env_params"])
    
    # Load model artifact
    model_artifact = api.artifact(f"{run.entity}/{run.project}/{run.name}-checkpoint:latest")
    model_path = model_artifact.download()
    
    # Load model parameters
    network = ActorCriticRNN(
        action_dim=env.action_space(agent='agent_0').shape[0], 
        config=config["learning"],
    )
    network_params = load_params(model_path + "/model.safetensors")

    # Check if any parameters contain NaN
    for key, param in network_params.items():
        if isinstance(param, jnp.ndarray) and jnp.any(jnp.isnan(param)):
            rospy.logerr(f"🚨 NaN detected in model parameter: {key} (model weights may be corrupted)")
            return None, None, None  # If weights are corrupt, return None

    return network, network_params, config

# Policy Inference Function
def policy_inference(observation, hidden_state):
    """Run trained policy on an observation."""
    
    obs, dones = observation  # Unpack tuple
    obs = jnp.array(obs, dtype=jnp.float32).reshape(1, 1, -1)
    dones = jnp.array(dones, dtype=jnp.bool_).reshape(1, 1,)

    rospy.loginfo(f"Policy Inference - Obs shape: {obs.shape}, Dones shape: {dones.shape}, Hidden state shape: {hidden_state.shape}")
    
    # Check for NaNs in observation before feeding to network
    if jnp.any(jnp.isnan(obs)):
        rospy.logerr("NaN detected in observation!")
        return None, hidden_state

    # Check for NaNs in hidden state before inference
    if jnp.any(jnp.isnan(hidden_state)):
        rospy.logerr("NaN detected in hidden state before inference!")
        hidden_state = ScannedRNN.initialize_carry(1, hidden_state.shape[-1])  # Reset hidden state
        return None, hidden_state

    try:
        hidden_state, pi, _, _ = network.apply(network_params, hidden_state, (obs, dones))
        rospy.loginfo(f"Policy Inference - pi type: {type(pi)}, pi contents: {pi}")

        # Extract policy parameters (mean, std)
        pi_mean = pi.mean()
        pi_std = pi.stddev()

        rospy.loginfo(f"Policy Parameters - Mean: {pi_mean}, Stddev: {pi_std}")

        if jnp.any(jnp.isnan(pi_mean)) or jnp.any(jnp.isnan(pi_std)):
            rospy.logerr("NaN detected in policy distribution parameters!")
            return None, hidden_state
        
        action = pi.sample(seed=jax.random.PRNGKey(0))  # Sample action from policy
        rospy.loginfo(f"Policy Inference - Action: {action}")
        return action, hidden_state
    except Exception as e:
        rospy.logerr(f"Policy inference failed: {e}")
        return None, hidden_state

# LIDAR Callback Function
def lidar_callback(data):
    """Process LIDAR scan and control the robot."""
    rospy.loginfo("Received LIDAR data!")

    global carry_state

    # rospy.loginfo(f"Published command: hiddden_state={carry_state}")

    # Preprocess LIDAR data
    input_data = np.array(data.ranges, dtype=np.float32)

    # ✅ Replace NaN and Inf values to prevent errors
    input_data = np.nan_to_num(input_data, nan=0.0, posinf=6.0, neginf=0.0) #posinf value equals to the max_lidar_range=6

    # ✅ Rescale LIDAR data to match expected model input size (205)
    expected_size = 205  
    input_data = np.interp(
        np.linspace(0, 1, expected_size),  # New size: 205
        np.linspace(0, 1, len(input_data)),  # Original size: 720
        input_data  
    )
    # input_data = jnp.array(input_data).reshape(1, -1)
    dones = jnp.zeros((1,), dtype=jnp.bool_)

    # rospy.loginfo(f"Published command: input_data={input_data}")

    # Get action from policy
    action, carry_state = policy_inference((input_data, dones), carry_state)
    
    if action is None:
        rospy.logwarn("Policy inference returned None. Skipping this cycle.")
        return

    # Convert action to velocity commands
    try:
        model_linear_velocity = float(action[0][0][0])  # Forward movement
        model_angular_velocity = float(action[0][0][1])  # Rotation

            # ✅ Scale down velocities to slow down the robot
        LINEAR_SPEED_SCALING = 0.3  # Reduce linear speed (e.g., 30% of original)
        ANGULAR_SPEED_SCALING = 0.5  # Reduce angular speed (e.g., 50% of original)

        model_linear_velocity *= LINEAR_SPEED_SCALING
        # model_angular_velocity *= ANGULAR_SPEED_SCALING

    except (IndexError, TypeError) as e:
        rospy.logerr(f"Invalid action output: {e}")
        return

    # Publish velocity commands
    cmd = Twist()
    cmd.linear.x = model_linear_velocity
    cmd.angular.z = model_angular_velocity
    pub.publish(cmd)

    rospy.loginfo(f"Published command: Linear={model_linear_velocity}, Angular={model_angular_velocity}")
    



# Initialize ROS Node & Variables
rospy.init_node("controller_test", anonymous=True)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
network, network_params, config = load_wandb_model()

# Set goal position for navigation
goal_x, goal_y = -2.5, 9.5  # Change to desired goal position
send_goal(goal_x, goal_y)


# Initialize Carry State
batch_size = 1
hidden_size = config["learning"]["HIDDEN_SIZE"]
carry_state = ScannedRNN.initialize_carry(batch_size, hidden_size)

# Subscribe to LIDAR Data
rospy.Subscriber("/front/scan", LaserScan, lidar_callback)

# Keep Node Running
rospy.spin()



