"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    return np.zeros((num_states, num_actions), dtype = np.float64)

# Step 2 - max_q_value
def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # TODO: index the row for `state` and return its maximum value
    return np.max(q_table[state])

# Step 3 - greedy_action
def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # TODO: return argmax over the action axis for this state's Q values
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    # TODO: draw a uniformly random action from the given Gymnasium action space
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # TODO: draw a uniform sample from rng and compare it to epsilon
    u = rng.random()
    return (  u < epsilon )

# Step 6 - epsilon_greedy_action
def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # Check if we should explore based on the epsilon value
    if should_explore(epsilon, rng):
        # Sample a completely random action from the action space
        return sample_random_action(action_space)
    else:
        # Exploit the best action from the Q-table for the current state
        return greedy_action(q_table, state)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon, epsilon*decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if done:
        return float(reward)
    
    max_a = max_q_value(q_table, next_state)
    return float(reward + gamma * max_a)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    return target - q_table[state, action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state, action] += alpha * error
    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # Step A: Pass the full env.action_space object as requested by the description
    action = epsilon_greedy_action( q_table, state, epsilon, env.action_space,  rng)
    
    # Step B: Step the environment with the selected action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Step C: Determine if the episode is finished
    done = terminated or truncated
    
    # Step D: Apply the Q-learning update in place with the correct argument order
    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    
    # Step E: Return the tuple ensuring standard Python types
    return (int(next_state), float(reward), bool(done))

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # Step A: Reset the environment to get the initial state
    # Gymnasium's env.reset() returns a tuple: (initial_state, info)
    state, _ = env.reset()
    
    total_reward = 0.0
    
    # Step B: Loop repeatedly up to max_steps
    for _ in range(max_steps):
        # Call the interaction_step function implemented in Step 011
        state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        
        # Accumulate the scalar reward
        total_reward += reward
        
        # Break early if the episode finishes (agent falls in a hole or reaches the goal)
        if done:
            break
            
    # Step C: Return the total scalar reward as a clean float
    return float(total_reward)

# Step 13 - train_q_learning
def train_q_learning(
    env, 
    num_episodes, 
    alpha=0.8, 
    gamma=0.95, 
    epsilon_start=1.0, 
    epsilon_min=0.01, 
    epsilon_decay=0.99, 
    seed=0, 
    max_steps=200
):
    # Step A: Seed all required components for reproducibility
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    
   # Step B: Initialize a fresh Q-table using the exact arguments required by Step 001
    q_table = init_q_table(env.observation_space.n, env.action_space.n)
    episode_returns = []
    epsilon = epsilon_start
    
    # Step C: Loop through the number of episodes
    for _ in range(num_episodes):
        # Run a complete training episode using your helper from Step 012
        total_reward = run_training_episode(
            env, q_table, epsilon, alpha, gamma, rng, max_steps=max_steps
        )
        episode_returns.append(total_reward)
        
        # Step D: Decay epsilon using your helper from Step 007
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)
        
    # Return the final trained Q-table and the list of episode returns
    return q_table, episode_returns

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

