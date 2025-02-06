import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import deque

# Define the Q-Network
def create_q_network(state_size, action_size):
    model = models.Sequential([
        layers.Dense(64, input_dim=state_size, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(action_size, activation="linear")  # Output Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

# Environment simulation (Instant Ink subscription context)
class InstantInkEnvironment:
    def __init__(self):
        self.state = [50, 0]  # [remaining pages, current subscription tier]
        self.tiers = [50, 100, 300]  # Example tiers
        self.page_cost = 0.1  # Per-page overage cost
        self.max_pages = 300
    
    def reset(self):
        self.state = [50, 0]  # Reset to default tier and pages
        return np.array(self.state, dtype=np.float32)
    
    def step(self, action):
        """
        Actions:
        0 - Do nothing
        1 - Upgrade subscription
        2 - Downgrade subscription
        """
        pages_left, tier = self.state
        
        if action == 1 and tier < len(self.tiers) - 1:
            tier += 1
        elif action == 2 and tier > 0:
            tier -= 1
        
        pages_left -= random.randint(0, 20)  # Random usage simulation
        reward = 0
        
        # Calculate reward: maximize savings or minimize overage
        if pages_left < 0:
            reward = -abs(pages_left) * self.page_cost  # Penalize for overage
        elif pages_left > 0 and pages_left < self.tiers[tier] * 0.2:
            reward = 5  # Bonus for efficient usage of subscription
        
        # End episode if too many pages are used
        done = pages_left < -self.max_pages
        self.state = [max(0, pages_left), tier]
        return np.array(self.state, dtype=np.float32), reward, done

# Parameters
state_size = 2  # [remaining pages, current subscription tier]
action_size = 3  # Actions: [Do nothing, Upgrade, Downgrade]
batch_size = 32
episodes = 500
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
memory = deque(maxlen=2000)

# Initialize Q-network and environment
env = InstantInkEnvironment()
q_network = create_q_network(state_size, action_size)
target_network = create_q_network(state_size, action_size)
target_network.set_weights(q_network.get_weights())

# Training loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, action_size)
        else:
            q_values = q_network.predict(state[np.newaxis], verbose=0)
            action = np.argmax(q_values[0])
        
        # Take action in the environment
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Store transition in memory
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Train the Q-network with a batch of experiences
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, targets = [], []

            for s, a, r, ns, d in batch:
                target = r
                if not d:
                    target += gamma * np.amax(target_network.predict(ns[np.newaxis], verbose=0)[0])
                target_q_values = q_network.predict(s[np.newaxis], verbose=0)[0]
                target_q_values[a] = target
                states.append(s)
                targets.append(target_q_values)

            q_network.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    # Update target network weights
    if episode % 10 == 0:
        target_network.set_weights(q_network.get_weights())
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Save the trained model
q_network.save("instantink_q_network.h5")
