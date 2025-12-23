import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Custom environment where the agent must reach a target number by incrementing or decrementing.
# Observation: current integer value (0 to 10)
# Actions: 0 = decrement, 1 = increment
# Reward: +1 for reaching the target, -0.1 otherwise
# Episode ends when target is reached or after 20 steps
class SimpleCounterEnv(gym.Env):
    def __init__(self, target=7):
        super().__init__()
        self.target = target  # The number the agent should reach
        # State is a single integer between 0 and 10
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        # Two possible actions: 0 (decrement), 1 (increment)
        self.action_space = spaces.Discrete(2)
        self.state = None  # Current state
        self.steps = 0  # Step counter
        self.max_steps = 20  # Max steps per episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random value between 0 and 10
        self.state = np.array([np.random.randint(0, 11)], dtype=np.int32)
        self.steps = 0
        info = {}
        return self.state, info

    def step(self, action):
        self.steps += 1  # Increment step counter
        if action == 0:
            # Decrement state, but not below 0
            self.state[0] = max(0, self.state[0] - 1)
        else:
            # Increment state, but not above 10
            self.state[0] = min(10, self.state[0] + 1)
        # Reward is +1 if target reached, else -0.1
        reward = 1.0 if self.state[0] == self.target else -0.1
        # Episode ends if target is reached
        terminated = self.state[0] == self.target
        # Or if max steps reached
        truncated = self.steps >= self.max_steps
        info = {}
        return self.state, reward, terminated, truncated, info

    def render(self):
        # Print the current state
        print(f"Current value: {self.state[0]}")

    def close(self):
        # No resources to clean up
        pass

# Example usage: run a random agent in the environment
if __name__ == "__main__":
    env = SimpleCounterEnv(target=7)  # Create environment with target 7
    obs, info = env.reset()  # Reset environment
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Take random action (0 or 1)
        obs, reward, terminated, truncated, info = env.step(action)  # Step environment
        env.render()  # Show state
        total_reward += reward  # Accumulate reward
        done = terminated or truncated  # Check if episode is over
    print(f"Episode finished. Total reward: {total_reward}")
    env.close()  # Clean up
