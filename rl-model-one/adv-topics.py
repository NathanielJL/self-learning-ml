# Advanced Topics in Reinforcement Learning: Guide and Examples

"""
This document introduces advanced RL topics, foundational concepts, and the main types of RL models, with code examples and explanations.
"""

# 1. RL Foundations: Types of Learning
# ------------------------------------
# - Value-based: Learn value functions (e.g., Q-learning, DQN)
# - Policy-based: Learn policies directly (e.g., REINFORCE, PPO, A2C)
# - Actor-Critic: Combine value and policy learning (e.g., A2C, PPO, SAC)

# 2. Value-Based Example: DQN
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=5000)

# 3. Policy-Based Example: PPO
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=5000)

# 4. Actor-Critic Example: A2C
from stable_baselines3 import A2C

env = gym.make('CartPole-v1')
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=5000)

# 5. Continuous Action Example: SAC
from stable_baselines3 import SAC

env = gym.make('Pendulum-v1')  # Continuous action space
action_space = env.action_space
model = SAC('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=5000)

# 6. Multi-Agent RL Example (using PettingZoo)
# Install: pip install pettingzoo[all] stable-baselines3[extra]
# This is a minimal example for a multi-agent environment.
# Note: Stable-Baselines3 does not natively support multi-agent, but you can train one agent at a time.
# See PettingZoo docs for more advanced usage.
"""
from pettingzoo.classic import tictactoe_v3
import numpy as np

env = tictactoe_v3.env()
env.reset()
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    action = env.action_space(agent).sample() if not termination else None
    env.step(action)
"""

# 7. Custom Reward Shaping Example
# Inherit from your environment and override step() to change reward logic.
class CustomRewardEnv(SimpleCounterEnv):
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        # Give extra reward for being close to the target
        if not terminated:
            reward += 0.5 * (1 - abs(self.state[0] - self.target) / 10)
        return state, reward, terminated, truncated, info

# 8. Curriculum Learning Example
# Vary environment parameters during training to increase difficulty.
import random
class CurriculumEnv(SimpleCounterEnv):
    def reset(self, seed=None, options=None):
        # Gradually increase the target as training progresses
        self.target = random.randint(5, 10)
        return super().reset(seed=seed, options=options)

# 9. Transfer Learning Example
from stable_baselines3 import PPO
import gymnasium as gym
# Load a pre-trained model and fine-tune on a new environment
model = PPO.load('ppo_cartpole')
model.set_env(gym.make('CartPole-v1'))
model.learn(total_timesteps=1000)  # Fine-tune

# 10. Monitoring and Visualization Example
# Use TensorBoard for live training stats
model = PPO('MlpPolicy', gym.make('CartPole-v1'), verbose=1, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=10000)
# Launch TensorBoard in terminal: tensorboard --logdir=./ppo_tensorboard/

# 11. Further Reading
# - Sutton & Barto, "Reinforcement Learning: An Introduction"
# - Spinning Up in Deep RL (OpenAI): https://spinningup.openai.com/
# - Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
# - Gymnasium docs: https://gymnasium.farama.org/
# - PettingZoo docs: https://pettingzoo.farama.org/

# --- RL Workflow Guide and Examples ---
# This section provides a step-by-step guide for common RL tasks with code examples.

import gymnasium as gym
from stable_baselines3 import PPO
import time

# 1. Create the environment with rendering
# 'render_mode="human"' opens a window to visualize the agent
env = gym.make('CartPole-v1', render_mode='human')

# 2. Create the RL model (PPO algorithm)
# 'MlpPolicy' is a neural network for policy and value function
model = PPO('MlpPolicy', env, verbose=1)

# 3. Train the model
# 'total_timesteps' controls how long the agent learns
model.learn(total_timesteps=10000)

# 4. Save the trained model to disk
model.save("ppo_cartpole")

# 5. Load a saved model (optional)
# model = PPO.load("ppo_cartpole", env=env)

# 6. Evaluate the trained agent
# Run several episodes and print the score for each
for i in range(5):
    obs, info = env.reset()
    done = False
    score = 0
    while not done:
        # Use the trained model to select an action
        action, _states = model.predict(obs, deterministic=True)
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        time.sleep(0.02)  # Slow down rendering for visibility
    print(f"Episode {i+1} score: {score}")

# 7. Close the environment
env.close()

# --- More RL Topics and Examples ---

# Saving and loading models
# model.save("my_model")
# model = PPO.load("my_model", env=env)

# Changing hyperparameters
# model = PPO('MlpPolicy', env, learning_rate=0.0003, n_steps=2048, batch_size=64)

# Using callbacks (e.g., for early stopping or saving best model)
"""
from stable_baselines3.common.callbacks import CheckpointCallback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_checkpoint')
model.learn(total_timesteps=10000, callback=checkpoint_callback)
"""

# Using custom environments (see two.py for an example)
# env = SimpleCounterEnv(target=7)

# Using vectorized environments for faster training
"""
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([lambda: gym.make('CartPole-v1') for _ in range(4)])
model = PPO('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=10000)
"""

# Using evaluation helpers
"""
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std: {std_reward}")
"""

# Using TensorBoard for monitoring
# model.learn(total_timesteps=10000, tensorboard_log="./ppo_tensorboard/")
# Launch TensorBoard: tensorboard --logdir=./ppo_tensorboard/

# For more, see the Stable-Baselines3 documentation and examples.
