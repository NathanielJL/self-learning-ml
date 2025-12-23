#Imports libraries
import gymnasium as gym
from stable_baselines3 import PPO
import time

# Create the environment with render_mode='human'
env = gym.make('CartPole-v1', render_mode='human')

# Create the RL model (PPO)
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model (optional)
for i in range(5):  # Run more episodes for a smoother plot
    obs, info = env.reset()
    done = False
    score = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        time.sleep(0.01)
    print(f"Episode {i+1} score: {score}")

env.close()
