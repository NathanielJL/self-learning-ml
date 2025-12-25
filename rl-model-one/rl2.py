import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make('MountainCar-v0', render_mode='human')
               
model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=50000)

model.save("ppo_mountaincar")

for i in range(10):
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