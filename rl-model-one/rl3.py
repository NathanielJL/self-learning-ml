import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make('CarRacing-v3')

model = PPO('CnnPolicy', env, verbose=1)

model.learn(total_timesteps=20000)

model.save("ppo_carracing")

for i in range(5):
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