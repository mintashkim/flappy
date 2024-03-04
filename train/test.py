import numpy as np
import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v3')
from envs.flappy_env import FlappyEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ppo.ppo import PPO # Customized
from stable_baselines3.common.evaluation import evaluate_policy
from envs.plot import plot


env = FlappyEnv()
env = VecMonitor(DummyVecEnv([lambda: env]))
save_path = os.path.join('saved_models_0')
loaded_model = PPO.load(save_path+"/best_model")

print("Evaluation start!")
evaluate_policy(loaded_model, env, n_eval_episodes=5, render=False)
env.close()

# tensorboard --logdir logs/PPO_

def test(self, model):
    for _ in range(5):
        obs = self.reset()
        # self.debug = True
        self.max_timesteps = 0.5 * self.max_timesteps
        log = {
            "t": np.empty((0,1)),
            "x": np.empty((0,6)),
            "xd": self.goal,
            "u": np.empty((0,3)),
        }
        t = 0
        log["x"] = np.append(log["x"], np.array([np.concatenate((self.estimator.pos(), self.estimator.vel()))]), axis=0)
        log["t"] = np.append(log["t"], t)
        total_reward = 0
        for _ in range(5000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.step(action)
            total_reward += reward
            t = t + self.dt
            log["x"] = np.append(log["x"], np.array([np.concatenate((self.estimator.pos(), self.estimator.vel()))]), axis=0)
            log["t"] = np.append(log["t"], t)
            log["u"] = np.append(log["u"], np.array([self.last_act]), axis=0)
            if terminated:
                print(f"Total reward: {total_reward}")
                total_reward = 0
                self.plot(log)
                obs = self.reset()
                log = {
                    "t": np.empty((0,1)),
                    "x": np.empty((0,6)),
                    "xd": self.goal,
                    "u": np.empty((0,3)),
                }
                t = 0
                log["x"] = np.append(log["x"], np.array([np.concatenate((self.estimator.pos(), self.estimator.vel()))]), axis=0)
                log["t"] = np.append(log["t"], t)
        self.debug = False
        print("Test completed")
        self.plot(log)
    return log

test(loaded_model)