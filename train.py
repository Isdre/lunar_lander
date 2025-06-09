import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os

# 1. Create training environment with more parallel environments
print("Creating training environment...")
vec_env = make_vec_env("LunarLander-v3", n_envs=16)  # More parallel environments for A2C
eval_env = make_vec_env("LunarLander-v3", n_envs=4)

# 2. Initialize A2C model with appropriate hyperparameters
print("Initializing A2C model...")
model = A2C("MlpPolicy", 
            vec_env, 
            learning_rate=0.0007,  # A2C typically uses higher learning rates
            n_steps=5,             # A2C uses shorter rollout lengths
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,           # Value function coefficient
            max_grad_norm=0.5,     # Gradient clipping
            rms_prop_eps=1e-5,     # RMSProp optimizer epsilon
            use_rms_prop=True,     # Use RMSProp optimizer
            normalize_advantage=True,
            policy_kwargs=dict(net_arch=[128, 128]),  # Larger network for A2C
            verbose=1)

# 3. Train model with callback for saving best model
print("Beginning A2C model training...")
# Create directory for best model
os.makedirs("./best_model", exist_ok=True)

eval_callback = EvalCallback(eval_env, 
                            best_model_save_path='./best_model/',
                            log_path='./logs/', 
                            eval_freq=5000,        # Evaluate more frequently with A2C
                            deterministic=True, 
                            render=False)

# Train for many steps - A2C often needs more steps than PPO
model.learn(total_timesteps=500000, callback=eval_callback)

# 4. Load and evaluate the best model
print("Loading and evaluating best model...")
best_model_path = "./best_model/best_model"

if os.path.exists(best_model_path + ".zip"):
    best_model = A2C.load(best_model_path, env=vec_env)
    print("Loaded best model from saved file")
else:
    best_model = model
    print("No saved model found, using current model")

mean_reward, std_reward = evaluate_policy(best_model, vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# No need for additional testing in training script
vec_env.close()
print("Training complete. Test using main.py")
