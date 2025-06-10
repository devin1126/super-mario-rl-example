import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import time
from utils import *

# Create model save directory
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

# Show CUDA status
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# Environment and training configuration
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = False
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

# Initialize and preprocess the Super Mario environment
env = gym_super_mario_bros.make(
    ENV_NAME,
    render_mode='human' if DISPLAY else 'rgb',
    apply_api_compatibility=True
)
env = JoypadSpace(env, RIGHT_ONLY)     # Limit to simple right-only actions
env = apply_wrappers(env)              # Apply frame processing, stacking, etc.

# Initialize agent
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load pretrained model if not training
if not SHOULD_TRAIN:
    folder_name = "2023-12-25-09_32_25"
    ckpt_name = "model_30000_iter.pt"
    # agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.05
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

# Test environment with a dummy action
env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

# Main training or evaluation loop
for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0

    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        total_reward += reward

        if DISPLAY:
            time.sleep(0.05)

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    print("Total reward:", total_reward,
          "Epsilon:", agent.epsilon,
          "Replay buffer size:", len(agent.replay_buffer),
          "Learn steps:", agent.learn_step_counter)

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, f"model_{i + 1}_iter.pt"))

env.close()
