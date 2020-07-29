# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:20:44 2020

@author: LOMO
"""
from matplotlib import pyplot as plt 
from PIL import Image
import time
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

from flatland.utils.rendertools import RenderTool
#from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

base_dir = Path('C:\\Users\\LOMO\\Desktop\\实习\\多智能体\\环境\\flatland-examples-master\\flatland-examples-master\\reinforcement_learning')\
    .resolve().parent
sys.path.append(str(base_dir))

from utils.observation_utils import normalize_observation
from reinforcement_learning.timer import Timer
from reinforcement_learning.dddqn_policy import DDDQNPolicy

a,b,c = [],[],[]
env_params = {
    # small_v0 config
    "n_agents": 2,
    "x_dim": 20,
    "y_dim": 20,
    "n_cities": 4,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,

    "seed": 42,
    "observation_tree_depth": 3,
    "observation_radius": 10,
    "observation_max_path_depth": 30
}

parser = ArgumentParser()
parser.add_argument("-n", "--n_episodes", dest="n_episodes", help="number of episodes to run", default=200, type=int)
parser.add_argument("--n_evaluation_episodes", dest="n_evaluation_episodes", help="number of evaluation episodes", default=100, type=int)
parser.add_argument("--checkpoint_interval", dest="checkpoint_interval", help="checkpoint interval", default=10, type=int)
parser.add_argument("--eps_start", dest="eps_start", help="max exploration", default=1.0, type=float)
parser.add_argument("--eps_end", dest="eps_end", help="min exploration", default=0.01, type=float)
parser.add_argument("--eps_decay", dest="eps_decay", help="exploration decay", default=0.99, type=float)
parser.add_argument("--buffer_size", dest="buffer_size", help="replay buffer size", default=int(1e6), type=int)
parser.add_argument("--buffer_min_size", dest="buffer_min_size", help="min buffer size to start training", default=0, type=int)
parser.add_argument("--batch_size", dest="batch_size", help="minibatch size", default=32, type=int)
parser.add_argument("--gamma", dest="gamma", help="discount factor", default=0.99, type=float)
parser.add_argument("--tau", dest="tau", help="soft update of target parameters", default=1e-3, type=float)
parser.add_argument("--learning_rate", dest="learning_rate", help="learning rate", default=0.52e-4, type=float)
parser.add_argument("--hidden_size", dest="hidden_size", help="hidden size (2 fc layers)", default=256, type=int)
parser.add_argument("--update_every", dest="update_every", help="how often to update the network", default=8, type=int)
parser.add_argument("--use_gpu", dest="use_gpu", help="use GPU if available", default=False, type=bool)
parser.add_argument("--num_threads", dest="num_threads", help="number of threads to use", default=1, type=int)
parser.add_argument("--render", dest="render", help="render 1 episode in 100", default=False, type=bool)
train_params = parser.parse_args()



n_agents = 5
x_dim = 35
y_dim = 35
n_cities = 4
max_rails_between_cities = 2
max_rails_in_city = 3
seed = 47

# Observation parameters
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

# Training parameters
eps_start = train_params.eps_start
eps_end = train_params.eps_end
eps_decay = train_params.eps_decay
n_episodes = train_params.n_episodes
checkpoint_interval = train_params.checkpoint_interval
n_eval_episodes = train_params.n_evaluation_episodes

# Set the seeds
random.seed(seed)
np.random.seed(seed)

# Break agents from time to time
malfunction_parameters = MalfunctionParameters(
    malfunction_rate=1. / 10000,  # Rate of malfunctions
    min_duration=15,  # Minimal duration
    max_duration=50  # Max duration
)

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

# Fraction of train which each speed
speed_profiles = {
    1.: 1.0,  # Fast passenger train
    1. / 2.: 0.0,  # Fast freight train
    1. / 3.: 0.0,  # Slow commuter train
    1. / 4.: 0.0  # Slow freight train
}

# Setup the environment
env = RailEnv(
    width=x_dim,
    height=y_dim,
    rail_generator=sparse_rail_generator(
        max_num_cities=n_cities,
        grid_mode=False,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_city
    ),
    schedule_generator=sparse_schedule_generator(speed_profiles),
    number_of_agents=n_agents,
    malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
    obs_builder_object=tree_observation,
    random_seed=seed
)

env.reset(regenerate_schedule=True, regenerate_rail=True)

# Setup renderer

env_renderer = RenderTool(env)
'''
env_renderer.render_env(show=True,show_predictions=False)
time.sleep(5)
env_renderer.close_window()
'''
n_features_per_node = env.obs_builder.observation_dim
n_nodes = 0
for i in range(observation_tree_depth + 1):
    n_nodes += np.power(4, i)
state_size = n_features_per_node * n_nodes

action_size = 5

# Max number of steps per episode
# This is the official formula used during evaluations
# See details in flatland.envs.schedule_generators.sparse_schedule_generator
max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))

action_count = [0] * action_size
action_dict = dict()
agent_obs = [None] * env.get_num_agents()
agent_prev_obs = [None] * env.get_num_agents()
agent_prev_action = [2] * env.get_num_agents()
update_values = False
smoothed_normalized_score = -1.0
smoothed_eval_normalized_score = -1.0
smoothed_completion = 0.0
smoothed_eval_completion = 0.0



policy = DDDQNPolicy(state_size, action_size, train_params)

def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer

def eval_policy(env, policy, n_eval_episodes, max_steps):
    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        final_step = 0

        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    # TODO pass parameters properly
                    # agent_obs[agent] = normalize_observation(obs[agent], tree_depth=2, observation_radius=10)
                    agent_obs[agent] = normalize_observation(obs[agent], tree_depth=2, observation_radius=10)

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps

for episode_idx in range(n_episodes + 1):
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    score = 0
    nb_steps = 0
    actions_taken = []

    # Build agent specific observations
    for agent in env.get_agent_handles():
        if obs[agent]:
            agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
            agent_prev_obs[agent] = agent_obs[agent].copy()

    # Run episode
    for step in range(max_steps - 1):
        for agent in env.get_agent_handles():
            if info['action_required'][agent]:
                # If an action is required, we want to store the obs at that step as well as the action
                update_values = True
                action = policy.act(agent_obs[agent], eps=eps_start)
                action_count[action] += 1
                actions_taken.append(action)
            else:
                update_values = False
                action = 0
            action_dict.update({agent: action})

        # Environment step
        next_obs, all_rewards, done, info = env.step(action_dict)

        #if train_params.render and episode_idx % checkpoint_interval == 0:
        '''
        env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=True,
                show_predictions=False
            )'''

        for agent in range(env.get_num_agents()):
            # Update replay buffer and train agent
            # Only update the values when we are done or when an action was taken and thus relevant information is present
            if update_values or done[agent]:
                policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                agent_prev_obs[agent] = agent_obs[agent].copy()
                agent_prev_action[agent] = action_dict[agent]

            # Preprocess the new observations
            if next_obs[agent]:
                agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
            score += all_rewards[agent]

        nb_steps = step

        if done['__all__']:
            break
    '''
    env_renderer.close_window()
    env_renderer = RenderTool(env)
    '''
    # Epsilon decay
    eps_start = max(eps_end, eps_decay * eps_start)

    # Collection information about training
    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
    completion = tasks_finished / max(1, env.get_num_agents())
    normalized_score = score / (max_steps * env.get_num_agents())
    action_probs = action_count / np.sum(action_count)
    action_count = [1] * action_size

    # Smoothed values for terminal display and for more stable hyper-parameter tuning
    smoothing = 0.99
    smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
    smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

    # Print logs
    if episode_idx % checkpoint_interval == 0:
        torch.save(policy.qnetwork_local, './checkpoints/testmulti-' + str(episode_idx) + '.pth')
        if train_params.render:
            env_renderer.close_window()
    a.append(normalized_score)  
    b.append(completion)          
    print(
        '\r🚂 Episode {}'
        '\t 🏆 Score: {:.3f}'
        ' Avg: {:.3f}'
        '\t 💯 Done: {:.2f}%'
        ' Avg: {:.2f}%'
        '\t 🎲 Epsilon: {:.2f} '
        '\t 🔀 Action Probs: {}'.format(
            episode_idx,
            normalized_score,
            smoothed_normalized_score,
            100 * completion,
            100 * smoothed_completion,
            eps_start,
            format_action_prob(action_probs)
        ), end=" ")

    # Evaluate policy
    '''
    if episode_idx % train_params.checkpoint_interval == 100:
        scores, completions, nb_steps_eval = eval_policy(env, policy, n_eval_episodes, max_steps)
        smoothing = 0.9
        smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
        smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
'''
