import getopt
import random
import sys
import time
from collections import deque
# make sure the root path is in system path
from pathlib import Path
import time
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch

from a_training.dueling_double_dqn import Agent
from a_training.yt_env_reward import RailEnv, RailEnvActions

from torch_training.dueling_double_dqn import Agent
from torch.utils.tensorboard import SummaryWriter
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation, split_tree_into_feature_groups
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_generators import complex_rail_generator

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

def available_actions(env, agent, allow_noop=True):
    if agent.position is None:
        return [1] * len(RailEnvActions)
    else:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    # some actions are always available:
    available_acts = [0] * len(RailEnvActions)
    available_acts[RailEnvActions.MOVE_FORWARD] = 1
    available_acts[RailEnvActions.STOP_MOVING] = 1
    if allow_noop:
        available_acts[RailEnvActions.DO_NOTHING] = 1
    # check if turn left/right are available:
    for movement in range(4):
        if possible_transitions[movement]:
            if movement == (agent.direction + 1) % 4:
                available_acts[RailEnvActions.MOVE_RIGHT] = 1
            elif movement == (agent.direction - 1) % 4:
                available_acts[RailEnvActions.MOVE_LEFT] = 1
    return available_acts

def my_controller(env, obs_env, obs, number_of_agents):
    global action
    for a in range(number_of_agents):
        agent = env.agents[a]
        # if done[a]:
        #     continue
        # if agent.speed_data['position_fraction'] > 5:
        #     agent.moving = False
        #     action_dict.update({a: 4})  # stop
        #     continue
        if info['action_required'][a]:
            if agent.position is not None:
                possible_transition_num = np.count_nonzero(env.rail.get_transitions(*agent.position, agent.direction))
                if possible_transition_num == 1:
                    if obs_env[a].childs['F'].dist_unusable_switch == 1:
                        action = policy.act(obs[a], eps=eps) # go or wait? to train a different net?
                        available_action_check = available_actions(env, agent, allow_noop=True)
                        if available_action_check[action]:
                            update_values[a] = True  # only when RL make a decision
                    else:
                        action = 2
                else:
                    action = policy.act(obs[a], eps=eps)
                    available_action_check = available_actions(env, agent, allow_noop=True)
                    if available_action_check[action]:
                        update_values[a] = True  # only when RL make a decision
            else:
                action = 2
        else:
            action = 0
        action_dict.update({a: action})
    return  action_dict


start = time.time()

random.seed(1)
np.random.seed(1)

# Parameters for the Environment
# x_dim = 25
# y_dim = 25
# n_agents = 3
n_cities = 2

x_dim = 20  # np.random.randint(8, 20)
y_dim = 20  # np.random.randint(8, 20)
n_agents = 2  # np.random.randint(3, 8)
n_goals = n_agents + np.random.randint(0, 3)
min_dist = int(0.75 * min(x_dim, y_dim))


# speed_ration_map = {1.: 1.0,  # Fast passenger train
#                     1. / 2.: 0.0,  # Fast freight train
#                     1. / 3.: 0.0,  # Slow commuter train
#                     1. / 4.: 0.0}  # Slow freight train
#
# TreeObservation = TreeObsForRailEnv(max_depth=2)
# env = RailEnv(width=x_dim,
#               height=y_dim,
#               max_num_cities = n_cities,
#               rail_generator=complex_rail_generator(
#                   nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
#                   max_dist=99999,
#                   seed=0
#               ),
#               # rail_generator=sparse_rail_generator(max_num_cities=5,
#               #                                      # Number of cities in map (where train stations are)
#               #                                      seed=1,  # Random seed
#               #                                      grid_mode=False,
#               #                                      max_rails_between_cities=3,
#               #                                      max_rails_in_city=3),
#               schedule_generator=complex_schedule_generator(),
#               # schedule_generator=sparse_schedule_generator(speed_ration_map),
#               number_of_agents=n_agents,
#               # malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
#               # Malfunction data generator
#               obs_builder_object=TreeObservation)

# Use a the malfunction generator to break agents from time to time
# stochastic_data = MalfunctionParameters(malfunction_rate=1./10000,  # Rate of malfunction occurence
#                                         min_duration=15,  # Minimal duration of malfunction
#                                         max_duration=50  # Max duration of malfunction
#                                         )


# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 1.0,  # Fast passenger train
                    1. / 2.: 0.0,  # Fast freight train
                    1. / 3.: 0.0,  # Slow commuter train
                    1. / 4.: 0.0}  # Slow freight train

nAgents = 2
fnMethod = random_rail_generator(cell_type_relative_proportion=[1] * 11)
env = RailEnv(width=10,
             height=10,
             max_num_cities = 2,
             rail_generator=fnMethod,
             number_of_agents=nAgents,
             obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))

max_num_cities = 2


# Reset env
env.reset(True,True)
# After training we want to render the results so we also load a renderer
# env_renderer = RenderTool(env, gl="PILSVG", )
#env_renderer = RenderTool(env)
#env_renderer.render_env(show=True, show_predictions=False)
#time.sleep(1)
#env_renderer.close_window()
# Given the depth of the tree observation and the number of features per node we get the following state_size
num_features_per_node = env.obs_builder.observation_dim
tree_depth = 2
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes

# The action space of flatland is 5 discrete actions
action_size = 5

# We set the number of episodes we would like to train on
if 'n_trials' not in locals():
    n_trials = 20000

# And the max number of steps we want to take per episode
# max_steps = int(3 * (env.height + env.width))
max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
# Define training parameters
eps = 1.
eps_end = 0.1
eps_decay = 0.998

# And some variables to keep track of the progress
action_dict = dict()
dead_lock = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []
action_prob = [0] * action_size
agent_obs = [None] * env.get_num_agents()
agent_next_obs = [None] * env.get_num_agents()
agent_obs_buffer = [None] * env.get_num_agents()
agent_action_buffer = [2] * env.get_num_agents()
cummulated_reward = np.zeros(env.get_num_agents())
update_values = [False] * env.get_num_agents()
# Now we load a Double dueling DQN agent
policy = Agent(state_size, action_size)

# check 
writer = SummaryWriter("runs/test9_2_agents_fnMethod_10000_treed=2")


for trials in range(1, n_trials + 1):

    # Reset environment

    obs, info = env.reset(False, False, False, None)

    done = env.dones

    for a in range(env.get_num_agents()):
        if obs[a]:
            agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
            agent_obs_buffer[a] = agent_obs[a].copy()

    score = 0
    env_done = 0

    for step in range(max_steps):

        action_dict = my_controller(env, obs, agent_obs, env.get_num_agents())

        obs, all_rewards, done, info = env.step(action_dict)

        """	
        env_renderer.render_env(
            show=True,
            frames=False,
            show_observations=True,
            show_predictions=False
        )
        """

        for a in range(env.get_num_agents()):
            # Only update the values when we are done
            # or when an action was taken and thus relevant information is present

            """
            if update_values or done[a]:
                policy.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a],
                           agent_obs[a], done[a])
                cummulated_reward[a] = 0.

                agent_obs_buffer[a] = agent_obs[a].copy()
                agent_action_buffer[a] = action_dict[a]
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
            """

            agent_obs_buffer[a] = agent_obs[a].copy()
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
            if update_values or done[a]:
                policy.step(agent_obs_buffer[a], action_dict[a], all_rewards[a],
                            agent_obs[a], done[a])

            score += all_rewards[a]

        # Copy observation
        if done['__all__']:
            env_done = 1
            break
    """
    env_renderer.close_window()
    env_renderer = RenderTool(env)
    """

    # Epsilon decay
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    # Collection information about training
    tasks_finished = 0
    for _idx in range(env.get_num_agents()):
         if done[_idx] == 1:
             tasks_finished += 1

    tasks_finished = 0
    for current_agent in env.agents:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1

    done_window.append(tasks_finished / max(1, env.get_num_agents()))
    scores_window.append(score / (max_steps*env.get_num_agents()))  # save most recent score
    scores.append(np.mean(scores_window))
    dones_list.append((np.mean(done_window)))

    done_per_episode = tasks_finished / max(1, env.get_num_agents())
    done_mean = np.mean(done_window)
    scores_per_episode = score / (max_steps * env.get_num_agents())
    scores_mean = np.mean(scores_window)

    # Save logs to tensorboard
    writer.add_scalar("results/completion_per_episode", done_per_episode, trials)
    writer.add_scalar("results/completion_mean", done_mean, trials)
    writer.add_scalar("results/scores_per_episode", scores_per_episode, trials)
    writer.add_scalar("results/scores_mean", scores_mean, trials)

    print(
        '\rTraining {} Agents on ({},{}).\t Episode {}\t Score: {:.3f}\t Mean_sore: {:.3f}\t tasks_finished:{}\t Dones: {:.2f}\t Done_mean: {:.2f}\t Epsilon: {:.2f} '.format(
            env.get_num_agents(), x_dim, y_dim,
            trials,
            scores_window[-1],
            np.mean(scores_window),
            tasks_finished,
            done_window[-1],
            np.mean(done_window),
            eps), end ="")

    if trials % 100 == 0:
        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t '.format(
                env.get_num_agents(), x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, ))
        torch.save(policy.qnetwork_local.state_dict(),
                   './Nets_test9/navigator_checkpoint' + str(trials) + '.pth')
        action_prob = [1] * action_size

print("\n")
print("------------------------------------------------")
print("it takes to finish: %s" % (time.time() - start))
