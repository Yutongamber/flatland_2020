import torch
from flatland.evaluators.client import FlatlandRemoteClient
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
import numpy as np
import time
from utils.observation_utils import normalize_observation
from timer import Timer
from dueling_double_dqn import Agent
from argparse import ArgumentParser, Namespace

remote_client = FlatlandRemoteClient()

#train_params = Namespace(batch_size=32, buffer_min_size=0, buffer_size=1000000, checkpoint_interval=50, eps_decay=0.99, eps_end=0, eps_start=0, gamma=0.99, hidden_size=256, learning_rate=5.2e-05, n_episodes=100, n_evaluation_episodes=100, num_threads=2, render=False, tau=0.001, update_every=8, use_gpu=True)
observation_tree_depth = 1
observation_radius = 10
observation_max_path_depth = 30

# n_features_per_node = 11
# n_nodes = 0
# for i in range(observation_tree_depth + 1):
#     n_nodes += np.power(4, i)
# state_size = n_features_per_node * n_nodes

state_size = 55
action_size = 5

# LOAD MODEL
policy = Agent(state_size, action_size)
policy.qnetwork_local.load_state_dict(torch.load("navigator_checkpoint1000.pth"))

#policy = DDDQNPolicy(state_size, action_size, train_params)
#policy.load('multi-9300')

my_observation_builder = TreeObsForRailEnv(max_depth=observation_tree_depth)
action_dict = dict()

evaluation_number = 0

def my_controller(env, obs, number_of_agents):
    for a in range(number_of_agents):
        agent = env.agents[a]
        if done[a]: 
           continue
        if agent.speed_data['position_fraction']>5:
           action_dict.update({a: 4}) # stop
           continue
        if info['action_required'][a]:
            if agent.position is not None:
                possible_transition_num = np.count_nonzero(env.rail.get_transitions(*agent.position, agent.direction))
                if possible_transition_num == 1:
                    action = 2
                else:
                    action = policy.act(normalize_observation(obs[a], observation_tree_depth, observation_radius=10), eps=0.01)
            else:
                action = 2
        else:
            action = 0
        action_dict.update({a: action})
    return  action_dict

evaluation_start = time.time() # added!
while True:

    evaluation_number += 1

    time_start = time.time()
    observation, info = remote_client.env_create(
        obs_builder_object=my_observation_builder
    )
    env_creation_time = time.time() - time_start
    if not observation:
        break

    print("Evaluation Number : {}".format(evaluation_number))
    
    local_env = remote_client.env
    done = local_env.dones
    number_of_agents = len(local_env.agents)
    agent_obs = [None] * number_of_agents

    for a in range(number_of_agents):
        if observation[a]:
            agent_obs[a] = normalize_observation(observation[a], observation_tree_depth, observation_radius=10)

    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0
    while True:
        time_start = time.time()

        action_dict = my_controller(remote_client.env, observation, number_of_agents)
        time_taken = time.time() - time_start
        time_taken_by_controller.append(time_taken)

        time_start = time.time()
     

        observation, all_rewards, done, info = remote_client.env_step(action_dict)

        steps += 1
        time_taken = time.time() - time_start
        time_taken_per_step.append(time_taken)

        if done['__all__']:
            print("Reward : ", sum(list(all_rewards.values())))
            break

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("=" * 100)
    print("=" * 100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(),
          np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("=" * 100)

print("Evaluation of all environments complete...")
print("Total time of evaluation: %s" %(time.time() - evaluation_start )) # added!
print(remote_client.submit())
