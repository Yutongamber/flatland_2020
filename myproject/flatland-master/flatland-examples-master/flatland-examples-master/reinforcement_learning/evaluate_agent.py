import math
import multiprocessing
import sys
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.timer import Timer
from utils.observation_utils import normalize_observation
from reinforcement_learning.dddqn_policy import DDDQNPolicy


def eval_policy(env_params, checkpoint, n_eval_episodes, max_steps, seed, render):
    # evaluation is faster on CPU, except if you have huge networks
    parameters = {
        'use_gpu': False
    }

    policy = DDDQNPolicy(state_size, action_size, Namespace(**parameters), evaluation_mode=True)
    policy.qnetwork_local = torch.load(checkpoint)

    env_params = Namespace(**env_params)

    # Environment parameters
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Malfunction and speed profiles
    # TODO pass these parameters properly from main!
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1. / 2000,  # Rate of malfunctions
        min_duration=20,  # Minimal duration
        max_duration=50  # Max duration
    )
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

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
    env.reset(True, True)

    if render:
        env_renderer = RenderTool(env, gl="PGL")

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []
    inference_times = []
    preproc_times = []
    agent_times = []
    step_times = []

    for episode_idx in range(n_eval_episodes):
        inference_timer = Timer()
        preproc_timer = Timer()
        agent_timer = Timer()
        step_timer = Timer()

        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        step_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        step_timer.end()

        if render:
            env_renderer.set_new_rail()

        final_step = 0

        for step in range(max_steps - 1):
            agent_timer.start()
            for agent in env.get_agent_handles():
                if obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(obs[agent], tree_depth=observation_tree_depth, observation_radius=observation_radius)
                    preproc_timer.end()

                action = 0
                if info['action_required'][agent]:
                    inference_timer.start()
                    action = policy.act(agent_obs[agent], eps=0.0)
                    inference_timer.end()
                action_dict.update({agent: action})
            agent_timer.end()

            step_timer.start()
            obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            if render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

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

        inference_times.append(inference_timer.get())
        preproc_times.append(preproc_timer.get())
        agent_times.append(agent_timer.get())
        step_times.append(step_timer.get())

        print(
            "☑️  Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} "
            "\t🚉 Env: {:.3f}s "
            "\t🤖 Agent: {:.3f}s (per step: {:.3f}s) \t[preproc: {:.3f}s \tinfer: {:.3f}s]".format(
                normalized_score,
                completion * 100.0,
                final_step,
                step_timer.get(),
                agent_timer.get(),
                agent_timer.get() / final_step,
                preproc_timer.get(),
                inference_timer.get())
        )

    return scores, completions, nb_steps, agent_times, step_times


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="checkpoint to load", required=True, type=str)
    parser.add_argument("-n", "--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)
    parser.add_argument("--use_gpu", dest="use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--render", help="render a single episode", default=False, type=bool)
    args = parser.parse_args()

    render = args.render
    nb_threads = 1
    eval_per_thread = args.n_evaluation_episodes

    if not render:
        nb_threads = multiprocessing.cpu_count()
        eval_per_thread = max(1, math.ceil(args.n_evaluation_episodes / nb_threads))

    total_nb_eval = eval_per_thread * nb_threads
    print("Will evaluate policy {} over {} episodes on {} threads.".format(args.file, total_nb_eval, nb_threads))

    if total_nb_eval != args.n_evaluation_episodes:
        print("(Rounding up from {} to fill all cores)".format(args.n_evaluation_episodes))

    env_params_dict = {
        # sample configuration
        "n_agents": 5,
        "x_dim": 35,
        "y_dim": 35,
        "n_cities": 4,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,

        "seed": 42,
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }
    env_params = Namespace(**env_params_dict)

    print("Environment parameters:")
    pprint(env_params_dict)

    # Calculate space dimensions and max steps
    tree_observation = TreeObsForRailEnv(max_depth=env_params.observation_tree_depth)
    num_features_per_node = tree_observation.observation_dim
    tree_depth = 2
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    action_size = 5
    state_size = num_features_per_node * nr_nodes
    max_steps = int(4 * 2 * (env_params.x_dim + env_params.y_dim + (env_params.n_agents / env_params.n_cities)))

    results = []
    if args.render:
        results.append(eval_policy(env_params_dict, args.file, eval_per_thread, max_steps, 0, render))

    else:
        with Pool() as p:
            results = p.starmap(eval_policy,
                                [(env_params_dict, args.file, eval_per_thread, max_steps, seed, render) for seed in
                                 range(nb_threads)])

    scores = []
    completions = []
    nb_steps = []
    times = []
    step_times = []
    for s, c, n, t, st in results:
        scores.append(s)
        completions.append(c)
        nb_steps.append(n)
        times.append(t)
        step_times.append(st)

    print("✅ Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} \tAgent total: {:.3f}s (per step: {:.3f}s)".format(
        np.mean(scores),
        np.mean(completions) * 100.0,
        np.mean(nb_steps),
        np.mean(times),
        np.mean(times) / np.mean(nb_steps)
    ))

    print("⏲️  Agent sum: {:.3f}s \tEnv sum: {:.3f}s \tTotal sum: {:.3f}s".format(
        np.sum(times),
        np.sum(step_times),
        np.sum(times) + np.sum(step_times)
    ))
