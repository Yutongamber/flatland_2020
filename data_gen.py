# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:42:34 2020

@author: LOMO
"""

import csv
import random
import time
import sys
from libs.cell_graph_dispatcher import CellGraphDispatcher
import flatland.core.env
import flatland.utils.rendertools as rt
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

import matplotlib.pyplot as plt
import numpy as np


from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator
import json
import pickle
import os

import pandas as pd


dim_list = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,]

num_agent = [15,20,25,30,35,40,45,50,55]


# Use a the malfunction generator to break agents from time to time
stochastic_data = MalfunctionParameters(malfunction_rate=1./100000,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )


# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train



origin_obs_row = ['dist_own_target_encountered','dist_other_target_encountered','dist_other_agent_encountered','dist_potential_conflict','dist_unusable_switch','dist_to_next_branch',
          'dist_min_to_target','num_agents_same_direction','num_agents_opposite_direction','num_agents_malfunctioning','speed_min_fractional','num_agents_ready_to_depart',
                ]
basic_obs_row = ['dist_own_target_encountered','dist_other_target_encountered','dist_other_agent_encountered','dist_potential_conflict','dist_unusable_switch','dist_to_next_branch',
          'dist_min_to_target','num_agents_same_direction','num_agents_opposite_direction','num_agents_malfunctioning','speed_min_fractional','num_agents_ready_to_depart',
                ]
origin_obs_row.extend(['L'+i for i in basic_obs_row])
origin_obs_row.extend(['LL'+i for i in basic_obs_row])
origin_obs_row.extend(['LF'+i for i in basic_obs_row])
origin_obs_row.extend(['LR'+i for i in basic_obs_row])
origin_obs_row.extend(['LB'+i for i in basic_obs_row])
origin_obs_row.extend(['F'+i for i in basic_obs_row])
origin_obs_row.extend(['FL'+i for i in basic_obs_row])
origin_obs_row.extend(['FF'+i for i in basic_obs_row])
origin_obs_row.extend(['FR'+i for i in basic_obs_row])
origin_obs_row.extend(['FB'+i for i in basic_obs_row])
origin_obs_row.extend(['R'+i for i in basic_obs_row])
origin_obs_row.extend(['RL'+i for i in basic_obs_row])
origin_obs_row.extend(['RF'+i for i in basic_obs_row])
origin_obs_row.extend(['RR'+i for i in basic_obs_row])
origin_obs_row.extend(['RB'+i for i in basic_obs_row])
origin_obs_row.extend(['B'+i for i in basic_obs_row])
origin_obs_row.extend(['BL'+i for i in basic_obs_row])
origin_obs_row.extend(['BF'+i for i in basic_obs_row])
origin_obs_row.extend(['BR'+i for i in basic_obs_row])
origin_obs_row.extend(['BB'+i for i in basic_obs_row])

def root_property(obs):
    root = []
    root.append(obs.dist_own_target_encountered) if str(obs.dist_own_target_encountered) != 'inf' else root.append(-2)
    root.append(obs.dist_other_target_encountered) if str(obs.dist_other_target_encountered) != 'inf' else root.append(-2)
    root.append(obs.dist_other_agent_encountered) if str(obs.dist_other_agent_encountered) != 'inf' else root.append(-2)
    root.append(obs.dist_potential_conflict) if str(obs.dist_potential_conflict) != 'inf' else root.append(-2)
    root.append(obs.dist_unusable_switch) if str(obs.dist_unusable_switch) != 'inf' else root.append(-2)
    root.append(obs.dist_to_next_branch) if str(obs.dist_to_next_branch) != 'inf' else root.append(-2)
    root.append(obs.dist_min_to_target) if str(obs.dist_min_to_target) != 'inf' else root.append(-2)
    root.append(obs.num_agents_same_direction) if str(obs.num_agents_same_direction) != 'inf' else root.append(-2)
    root.append(obs.num_agents_opposite_direction) if str(obs.num_agents_opposite_direction) != 'inf' else root.append(-2)
    root.append(obs.num_agents_malfunctioning) if str(obs.num_agents_malfunctioning) != 'inf' else root.append(-2)
    root.append(obs.speed_min_fractional) if str(obs.speed_min_fractional) != 'inf' else root.append(-2)
    root.append(obs.num_agents_ready_to_depart) if str(obs.num_agents_ready_to_depart) != 'inf' else root.append(-2)
    return root

def LFRB(X, root, obs):
    try:
        if str(obs.childs[X]) == '-inf':
            #print(X)
            root.extend([-1 for _ in range(60)])
        else:
            root.extend(root_property(obs.childs[X]))
            if str(obs.childs[X].childs['L']) == '-inf':
                root.extend([-1 for _ in range(12)])
            else:
                root.extend(root_property(obs.childs[X].childs['L']))
            if str(obs.childs[X].childs['F']) == '-inf':
                root.extend([-1 for _ in range(12)])
            else:
                root.extend(root_property(obs.childs[X].childs['F']))
            if str(obs.childs[X].childs['R']) == '-inf':
                root.extend([-1 for _ in range(12)])
            else:
                root.extend(root_property(obs.childs[X].childs['R']))    
            if str(obs.childs[X].childs['B']) == '-inf':
                root.extend([-1 for _ in range(12)])
            else:
                root.extend(root_property(obs.childs[X].childs['B']))
    except Exception as e:
        pass
    return root

def obs_property(obs):
    root = []
    try:
        root.append(obs.dist_own_target_encountered)
        root.append(obs.dist_other_target_encountered)
        root.append(obs.dist_other_agent_encountered)
        root.append(obs.dist_potential_conflict)
        root.append(obs.dist_unusable_switch)
        root.append(obs.dist_to_next_branch)
        root.append(obs.dist_min_to_target)
        root.append(obs.num_agents_same_direction)
        root.append(obs.num_agents_opposite_direction)
        root.append(obs.num_agents_malfunctioning)
        root.append(obs.speed_min_fractional)
        root.append(obs.num_agents_ready_to_depart)
    except Exception as e:
        pass
  
    LFRB('L',root,obs)
    LFRB('F',root,obs)
    LFRB('R',root,obs)
    LFRB('B',root,obs)
    return root

agent_row = ['initial_position','initial_direction','direction','target','moving','position_fraction','speed','transition_action_on_cellexit',
             'malfunction','malfunction_rate','next_malfunction','nr_malfunctions','moving_before_malfunction','handle','status','position','old_direction','old_position','priority']
def agents_property(env_agent,order):
    return [env_agent.initial_position,
            env_agent.initial_direction,
            env_agent.direction,
            env_agent.target,
            env_agent.moving,
            env_agent.speed_data['position_fraction'],
            env_agent.speed_data['speed'],
            int(env_agent.speed_data['transition_action_on_cellexit']),
            env_agent.malfunction_data['malfunction'],
            env_agent.malfunction_data['malfunction_rate'],
            env_agent.malfunction_data['next_malfunction'],
            env_agent.malfunction_data['nr_malfunctions'],
            env_agent.malfunction_data['moving_before_malfunction'],
            env_agent.handle,
            int(env_agent.status),
            env_agent.position,
            env_agent.old_direction,
            env_agent.old_position,
            np.where(np.array(order)==env_agent.handle)[0][0]]

agent_row.extend(origin_obs_row)
agent_row.append('action')


for trials in range(1, 1500):   # 每一个trials生成一个环境，写入一对csv

        # Reset environment
    x_dim = np.random.choice(dim_list)
    y_dim = x_dim
    n_agents = np.random.choice(num_agent)
    try:
        env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=50,
                                                       # Number of cities in map (where train stations are)
                                                       seed=np.random.choice(range(0,1000000)),  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=3,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=TreeObservation)

        obs = env.reset(True, True)
        print('x_dim:',x_dim,'agents:',n_agents)
    except Exception as e:
        print('pass')
        continue
    
    filename = 'test'+str(trials)+'.csv'  # 保存路径，文件名
    final_filename = 'final_'+filename
    f = open(filename,'w',encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(agent_row)
    dispatcher = CellGraphDispatcher(env)
    for step in range(1000):
        #env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
    
        #if record_images:
         #   env_renderer.gl.save_image("./Images/flatland_frame_{:04d}.bmp".format(frame_step))
         #   frame_step += 1
    
        # Action
    
        
        action_dict = dispatcher.step(step)
        # Environment step
        if np.random.choice([1,2,3,4]) == 3:
            print(step)
            for i in range(len(env.agents)):
                to_be_write = []
                to_be_write.extend(agents_property(env.agents[i],dispatcher.agents_order))
                if step == 0:
                    to_be_write.extend(obs_property(obs[0][i]))
                else:
                    to_be_write.extend(obs_property(obs[i]))
                to_be_write.append(int(action_dict[i]))
                csv_writer.writerow(to_be_write)
            
        obs, all_rewards, done, _ = env.step(action_dict)
        if done['__all__']:
            print('done_all')
            break
    f.close()

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        continue
    df = df.dropna(subset=['action'])
    df['old_direction'].fillna(-1,inplace=True)
    df['moving'] = df['moving'].apply(lambda x:0 if x is False else 1)
    df['moving_before_malfunction'] = df['moving_before_malfunction'].apply(lambda x:0 if x is False else 1)
    df = df.dropna(axis=1,how='any')
    df = df.drop(['initial_position'],axis=1)
    df = df.drop(['target'],axis=1)
    
    df.to_csv(final_filename, index=False, header=True)














