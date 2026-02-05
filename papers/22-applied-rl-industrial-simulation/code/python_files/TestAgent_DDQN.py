# %%
from RL_Utils.Agent_DDQN import Agent
from Environment.Simio_Environment import simio_env_descrete

from copy import deepcopy
import numpy as np

import pickle
import os

# %%
lr = 5e-3
gamma = 1

epsilon_decay = 1 - (3e-4)

epsilon_decay_start_episode = 200

epsilon_min=0.0

mem_size = 10000
batch_size = 256

target_update_frequency = 200

num_priorities = 10
reward_function_type=3
beta = 2

epsilon_start = 0

# %%

def reward_function(cycle_time, cycletime_array, type=1):
    if type==1:
        reward = 60 * (np.mean(np.array(cycletime_array)) - cycle_time)

    if type==2:
        reward = 1 - (cycle_time/np.mean(np.array(cycletime_array)))^beta

    if type==3:
        reward = 60 * (np.mean(np.array(cycletime_array)[0:epsilon_decay_start_episode]) - cycle_time)

    if type==4:
        reward = 1 - (cycle_time/np.mean(np.array(cycletime_array)[0:epsilon_decay_start_episode]))^beta

    if type==5:
        reward = 60 * (14 - cycle_time)/14

    return reward

# %%
try:
    run_num = pickle.load(open('./Data/testrun_num.pickle', 'rb'))
    agent = pickle.load(open('./Data/testagent.pickle', 'rb'))
    env = pickle.load(open('./Data/testenv.pickle', 'rb'))

    scores_array = pickle.load(open('./results/DDQN/testscores_array.pickle', 'rb'))
    cycletime_array = pickle.load(open('./results/DDQN/testcycletime_array.pickle', 'rb'))

except:
    run_num = 0
    env = simio_env_descrete(num_priorities=num_priorities)
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape,
                    lr=lr, gamma=gamma, mem_size=mem_size, batch_size=batch_size,
                    epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, target_update_frequency=target_update_frequency)
    agent.load_model()
    agent.epsilon = epsilon_start

    scores_array = []
    cycletime_array = []
    
    pickle.dump(scores_array, open('./results/DDQN/testscores_array.pickle', 'wb'))
    pickle.dump(cycletime_array, open('./results/DDQN/testcycletime_array.pickle', 'wb'))


if(run_num > 0):

    obs = pickle.load(open('./Data/final_obs.pickle', 'rb'))
    new_obs = obs
    action = pickle.load(open('./Data/final_action.pickle', 'rb'))

    cycle_time_file_name = "./cycle_times/CycleTime_Experiment1_Scenario1_Rep"+str(run_num)+".txt"
    txt_file = open(cycle_time_file_name, "r")
    cycle_time = txt_file.read()
    cycle_time = float(cycle_time)
    txt_file.close()
    os.remove(cycle_time_file_name)

    cycletime_array.append(cycle_time)

    reward = reward_function(cycle_time=cycle_time, cycletime_array=cycletime_array, type=reward_function_type)
    #reward = pickle.load(open('./Data/final_reward.pickle', 'rb'))
    done = True

    scores_array.append(reward)

    pickle.dump(scores_array, open('./results/DDQN/testscores_array.pickle', 'wb'))
    pickle.dump(cycletime_array, open('./results/DDQN/testcycletime_array.pickle', 'wb'))


# %%
done = False
obs = env.reset()

while(not(done)):
    action = agent.choose_action(obs)
    new_obs, reward, done, _ = env.step(action)

    if not(done):
        obs = deepcopy(new_obs)

    if done:

        pickle.dump(obs, open('./Data/final_obs.pickle', 'wb'))
        pickle.dump(action, open('./Data/final_action.pickle', 'wb'))
        pickle.dump(reward, open('./Data/final_reward.pickle', 'wb'))

pickle.dump(agent, open('./Data/testagent.pickle', 'wb'))
pickle.dump(env, open('./Data/testenv.pickle', 'wb'))

run_num = run_num+1
pickle.dump(run_num, open('./Data/testrun_num.pickle', 'wb'))