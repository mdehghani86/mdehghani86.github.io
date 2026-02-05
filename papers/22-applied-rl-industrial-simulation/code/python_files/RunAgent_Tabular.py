# %%
from RL_Utils.Agent_Tabular import Agent
from Environment.Simio_Environment import simio_env_descrete

from copy import deepcopy
import numpy as np

import pickle
import os

# %%
lr = 1e-2
gamma = 1

epsilon_decay = 1 - (7e-3)

epsilon_decay_start_episode = 200

epsilon_min=0.0

num_priorities = 6
reward_function_type=1
beta = 2

epsilon_start = 1

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
    run_num = pickle.load(open('./Data/run_num.pickle', 'rb'))
    agent = pickle.load(open('./Data/agent.pickle', 'rb'))
    env = pickle.load(open('./Data/env.pickle', 'rb'))

    eps_array = pickle.load(open('./Results/Tabular/eps_array.pickle', 'rb'))
    scores_array = pickle.load(open('./Results/Tabular/scores_array.pickle', 'rb'))
    loss_array = pickle.load(open('./Results/Tabular/loss_array.pickle', 'rb'))
    cycletime_array = pickle.load(open('./results/Tabular/cycletime_array.pickle', 'rb'))

except:
    run_num = 0
    env = simio_env_descrete(num_priorities=num_priorities)
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape,
                    lr=lr, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    agent.epsilon = epsilon_start
    eps_array = []
    scores_array = []
    loss_array = []
    cycletime_array = []

    pickle.dump(eps_array, open('./Results/Tabular/eps_array.pickle', 'wb'))
    pickle.dump(scores_array, open('./Results/Tabular/scores_array.pickle', 'wb'))
    pickle.dump(loss_array, open('./Results/Tabular/loss_array.pickle', 'wb'))
    pickle.dump(cycletime_array, open('./Results/Tabular/cycletime_array.pickle', 'wb'))

if(run_num > 0):

    ep_losses = pickle.load(open('./Data/ep_losses.pickle', 'rb'))

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

    loss = agent.train(obs, new_obs, action, reward, done)

    if(run_num<epsilon_decay_start_episode):
        agent.epsilon = epsilon_start

    ep_losses.append(loss)

    eps_array.append(agent.epsilon)
    scores_array.append(reward)
    loss_array.append(ep_losses)

    agent.save_model()

    pickle.dump(eps_array, open('./Results/Tabular/eps_array.pickle', 'wb'))
    pickle.dump(scores_array, open('./Results/Tabular/scores_array.pickle', 'wb'))
    pickle.dump(loss_array, open('./Results/Tabular/loss_array.pickle', 'wb'))
    pickle.dump(cycletime_array, open('./Results/Tabular/cycletime_array.pickle', 'wb'))

# %%
done = False
obs = env.reset()

ep_losses = []

while(not(done)):
    action = agent.choose_action(obs)
    new_obs, reward, done, _ = env.step(action)

    score = reward

    if not(done):

        loss = agent.train(obs, new_obs, action, reward, done)
        obs = deepcopy(new_obs)
        ep_losses.append(loss)

    if done:
        
        pickle.dump(obs, open('./Data/final_obs.pickle', 'wb'))
        pickle.dump(action, open('./Data/final_action.pickle', 'wb'))
        pickle.dump(reward, open('./Data/final_reward.pickle', 'wb'))
    

pickle.dump(agent, open('./Data/agent.pickle', 'wb'))
pickle.dump(env, open('./Data/env.pickle', 'wb'))

run_num = run_num+1
pickle.dump(run_num, open('./Data/run_num.pickle', 'wb'))
pickle.dump(ep_losses, open('./Data/ep_losses.pickle', 'wb'))