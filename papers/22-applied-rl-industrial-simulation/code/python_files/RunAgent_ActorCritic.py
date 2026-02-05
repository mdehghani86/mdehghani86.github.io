# %%
from RL_Utils.Agent_ActorCritic import Agent
from Environment.Simio_Environment import simio_env_continuous

from copy import deepcopy
import numpy as np

import pickle
import os

# %%
alr = 5e-4
blr = 1e-4

gamma = 1

reward_function_type = 5
beta = 2

epsilon_decay = 1 - (3e-4)
epsilon_min = 0.1
epsilon_decay_start_episode = 200

epsilon_start = 1

# %%


def reward_function(cycle_time, cycletime_array, type=1):
    if type == 1:
        reward = 60 * (np.mean(np.array(cycletime_array)) - cycle_time)

    if type == 2:
        reward = 1 - (cycle_time/np.mean(np.array(cycletime_array))) ^ beta

    if type == 3:
        reward = 60 * (np.mean(np.array(cycletime_array)
                       [0:epsilon_decay_start_episode]) - cycle_time)

    if type == 4:
        reward = 1 - (cycle_time/np.mean(np.array(cycletime_array)
                      [0:epsilon_decay_start_episode])) ^ beta

    if type == 5:
        reward = 60 * (20 - cycle_time)/5

    return reward


# %%
try:
    run_num = pickle.load(open('./Data/run_num.pickle', 'rb'))
    agent = pickle.load(open('./Data/agent.pickle', 'rb'))
    env = pickle.load(open('./Data/env.pickle', 'rb'))

    scores_array = pickle.load(
        open('./results/ActorCritic/scores_array.pickle', 'rb'))
    loss_array = pickle.load(
        open('./results/ActorCritic/loss_array.pickle', 'rb'))
    cycletime_array = pickle.load(
        open('./results/ActorCritic/cycletime_array.pickle', 'rb'))
    eps_array = pickle.load(
        open('./results/ActorCritic/eps_array.pickle', 'rb'))

except:
    run_num = 0
    env = simio_env_continuous()
    agent = Agent(num_actions=1, input_dims=env.observation_space.shape,
                  alr=alr, blr=blr, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    agent.epsilon = epsilon_start

    scores_array = []
    loss_array = []
    cycletime_array = []
    eps_array = []

    pickle.dump(scores_array, open(
        './results/ActorCritic/scores_array.pickle', 'wb'))
    pickle.dump(loss_array, open(
        './results/ActorCritic/loss_array.pickle', 'wb'))
    pickle.dump(cycletime_array, open(
        './results/ActorCritic/cycletime_array.pickle', 'wb'))
    pickle.dump(eps_array, open(
        './results/ActorCritic/eps_array.pickle', 'wb'))


if(run_num > 0):

    ep_losses = pickle.load(open('./Data/ep_losses.pickle', 'rb'))

    obs = pickle.load(open('./Data/final_obs.pickle', 'rb'))
    new_obs = obs
    action = pickle.load(open('./Data/final_action.pickle', 'rb'))

    cycle_time_file_name = "./cycle_times/CycleTime_Experiment1_Scenario1_Rep" + \
        str(run_num)+".txt"
    txt_file = open(cycle_time_file_name, "r")
    cycle_time = txt_file.read()
    cycle_time = float(cycle_time)
    txt_file.close()
    os.remove(cycle_time_file_name)

    reward = reward_function(
        cycle_time=cycle_time, cycletime_array=cycletime_array, type=reward_function_type)
    #reward = pickle.load(open('./Data/final_reward.pickle', 'rb'))
    done = True

    actor_loss, critic_loss = agent.train(
        obs=obs, new_obs=new_obs, reward=reward, done=done)
    loss = [actor_loss, critic_loss]

    if(run_num < epsilon_decay_start_episode):
        agent.epsilon = epsilon_start

    ep_losses.append(loss)

    cycletime_array.append(cycle_time)
    scores_array.append(reward)
    loss_array.append(ep_losses)

    agent.save_model()

    eps_array.append(agent.epsilon)

    pickle.dump(scores_array, open(
        './results/ActorCritic/scores_array.pickle', 'wb'))
    pickle.dump(loss_array, open(
        './results/ActorCritic/loss_array.pickle', 'wb'))
    pickle.dump(cycletime_array, open(
        './results/ActorCritic/cycletime_array.pickle', 'wb'))
    pickle.dump(eps_array, open(
        './results/ActorCritic/eps_array.pickle', 'wb'))


# %%
done = False
obs = env.reset()

ep_losses = []

while(not(done)):
    action = agent.choose_action(obs)
    new_obs, reward, done, _ = env.step(action)

    if not(done):

        actor_loss, critic_loss = agent.train(
            obs=obs, new_obs=new_obs, reward=reward, done=done)
        loss = [actor_loss, critic_loss]

        obs = deepcopy(new_obs)

        ep_losses.append(loss)

    if done:

        pickle.dump(obs, open('./Data/final_obs.pickle', 'wb'))
        pickle.dump(action, open('./Data/final_action.pickle', 'wb'))
        pickle.dump(reward, open('./Data/final_reward.pickle', 'wb'))

agent.save_model()
pickle.dump(agent, open('./Data/agent.pickle', 'wb'))
pickle.dump(env, open('./Data/env.pickle', 'wb'))

run_num = run_num+1
pickle.dump(run_num, open('./Data/run_num.pickle', 'wb'))
pickle.dump(ep_losses, open('./Data/ep_losses.pickle', 'wb'))
