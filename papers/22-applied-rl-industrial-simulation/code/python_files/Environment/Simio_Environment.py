import gym
from gym import spaces

import numpy as np
from copy import deepcopy
import pandas as pd

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


class simio_env_descrete(gym.Env):

    def __init__(self, num_priorities=4):
        super(simio_env_descrete, self).__init__()

        self.current_step = 0

        self.reward_range = (-200, 200)

        self.action_space = spaces.Discrete(num_priorities)

        _ = self.reset()
        num_dim = len(self.df_copy.columns)

        low_arr = np.zeros(num_dim)
        high_arr = np.ones(num_dim) * 200

        self.observation_space = spaces.Box(
            low=low_arr, high=high_arr, dtype=np.float64)

        self.num_actions = num_priorities

    def Reward_Function(self, done, priority):
        self.cumsumreward = self.cumsumreward + \
            self.df_copy.loc[(
                self.episode_order[self.current_step - 1]), 'MeanProcTime']
        if done:

            '''
            txt_file = open('./cycle_times/CycleTime.txt', "r")

            cycle_time = txt_file.read()
            cycle_time = float(cycle_time)
            txt_file.close()
            '''
            #cycle_time = 2.5
            #reward = 150 - (cycle_time * 60)
            reward = self.cumsumreward

        else:
            #reward = 0
            reward = - \
                (((200/self.num_actions)*priority) -
                 self.df_copy.loc[(self.episode_order[self.current_step - 1]), 'MeanProcTime'])**2

        return reward

    def reset(self):

        self.current_step = 0
        self.cumsumreward = 0

        self.df = pd.read_excel(
            open('../InputData.xlsx', 'rb'), sheet_name='InputData')

        self.df_copy = deepcopy(self.df)
        self.df_copy = self.df_copy.drop(
            columns=['AppointmentTime', 'Priority', 'NumberAppointments'])

        self.episode_order = self.df.index.to_numpy()
        np.random.shuffle(self.episode_order)

        self.max_steps = self.episode_order.shape[0]

        self.obs = self.df_copy.loc[self.episode_order[self.current_step]].to_numpy(
        ).tolist()

        self.current_step += 1

        return self.obs

    def step(self, priority):

        self.df.loc[(self.episode_order[self.current_step - 1]),
                    'Priority'] = priority

        if (self.current_step == self.max_steps):
            done = True
            wb = Workbook()
            ws = wb.active
            ws.title = "InputData"
            for r in dataframe_to_rows(self.df, index=False, header=True):
                ws.append(r)
            wb.save("../InputData.xlsx")

            new_obs = None
        else:
            done = False
            new_obs = self.df_copy.loc[self.episode_order[self.current_step]].to_numpy(
            ).tolist()

        reward = self.Reward_Function(done, priority)

        self.obs = new_obs
        self.current_step += 1

        return new_obs, reward, done, {}

    def render(self, mode='human', close=False):
        return


class simio_env_continuous(gym.Env):

    def __init__(self):
        super(simio_env_continuous, self).__init__()

        self.current_step = 0

        self.reward_range = (-200, 200)

        self.action_space = spaces.Box(low=np.array(
            [0]), high=np.array([1]), dtype=np.float64)

        _ = self.reset()
        num_dim = len(self.df_copy.columns)

        low_arr = np.zeros(num_dim) * -1000
        high_arr = np.ones(num_dim) * 1000

        self.observation_space = spaces.Box(
            low=low_arr, high=high_arr, dtype=np.float64)

    def Reward_Function(self, done, priority):

        self.cumsumreward = self.cumsumreward + priority * \
            self.df_copy.loc[(
                self.episode_order[self.current_step - 1]), 'MeanProcTime']
        if done:

            '''
            txt_file = open('./cycle_times/CycleTime.txt', "r")

            cycle_time = txt_file.read()
            cycle_time = float(cycle_time)
            txt_file.close()
            '''

            #cycle_time = 2.5
            #reward = 150 - (cycle_time * 60)
            reward = self.cumsumreward/1000

        else:
            #reward = 0
            reward = - \
                ((200*priority) -
                 self.df_copy.loc[(self.episode_order[self.current_step - 1]), 'MeanProcTime'])**2
        return reward

    def reset(self):

        self.current_step = 0
        self.cumsumreward = 0

        self.df = pd.read_excel(
            open('../InputData.xlsx', 'rb'), sheet_name='InputData')

        self.df_copy = deepcopy(self.df)
        self.df_copy = self.df_copy.drop(
            columns=['AppointmentTime', 'Priority', 'NumberAppointments'])

        self.episode_order = self.df.index.to_numpy()
        np.random.shuffle(self.episode_order)

        self.max_steps = self.episode_order.shape[0]

        self.obs = self.df_copy.loc[self.episode_order[self.current_step]].to_numpy(
        ).tolist()

        self.current_step += 1

        return self.obs

    def step(self, priority):

        self.df.loc[(self.episode_order[self.current_step - 1]),
                    'Priority'] = priority

        if (self.current_step == self.max_steps):
            done = True
            wb = Workbook()
            ws = wb.active
            ws.title = "InputData"
            for r in dataframe_to_rows(self.df, index=False, header=True):
                ws.append(r)
            wb.save("../InputData.xlsx")
            new_obs = None

        else:
            done = False
            new_obs = self.df_copy.loc[self.episode_order[self.current_step]].to_numpy(
            ).tolist()

        reward = self.Reward_Function(done, priority)

        self.obs = new_obs
        self.current_step += 1

        return new_obs, reward, done, {}

    def render(self, mode='human', close=False):
        return
