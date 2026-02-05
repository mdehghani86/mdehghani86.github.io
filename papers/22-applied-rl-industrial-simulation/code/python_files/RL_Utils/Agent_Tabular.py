import numpy as np
import pandas as pd


class Agent():
    def __init__(self, n_actions, input_dims,
                 lr=1e-2, gamma=0.9, epsilon_decay=0.995, epsilon_min=0):

        self.n_actions = n_actions
        self.input_dims = input_dims

        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.epsilon = 1

        self.epsilon_min = epsilon_min

        df = pd.read_excel(
            open('../InputData.xlsx', 'rb'), sheet_name='InputData')

        df = df.drop(columns=['AppointmentTime',
                     'Priority', 'NumberAppointments'])

        df['index_val'] = df['MeanProcTime'].astype(str) + '_' + df['StDevProcessingTime'].astype(
            str) + '_' + df['MeanSetupTime'].astype(str) + '_' + df['StDevSetupTime'].astype(str) + '_' + df['DueDate'].astype(str)

        indexes = df['index_val'].tolist()

        data = [[0] * (n_actions)]*len(indexes)

        col_lab = [0]*(n_actions)

        for i in range(1, len(col_lab)):
            col_lab[i] = i
        self.q_table = pd.DataFrame(data, columns=col_lab, index=indexes)

    def get_obs(self, obs):
        x = ''

        x = x + str(int(obs[0])) + '_' + str(int(obs[1])) + '_' + str(int(obs[2])) + \
            '_' + str(int(obs[3])) + '_' + str(obs[4])

        return x

    def choose_action(self, obs):
        obs = self.get_obs(obs)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.q_table.loc[obs].to_numpy())

        return (action+1)

    def train(self, obs, new_obs, action, reward, done):

        action = action - 1
        obs = self.get_obs(obs)

        if not(done):
            new_obs = self.get_obs(new_obs)
            q_next = max(self.q_table.loc[new_obs])
        else:
            q_next = 1 - int(done)

        loss = ((reward + (self.gamma * q_next)) -
                       self.q_table.loc[obs, action])

        self.q_table.loc[obs, action] = self.q_table.loc[obs, action] + \
            self.lr * loss

        if done:
            self.epsilon = max((self.epsilon * self.epsilon_decay), self.epsilon_min)

        return  abs(loss)

    def save_model(self, file_path="./rl_models/Tabular/q_table.csv"):
        self.q_table.to_csv(file_path)

    def load_model(self, file_path="./rl_models/Tabular/q_table.csv"):
        self.q_table = pd.read_csv(file_path, index_col=0)
        numeric_cols = [int(numeric_string) for numeric_string in self.q_table.columns]
        self.q_table.columns = numeric_cols