# =========================================================================
#  Paper: S. Belsare, E. Diaz Badilla, and M. Dehghanimohammadabadi,
#         "Applied RL for Decision Making in Industrial Simulation
#         Environments," in Proc. Winter Simulation Conference (WSC), 2022.
#
#  Description: This script serves as the bridge between Simio simulation
#               software and Python-based Reinforcement Learning agents.
#               Simio calls this script to trigger RL agent training or
#               testing. Supported RL algorithms:
#               (1) Tabular Q-Learning
#               (2) Deep Double Q-Network (DDQN)
#               (3) Soft Actor-Critic (SAC)
#               (4) Actor-Critic
#
#  Requirements: Python 3.x with PyTorch
#                Conda environment "myenv_pytorch"
#                Simio Simulation Software
#                InputData.xlsx / InputData_DDQN.xlsx / InputData_Tabular.xlsx
#
#  Usage: Called from Simio via Python integration.
#         args[0] = choice (1=Tabular, 2=DDQN, 3=SAC, 4=ActorCritic)
#         args[1] = testing (1=train, 2=test)
# =========================================================================

# %%
import os


def call_python(args):
    """
    Main entry point called by Simio simulation.
    Activates conda environment and runs the appropriate RL agent.
    """

    ## --- Parse Arguments ---
    # choice: {1='Tabular', 2='DDQN', 3='SAC', 4='ActorCritic'}
    # testing: {1='train', 2='test'}
    try:
        choice = int(args[0])
        testing = int(args[1])
    except:
        choice = 1
        testing = 2

    ## --- Define Agent Scripts ---
    train_scripts = {
        1: 'RunAgent_Tabular.py',
        2: 'RunAgent_DDQN.py',
        3: 'RunAgent_SAC.py',
        4: 'RunAgent_ActorCritic.py'
    }

    test_scripts = {
        1: 'TestAgent_Tabular.py',
        2: 'TestAgent_DDQN.py',
        3: 'TestAgent_SAC.py',
        4: 'TestAgent_ActorCritic.py'
    }

    ## --- Select and Execute Script ---
    if testing == 1:
        script = train_scripts.get(choice, train_scripts[1])
    else:
        script = test_scripts.get(choice, test_scripts[1])

    command = (
        r'call conda activate myenv_pytorch && '
        r'cd python_files && '
        f'python ./{script} && '
        r'cd .. && '
        r'call conda deactivate'
    )
    os.system(command)
