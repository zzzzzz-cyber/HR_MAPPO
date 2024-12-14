from TD3_VAE import TD3_VAE
from POTD3 import POTD3
from DDPG_VAE import MADDPG
from MAPPO import MAPPO
from MA_HPPO import MA_HPPO
from MAPPO_VAE import MAPPO_VAE
from MAPPO_Transformer import MAPPO_TFM
from MAPPO_VAE_Transformer import MAPPO_VAE_Trans
from MAPPO_VAE_Transformer_5 import MAPPO_VAE_Trans_5
from MAPPO_VAE_Transformer_10 import MAPPO_VAE_Trans_10
from MAPPO_VAE_Transformer_15 import MAPPO_VAE_Trans_15
from MAPPO_VAE_Transformer_20 import MAPPO_VAE_Trans_20
from MAPPO_VAE_Transformer1 import MAPPO_VAE_Trans_u
from LSTM_COM_DDPG import LSTM_COM_DDPG
from PER_CMDRL import PER_CMDRL
from LSTM_PD3QN import LSTM_PD3QN
from IPPO import IPPO
from CCM_MADRL import CCM_MADDPG
import matplotlib.pyplot as plt
from mec_env import MecEnv
# from mec_env_independence import MecEnv
import sys

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 64
NUMBER_OF_EVAL_EPISODES = 1

DONE_PENALTY = None

ENV_SEED = 37
NUMBERofAGENTS = 50
def create_ddpg(algorithm ,InfdexofResult, env, env_eval, EPISODES_BEFORE_TRAIN, MAX_EPISODES):
    MRL = algorithm
    ccmaddpg = MRL(InfdexofResult=InfdexofResult, env=env, env_eval=env_eval, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size,
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound, batch_size=EPISODES_BEFORE_TRAIN,episodes_before_train = EPISODES_BEFORE_TRAIN, epsilon_decay= MAX_EPISODES)
                  
    ccmaddpg.interact(MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES)
    return ccmaddpg

def plot_ddpg(ddpg, algorithm_str, parameter, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend([algorithm_str])
    plt.savefig("./Figures/" + algorithm_str + "_%s.png" % parameter)

def run(InfdexofResult, algorithm, algorithm_str):
    env = MecEnv(n_agents=NUMBERofAGENTS, batch_size=EPISODES_BEFORE_TRAIN, env_seed=ENV_SEED)
    eval_env = MecEnv(n_agents=NUMBERofAGENTS, batch_size=EPISODES_BEFORE_TRAIN, env_seed=ENV_SEED) #ENV_SEED will be reset at set()
    ddpg = [create_ddpg(algorithm, InfdexofResult, env, eval_env, EPISODES_BEFORE_TRAIN, MAX_EPISODES)]
    plot_ddpg(ddpg, algorithm_str, "%s" % InfdexofResult)



if __name__ == "__main__":
    # [POTD3, CCM_MADDPG, MAPPO, LSTM_PD3QN, PER_CMDRL, LSTM_COM_DDPG, MAPPO_VAE_Trans]
    algorithms = [MAPPO_VAE_Trans]
    algorithms_str = ['MAPPO_VAE_Trans']
    # index = ['10']
    # index = ['5', '6', '7', '8', '9']
    index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in index:
        for algorithm, algorithm_str in zip(algorithms, algorithms_str):
            InfdexofResult = i  # set run runnumber for indexing results,
            run(InfdexofResult, algorithm, algorithm_str)
