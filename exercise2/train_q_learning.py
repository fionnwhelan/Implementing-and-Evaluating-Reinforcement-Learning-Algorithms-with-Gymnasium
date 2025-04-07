import gymnasium as gym
from tqdm import tqdm

from rl2025.constants import EX2_QL_CONSTANTS as CONSTANTS
from rl2025.exercise2.agents import QLearningAgent
from rl2025.exercise2.utils import evaluate
import os

CONFIG = {
    "eval_freq": 1000, # keep this unchanged
    "alpha": 0.05,
    "epsilon": 0.9,
    "gamma": 0.99,
}
CONFIG.update(CONSTANTS)


def q_learning_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(CONFIG["env"], render_mode="human")
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"]+1)):
        obs, _ = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            agent.learn(obs, act, reward, n_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        total_reward += episodic_return

        # if eps_num > 0 and eps_num % config["eval_freq"] == 0:
        if eps_num == 10000:
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table

# The original training section
if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    total_reward, evaluation_return_means, _, q_table = train(env, CONFIG)
    
# if __name__ == "__main__":
#     env = gym.make(CONFIG["env"])
#     configurations = [{"gamma": 0.99}, {"gamma": 0.8}]
#     gamma_final_eval_returns = {}
#     mean_final_eval_returns = {}
#     for config in configurations:
#         CONFIG.update(config)
#         gamma_final_eval_returns[CONFIG['gamma']] = []
#         for _ in range(10):
#             total_reward, evaluation_return_means, _, q_table = train(env, CONFIG)
#             gamma_final_eval_returns[CONFIG['gamma']].append(evaluation_return_means[-1])
#         mean_final_eval_returns[CONFIG['gamma']] = sum(gamma_final_eval_returns[CONFIG['gamma']]) / len(gamma_final_eval_returns[CONFIG['gamma']])

#     # Save the evaluation returns to a file

#     if not os.path.exists('EX2_ouputs/slippery_false'):
#         os.makedirs('EX2_ouputs/slippery_false')
        
#     with open('EX2_ouputs/slippery_false/q_learning_eval_returns.txt', 'w') as f:
#         f.write("Slippery = False. \nMean final evaluation returns for different gamma values:\n")
#         for gamma, mean_return in mean_final_eval_returns.items():
#             f.write(f"Gamma {gamma}: {mean_return}\n")
        
#         f.write("\nAll final evaluation returns for different gamma values:\n")
#         for gamma, returns in gamma_final_eval_returns.items():
#             f.write(f"Gamma {gamma}: {returns}\n")

#     print("Evaluation returns saved to q_learning_eval_returns.txt")



