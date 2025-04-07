
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the Every-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and Every-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) Every-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / Every-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "In every visit Monte Carlo, we directly discount the return using gamma all the way to the end of the episode. This has an immediate impact on the Q values we calculate for each state action pair. In contrast, Q-learning bootstraps updates just one step into the future. Resultingly, gamma doesn't have as direct an effect on the Q-values learned during Q-learning as it does on every visit Monte Carlo. As a result, changing gamma has a more gradual effect in Q-learning, and so we notice a smaller impact on the evaluation returns compared to Monte Carlo."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer

def question2_5() -> str:
    """
    (Short answer question):
    Provide a short explanation (<200 words) on the differences between the non-slippery and the slippery varian of the problem for [Q-learning / Every-Visit Monte Carlo].
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "In general, higher returns are observed in the non-slippery case. In the slippery case, state transitions are outside the full control of the agent. This means it may end up in a hole even when it's action selected is not to do so. Particularly, in our map, the reward is at the bottom right. It has to navigate down a narrow corridor beside two holes in order to reach the reward. In the slippery case, it is more likely to fall in the holes, particularly in these locations, where there is nowhere else to navigate. "  # TYPE YOUR ANSWER HERE (200 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "e"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "In our implementation of the exponential decay strategy, the epsilon is decayed at the end of each episode, based on the value of epsilon before that episode began. Resultingly, the rate at which the value of epsilon decays is highly dependent on the number of timesteps per episode. If episodes are short (as in the case in some environments), epsilon will be decayed rapidly, while if they are long, epsilon will be decayed more slowly. This makes it less applicable across different environments compared to strategies based on an exploration fraction, which doesn't depend on episode length."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "As the agent gets better at playing the game, estimating the reward becomes more difficult (in our case, because it is no longer always -200). As the agent learns to reach the goal the episode length, and therefore reward, become more variable, and therefore harder for the Q network to estimate. This results in an increase in the loss throughout training. "  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The spikes are due to the target network being updated. It can be seen from the graph that these updates occur every 2000 timesteps, which is our target update frequency. When we update our target network, we are essentially shifting the target the Q-network is aiming towards. This means the Q-network has to readjust towards the new target. Before it has time to readjust, it will produce a large loss on the new target. "  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    Note: This is a bonus question, which can be ignored. If you choose to answer this question, 
    you should also include the relevant code in the zip file that you submit.
    (Short answer):
    Provide a short description (200 words max) describing your approach. State and explain the 
    problem that you have chosen for this question and describe your answer. 
    (Long  answer): If you choose to prepare a longer answer, please state here in what form you 
    are submitting your answer. This can be for example by submission of a PDF or by a link. 
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "PDF"  # TYPE YOUR ANSWER HERE (200 words max)
    return answer
