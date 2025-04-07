from abc import ABC, abstractmethod
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import DefaultDict, Dict, Iterable, List
from collections import defaultdict

from rl2025.exercise3.networks import FCNetwork
from rl2025.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see https://gymnasium.farama.org/api/spaces/ for more information on Gymnasium spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min


        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        # Adding log strategy
        elif self.epsilon_decay_strategy == "log":
            # For new decay strategy
            self.timestep_counter = 0
            self.t_max = 350000
            self.scaling_factor = 1
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(*args, **kwargs):
            ### PUT YOUR CODE HERE ###
            # Epsilon start and epsilon min are attributes of the class so shouldn't need to pass these to function
            # Don't want to reduce epsilon below the minimum value
            if timestep >= max_timestep * self.exploration_fraction: 
                self.epsilon = self.epsilon_min
                return
            
            self.epsilon = self.epsilon_start - ((self.epsilon_start - self.epsilon_min) / (max_timestep * self.exploration_fraction)) * timestep
                 

        def epsilon_exponential_decay(*args, **kwargs):
            ### PUT YOUR CODE HERE ###
            
            # Using formula that sum to i = (n (n + 1))/2
            # self.epsilon = max(self.epsilon_min, self.epsilon_start * self.epsilon_exponential_decay_factor **(((timestep*(timestep + 1))/ 2) / max_timestep))
            epsilon = self.epsilon * self.epsilon_exponential_decay_factor ** (timestep/max_timestep)
            self.epsilon = max(epsilon, self.epsilon_min) # as told to implement in drop in lab

        def epsilon_log_decay():
            # if timestep = 0, epsilon = epsilon start
            if timestep == 0:
                self.epsilon = self.epsilon_start
                self.timestep_counter += 1
                return
            
            if self.timestep_counter == 1: # The first log value we calculate - have to normalise by this
                first_log_value = np.log(self.t_max - timestep)
                self.scaling_factor = self.epsilon_start / first_log_value
                self.timestep_counter += 1
            
            if self.t_max - timestep < 1: # This would cause an error in log calculation
                self.epsilon = self.epsilon_min
                return

            self.epsilon = max(self.epsilon_min, self.scaling_factor * np.log(self.t_max - timestep)) # Ensures we return at least self.epsilon_min


        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # linear decay
            ### PUT YOUR CODE HERE ###
            epsilon_linear_decay()
        elif self.epsilon_decay_strategy == "exponential":
            # exponential decay
            ### PUT YOUR CODE HERE ###
            epsilon_exponential_decay()
        elif self.epsilon_decay_strategy == "log":
            epsilon_log_decay()
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        

        # the forward function appears to expect a pytorch tensor, but obs is a numpy array. Have to convert to pytorch tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension and convert to tensor
        
        # No gradients required 
        with torch.no_grad():
            output = self.critics_net.forward(obs_tensor)

        # output will contain a value for each action. Take the argmax to find the best action
        # Not sure yet if this should return a batch of indices or just one. It seems the batch size is only to do with the update
        # and the action is just taking one action at each specific timestep. Therefore, should just return the index of the specific action
        # Training seems to be done in the same format as the q_learning, so can return the index of the selected action
        # print('DBUG: action index: ', action_index)
        if explore and np.random.random() < self.epsilon:
            return self.action_space.sample()
                # sample() returns an integer representing a random sample from the action space
                # print('DBUG random sample: ', self.action_space.sample())   
        else:
            action_index = torch.argmax(output, dim=1).item() # Item converts the torch tensor to an index. Not sure if it is necessary
            return action_index
        
        

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###

        # batch contains 64 (by default) experiences, in a named tuple
        # Step 1 - Calculate the y_j for each of the samples in the minibatch
        # Each of the named items in batch contains a numpy array. Will have to convert to pytorch arrays to be processed by model.
        # PyTorch models can take batches of inputs, so could process a number of states (64 by default in this case) at the same time
        # next_states etc. are stored in the replay buffer as numpy arrays, not as State or Action objects, so can convert them directly to torch tensors
        # Since these have been converted to torch tensors, will gradients be backpropagated all the way to these, updating them??
        next_states = torch.FloatTensor(batch.next_states)  # Convert numpy array to PyTorch tensor
        states = torch.FloatTensor(batch.states)
        # We use actions to index the output from the model, to access the Q value of the specific action taken. This is why it is a 
        # LongTensor, which stores integers instead of floats
        actions = torch.LongTensor(batch.actions.type(torch.int64))
        rewards = torch.FloatTensor(batch.rewards)
        done = torch.FloatTensor(batch.done) # batch.done is a numpy array of Boolean values. In the FloatTensor, True will be converted to 1.0 and False to 0.0

        # Compute target Q values
        # Don't need gradients for the target network
        with torch.no_grad():
            target_q_values = self.critics_target.forward(next_states)
            # print('DBUG target_q_values shape', target_q_values.shape)
            max_target_q_values = torch.max(target_q_values, dim=1)[0]  # Max returns max values (as a tensor) as well as max indices (as a tensor). Index with [0] to get max values
            # print('DBUG: max target q values shape: ', max_target_q_values.shape)
        # print('DBUG done shape: ', done.shape)

        # Zero out values for terminal states. This will result in y just being reward. Have to squeeze done as it is of shape [64, 1] which results in incorrect
        # broadcasting with max_target_q_values which is of shape [64]
        max_target_q_values = max_target_q_values * (1 - done.squeeze())  
        # print('DBUG: max target q values after masking shape and type : ', max_target_q_values.shape)
        # print('DBUG rewards shape: ', rewards.shape)
        # print('DBUG gamma*max_target_q_values shape: ', (self.gamma * max_target_q_values).shape)
        # Reward is of shape [64, 1], so need to squeeze it to be of shape [64], so that y is also of shape [64]
        y = rewards.squeeze() + self.gamma * max_target_q_values
        # print('DBUG y shape', y.shape)

        # Calculate critics net q values 
        q_values = self.critics_net.forward(states)
        # print('DBUG q_values shape:', q_values.shape)

        # Have to get the Q value of the action actually taken.
        q_values_taken = q_values[torch.arange(q_values.shape[0]), actions.squeeze()] # q_values[0] contains the batch size. Actions is of shape [64, 1], so have to squeeze to get to shape [64] for correct indexing of q_values
        # print('DBUG q_values_taken shape: ', q_values_taken.shape)
        loss_vector = (y - q_values_taken)**2

        mean_square_error = loss_vector.mean()

        # Zero out the gradients
        self.critics_optim.zero_grad()

        # Backpropagate gradients
        mean_square_error.backward()

        # Update the weights of the network
        self.critics_optim.step()

        # Have to update the target network if we are at the specified timestep
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.hard_update(self.critics_net)

        q_loss = mean_square_error
        return {"q_loss": q_loss}


class DiscreteRL(Agent):
    """The DiscreteRL Agent for Ex 3 using tabular Q-Learning without neural networks
    
    This agent implements standard Q-learning with a discretized state space for
    environments with continuous state spaces. Suitable for small state-action spaces.
    
    :attr gamma (float): discount factor for future rewards
    :attr epsilon (float): probability of choosing a random action for exploration
    :attr alpha (float): learning rate for Q-value updates
    :attr n_acts (int): number of possible actions in the environment
    :attr q_table (DefaultDict): table storing Q-values for state-action pairs
    :attr position_bins (np.ndarray): bins for discretizing position dimension
    :attr velocity_bins (np.ndarray): bins for discretizing velocity dimension

        ** YOU CAN CHANGE THE PROVIDED SETTINGS **

    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float = 0.99,
        epsilon: float = 0.99,
        alpha: float = 0.05,
        **kwargs
    ):
        """Constructor of DiscreteRL agent

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount factor gamma
        :param epsilon (float): epsilon for epsilon-greedy action selection
        :param alpha (float): learning rate alpha
        """
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = kwargs.get("learning_rate", alpha)
        self.n_acts: int = action_space.n
        
        super().__init__(action_space=action_space, observation_space=observation_space)
        
        # Initialize Q-table as defaultdict with default value of 0 for any new state-action pair
        # This avoids having to initialize all possible state-action pairs explicitly
        self.q_table: DefaultDict = defaultdict(lambda: 0)

        # For mountain car environment discretization - creates k bins for each dimension, e.g. k=8
        # Position range: -1.2 to 0.6 (8 bins)
        self.position_bins = np.linspace(-1.2, 0.6, 8)
        # Velocity range: -0.07 to 0.07 (8 bins)
        self.velocity_bins = np.linspace(-0.07, 0.07, 8)

    def discretize_state(self, obs: np.ndarray) -> int:
        """Discretizes a continuous state observation into a unique integer identifier.

        Converts continuous observation values into discrete bins and creates
        a unique integer identifier for the discretized state.

        :param obs (np.ndarray): continuous state observation (position, velocity)
        :return (int): unique integer identifier for the discretized state
        """
        # Convert continuous position to discrete bin index
        position_idx = np.digitize(obs[0], self.position_bins) - 1
        
        # Convert continuous velocity to discrete bin index
        velocity_idx = np.digitize(obs[1], self.velocity_bins) - 1
        
        # Create a unique integer ID by combining position and velocity indices
        # This creates a unique ID for each discretized state using a simple hash function
        unique_state_id = position_idx * len(self.velocity_bins) + velocity_idx
        return unique_state_id

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Returns an action using epsilon-greedy action selection.

        With probability epsilon, selects a random action for exploration.
        Otherwise, selects the action with the highest Q-value for the current state.

        :param obs (np.ndarray): current observation state
        :param explore (bool): flag indicating whether exploration should be enabled
        :return (int): action the agent should perform (index from action space)
        """
        # Discretize the observation
        state = self.discretize_state(obs)

        # Epsilon-greedy action selection
        if explore and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            # Get Q-values for all actions in current state
            q_values = [self.q_table[(state, a)] for a in range(self.n_acts)]
            # Return action with highest Q-value (randomly break ties)
            return np.random.choice(np.flatnonzero(q_values == np.max(q_values))) # will this return index of actual value? Indices - np.flatnonzero turns the boolean array into indices of max values

    def update(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience using Q-learning algorithm.

         ** YOU NEED TO IMPLEMENT THIS FUNCTION FOR Q3 BUT YOU CAN REUSE YOUR Q LEARNING CODE FROM Q2 (you can include it here or you adapt the files from Q2 to work of the mountain car problem **

        Implements the Q-learning update equation:
        Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        :param obs (np.ndarray): current observation state
        :param action (int): applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray): next observation state
        :param done (bool): flag indicating whether episode is done
        :return (float): updated Q-value for current observation-action pair
        """


        # Convert continuous observations to discrete state identifiers
        state = self.discretize_state(obs)         # Current state
        next_state = self.discretize_state(n_obs)   # Next state


        best_q = np.max([self.q_table[(next_state, act)] for act in range(self.n_acts)])

        if done:
            best_q = 0


        self.q_table[(state, action)] = self.q_table[(state, action)] + self.alpha*(reward + self.gamma*best_q - self.q_table[(state, action)])
        ### PUT YOUR CODE HERE ###

        return {f"Q_value_{state}" : self.q_table[(state, action)]}


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters (specifically epsilon for exploration).

        ** YOU CAN CHANGE THE PROVIDED SCHEDULING **

        Implements a linear decay schedule for epsilon, reducing from 1.0 to 0.01
        over the first 20% of total timesteps.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        decay_progress = min(1.0, timestep / (0.20 * max_timestep))
        self.epsilon = 1.0 - decay_progress * 0.99  # Decays from 1.0 to 0.01

