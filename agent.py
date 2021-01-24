from network import QNetwork # neural network
import random
import torch.optim as optim
from collections import namedtuple, deque
import torch

# set up the hyperparameters to use
BUFFER_SIZE = int(1e5) # replay experience buffer size. Original 1e5
BATCH_SIZE = 64        # minibatch size - this is the minibatch that we select randomly from the buffer to LEARN
                        # original set at 64
GAMMA = 0.99           # discount factor. Original set at 0.99
TAU = 1e-3             # for soft update of target parameters. Original set at 1e-3
LR = 5e-4              # learning rate. Original set at 5e-4
UPDATE_EVERY = 4       # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use GPU if available, otherwise CPU

# set up the agent
class Agent():
    '''Interacts and learns from the environment'''
    
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed(int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # initialise the timestep (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in the replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step +1) % UPDATE_EVERY
        
        if self.t_step ==0:
            # Get random subset from the memory, but ONLY if there are enough samples
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy
        Params
        ------
            state(array_like): current state
            eps(float): epsilon, epsilon-greedy action selection (to keep element of exploration)
        """
        # convert the state from the Unity network into a torch tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Note to pass it through the deep network, we need to take the numpy array and:
        # 1 - convert it to torch array with from_numpy()
        # 2 - convert it to float 32 as that is what is expected. Use .float()
        # 3 - Add a dimension on axis 0 with .unsqueeze(0). Because pytorch expects a BATCH of 1 dimensional arrays
        # to be fed into its network. For example feeding in a batch of 64 arrays, each of length 37. In our case,
        # with reinforcement learning we are only feeding one at a time, but the network still expects it to be 2D.
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        """Update value paratmers of the deep-Q network using given batch of experience tuples
        
        Params
        ------
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # get the max predicted Q values for the next states, from the target model
        # note: detach just detaches the tensor from the grad_fn - i.e. we are going to do some non-tracked
        # computations based on the value of this tensor (we DON'T update the target model at this stage)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimise the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params
        ======
            local_model (Pytorch model): weights will be copied from
            taret_model (Pytorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialise a ReplayBuffer object
        Params
        ------
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch put through the network
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e) # note it is a deque, so if there are more experiences than the buffer_size,
        # then it will remove the oldest experiences from the memory
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k = self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return(states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
        