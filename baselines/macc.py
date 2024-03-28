import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
import sys 
sys.path.append("..") 
from action_utils import select_action, translate_action
from consensus_builder import ConsensusBuilder

class MACC(nn.Module):
    def __init__(self, args, num_inputs):
        super(MACC, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)

        self.obs_encoder = nn.Linear(num_inputs, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell = nn.LSTMCell(args.hid_size, args.hid_size)

        self.consensus_builder = ConsensusBuilder(args.hid_size, args)
        # TODO: add consensus_builder_embedding_size in args
        self.embedding_net = nn.Embedding(args.consensus_builder_size+1, args.consensus_builder_embedding_size)
        self.latent_consensus_encoder = nn.Sequential(
            nn.Linear(args.consensus_builder_embedding_size, args.hid_size),
            nn.ReLU(),
            nn.Linear(args.hid_size, args.hid_size)
        )

        self.value_head = nn.Linear(args.hid_size, 1)

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.action_heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        # agent_mask = agent_mask.view(1, 1, n)
        # agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone() # clone gives the full tensor and avoid the error

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask


    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        obs, extras = x
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        hidden_state, cell_state = self.lstm_cell(encoded_obs, (hidden_state, cell_state))

        # TODO: add comm and consensus
        comm = hidden_state
        comm = comm * agent_mask
        with torch.no_grad():
            latent_consensus_representations = self.consensus_builder.calc_student(comm)
            latent_consensus_id = F.softmax(latent_consensus_representations, dim=-1).detach().max(-1)[1].unsqueeze(-1)
            latent_consensus_id[agent_mask.unsqueeze(0).expand(batch_size, -1, -1) == 0] = self.args.consensus_builder_size
            latent_consensus_embedding = self.embedding_net(latent_consensus_id.squeeze(-1))
        latent_consensus = self.latent_consensus_encoder(latent_consensus_embedding)
        latent_consensus = latent_consensus * agent_mask
     
        value_head = self.value_head(torch.cat((hidden_state, latent_consensus), dim=-1))
        if self.continuous:
            action_mean = self.action_mean(torch.cat(hidden_state, latent_consensus))
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action_out = (action_mean, action_log_std, action_std)
        else:
            action_out = [F.log_softmax(action_head(torch.cat((hidden_state, latent_consensus), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())
    
    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

