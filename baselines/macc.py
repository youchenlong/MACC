import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
import sys 
sys.path.append("..") 
from action_utils import select_action, translate_action

class MACC(nn.Module):
    def __init__(self, args, num_inputs):
        super(MACC, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        self.obs_encoder = nn.Linear(num_inputs, args.hid_size)
        self.message_encoder = nn.Linear(args.hid_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell = nn.LSTMCell(args.hid_size, args.hid_size)

        # self.value_head = nn.Linear(args.hid_size, 1)
        # self.action_heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])

        self.value_head = nn.Linear(args.hid_size * 2, 1)
        self.action_heads = nn.ModuleList([nn.Linear(args.hid_size * 2, o) for o in args.naction_heads])


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone()

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
        # encoded_obs: [bs * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        # hidden_state: [(bs * n) * hid_size]
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        # agent_mask: [bs * n * n * 1]
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # agent_mask_alive: [bs * n * n * 1]
        agent_mask_alive = agent_mask.clone()
        # agent_mask_transpose: [bs * n * n * 1]
        agent_mask_trasnpose = agent_mask.transpose(1, 2)

        # comm: [bs * n * hid_size]
        comm = hidden_state.view(batch_size, n, self.hid_size)
        # comm: [bs * n * n * hid_size]
        comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)
        comm = comm * agent_mask_alive
        comm = comm * agent_mask_trasnpose
        comm_sum = comm.sum(dim=1)
        # c: [bs * n * hid_size]
        c = self.message_encoder(comm_sum)

        """
        # inp: [bs * n * hid_size]
        inp = encoded_obs + c
        # inp: [(bs * n) * hid_size]
        inp = inp.view(batch_size * n, self.hid_size)
        hidden_state, cell_state = self.lstm_cell(inp, (hidden_state, cell_state))
        """

        # inp: [bs * n * hid_size]
        inp = encoded_obs
        # inp: [(bs * n) * hid_size]
        inp = inp.view(batch_size * n, self.hid_size)
        hidden_state, cell_state = self.lstm_cell(inp, (hidden_state, cell_state))

        h = hidden_state.view(batch_size, n, self.hid_size)
        value_head = self.value_head(torch.cat((h, c), dim=-1))
        action_out = [F.log_softmax(head(torch.cat((h, c), dim=-1)), dim=-1) for head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

