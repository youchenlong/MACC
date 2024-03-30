import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, input_size, mlp_hid_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, mlp_hid_size),
            nn.ReLU(),
            nn.Linear(mlp_hid_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)
    
class ConsensusBuilder(nn.Module):
    def __init__(self, hid_size, args):
        super(ConsensusBuilder, self).__init__()
        self.args = args
        self.hid_size = hid_size
        self.consensus_builder_hid_size = args.consensus_builder_hid_size
        self.consensus_builder_size = args.consensus_builder_size

        # student
        self.state2query = nn.Linear(hid_size, 32)
        self.state2key = nn.Linear(hid_size, 32)
        self.state2value = nn.Linear(hid_size, self.consensus_builder_hid_size)
        self.project_net = MLP(self.consensus_builder_hid_size, self.consensus_builder_hid_size, self.consensus_builder_size)

        # teacher
        self.teacher_state2query = nn.Linear(hid_size, 32)
        self.teacher_state2key = nn.Linear(hid_size, 32)
        self.teacher_state2value = nn.Linear(hid_size, self.consensus_builder_hid_size)
        self.teacher_project_net = MLP(self.consensus_builder_hid_size, self.consensus_builder_hid_size, self.consensus_builder_size)

    def calc_student(self, inputs):
        """
        inputs: [bs * n * hid_size]
        """
        query = self.state2query(inputs)
        key = self.state2key(inputs)
        value = self.state2value(inputs)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hid_size)
        attn = F.softmax(scores, dim=-1)
        representation = torch.matmul(attn, value)
        project = self.project_net(representation)
        return project
    
    def calc_teacher(self, inputs):
        """
        inputs: [bs * n * hid_size]
        """
        query = self.teacher_state2query(inputs)
        key = self.teacher_state2key(inputs)
        value = self.teacher_state2value(inputs)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hid_size)
        attn = F.softmax(scores, dim=-1)
        representation = torch.matmul(attn, value)
        projection = self.teacher_project_net(representation)
        return projection
    
    def EMA(self, source, target):
        for param_o, param_t in zip(source.parameters(), target.parameters()):
            param_t.data = param_t.data * self.args.ema_tau + param_o.data * (1. - self.args.ema_tau)
    
    def update_targets(self):
        self.EMA(self.state2query, self.teacher_state2query)
        self.EMA(self.state2key, self.teacher_state2key)
        self.EMA(self.state2value, self.teacher_state2value)
        self.EMA(self.project_net, self.teacher_project_net)

    def update_parameters(self):
        return list(self.state2query.parameters()) + list(self.state2key.parameters()) + \
               list(self.state2value.parameters()) + list(self.project_net.parameters())
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--consensus_builder_embedding_size', type=int, default=4,
                    help="consensus_builder_embedding_size")
    parser.add_argument('--consensus_builder_hid_size', type=int, default=64,
                        help="consensus_builder_hid_size")
    parser.add_argument('--consensus_builder_size', type=int, default=4,
                        help="consensus_builder_size")
    parser.add_argument('--online_temperature', type=float, default=0.1,
                        help="online_temperature")
    parser.add_argument('--target_temperature', type=float, default=0.04,
                        help="target_temperature")
    parser.add_argument('--center_tau', type=float, default=0.9,
                        help="center_tau")
    parser.add_argument('--ema_tau', type=float, default=0.996,
                        help="ema_tau")
    args = parser.parse_args()
    bs = 32
    n = 10
    hid_size = 64
    consensus_builder = ConsensusBuilder(hid_size, args)
    embedding_net = nn.Embedding(args.consensus_builder_size+1, args.consensus_builder_embedding_size)

    agent_mask = torch.ones(n, 1)
    comm = torch.rand(bs, n, hid_size)

    print(comm.shape)
    latent_consensus_projection = consensus_builder.calc_student(comm)
    latent_consensus_id = F.softmax(latent_consensus_projection, dim=-1).detach().max(-1)[1].unsqueeze(-1)
    print(latent_consensus_id.shape)
    latent_consensus_id[agent_mask.unsqueeze(0).expand(bs, -1, -1) == 0] = args.consensus_builder_size
    latent_consensus_embedding = embedding_net(latent_consensus_id.squeeze(-1))
    print(latent_consensus_embedding.shape)
