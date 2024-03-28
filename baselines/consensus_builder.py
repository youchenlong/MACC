import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.consensus_builder_hid_size = self.args.consensus_builder_hid_size
        self.consensus_builder_size = self.args.args.consensus_builder_size

        # student
        self.view_obs_net = MLP(self.hid_size, self.consensus_builder_hid_size*2, self.consensus_builder_hid_size)
        self.project_net = MLP(self.consensus_builder_hid_size, self.consensus_builder_hid_size, self.consensus_builder_size)

        # teacher
        self.teacher_view_obs_net = MLP(self.hid_size, self.consensus_builder_hid_size*2, self.consensus_builder_hid_size)
        self.teacher_project_net = MLP(self.consensus_builder_hid_size, self.consensus_builder_hid_size, self.consensus_builder_size)

    def calc_student(self, inputs):
        representation = self.view_obs_net(inputs)
        project = self.project_net(representation)
        return project
    
    def cal_teacher(self, inputs):
        representation = self.teacher_view_obs_net(inputs)
        project = self.teacher_project_net(representation)
        return project
    
    def update(self):
        for param_o, param_t in zip(self.view_obs_net.parameters(), self.teacher_view_obs_net.parameters()):
            param_t.data = param_t.data * self.args.tau + param_o.data * (1. - self.args.tau)
        
        for param_o, param_t in zip(self.project_net.parameters(), self.teacher_project_net.parameters()):
            param_t.data = param_t.data * self.args.tau + param_o.data * (1. - self.args.tau)

    def update_parameters(self):
        return list(self.view_obs_net.parameters()) + list(self.project_net.parameters())