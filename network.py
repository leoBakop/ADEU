import torch
import torch.nn as nn 
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

'''
the following three classes (core and head) are responsible for generating an Bootstrap DQN ensemble
'''

def weights_init(m):
    """custom weights initialization"""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        print("default init")
        #m.weight.data.normal_(0.0, 0.02)
        #m.bias.data.fill_(0)
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' %classtype)


class CoreNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(CoreNetMLP, self).__init__()
        # Two hidden layers of size 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64*7*7)
        
        # Initialize weights (same as your conv version)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x




class CoreNet(nn.Module):
    def __init__(self, num_channels=4):
        super(CoreNet, self).__init__()
        self.num_channels = num_channels
        # params from ddqn appendix
        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        reshape = 64*7*7
        x = x.view(-1, reshape)
        return x
    
class HeadNet(nn.Module):
    def __init__(self, n_actions=4):
        super(HeadNet, self).__init__()
        mult = 64*7*7
        self.fc1 = nn.Linear(mult, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, num_channels, cnn = True, state_size = -1):
        super(EnsembleNet, self).__init__()
        if not cnn and state_size == -1:
            raise 'You should define the size of the input layer'
        self.core_net = CoreNet(num_channels=num_channels) if cnn else CoreNetMLP(input_dim=state_size, hidden_dim=512)
        self.net_list = nn.ModuleList([HeadNet(n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k=None):
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads




class DeepNAF(nn.Module):
    def __init__(self, state_size, action_size, layer_size_1, layer_size_2, seed,device, c = 1, max_speed = 1):
        super(DeepNAF, self).__init__()

        self.input_shape = state_size
        self.action_size = action_size
        self.device = device
        self.c = c
        self.testing_period = False
        concat_size = state_size+layer_size_1
        self.max_speed = max_speed
        self.fc1 = nn.Linear(self.input_shape, layer_size_1)
        self.bn1 = nn.BatchNorm1d(layer_size_1)
        self.fc2 = nn.Linear(concat_size, layer_size_2)
        self.bn2 = nn.BatchNorm1d(layer_size_2)
        self.fc3 = nn.Linear(concat_size, layer_size_2)
        self.bn3 = nn.BatchNorm1d(layer_size_2)
        self.fc4 = nn.Linear(concat_size, layer_size_2)
        self.bn4 = nn.BatchNorm1d(layer_size_2)
        self.action_values = nn.Linear(layer_size_2, action_size) #that is the mu
        self.value = nn.Linear(layer_size_2, 1) #that is v
        self.matrix_entries = nn.Linear(layer_size_2, int(self.action_size*(self.action_size+1)/2)) #that is the L


    def set_c(self, c):
        self.c = c
    
    def set_testing_period(self, period):
        self.testing_period = period

    def forward(self, input_, action=None):
        
        input_ = input_.to(self.device)
        x = torch.relu(self.fc1(input_))
        x = self.bn1(x)
        x = torch.relu(self.fc2(torch.cat([x, input_], dim=1)))
        x = self.bn2(x)
        x = torch.relu(self.fc3(torch.cat([x, input_], dim=1)))
        x = self.bn3(x)
        x = torch.relu(self.fc4(torch.cat([x, input_], dim=1)))

        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(input_.device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        if self.c < 10**(-2): return action_value, Q, V, action_value

        # Replace zero values with the small positive number
        P[P == 0] = 1e-4      
        self.noise_cov= self.c*torch.inverse(P) #just in order to export it
        #if self.testing_period: #in case of testing/evaluation there is no need for the noisy action
        #    return action_value, Q, V, action_value 
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), self.noise_cov)
        action = dist.sample()
        action = torch.clamp(action, min=-self.max_speed, max=self.max_speed) #normillization of the action
        return action, Q, V, action_value
        
class NAF(nn.Module):
    def __init__(self, state_size, action_size, layer_size_1, layer_size_2,device, c = 1.0, max_speed = 1, noisy_nets = False):
        super(NAF, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.device = device
        self.testing_period = False
        
        self.head_1 = nn.Linear(self.input_shape, layer_size_1)
        self.bn1 = nn.BatchNorm1d(layer_size_1)
        self.ff_1 = nn.Linear(layer_size_1, layer_size_2) if not noisy_nets else NoisyLinear(layer_size_1, layer_size_2)
        self.bn2 = nn.BatchNorm1d(layer_size_2)
        self.action_values = nn.Linear(layer_size_2, action_size)
        self.value = nn.Linear(layer_size_2, 1)
        self.matrix_entries = nn.Linear(layer_size_2, int(self.action_size*(self.action_size+1)/2))
        self.c =c
        self.max_speed = max_speed
        self.noise_cov = torch.zeros((self.action_size, self.action_size)).to(self.device)

    
    def forward(self, input_, action=None):
        
        input_ = input_.to(self.device)
        x = torch.relu(self.head_1(input_))
        x = self.bn1(x)
        x = torch.relu(self.ff_1(x))
        x = self.bn2(x)
        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(input_.device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        if self.c < 10**(-2): return action_value, Q, V, action_value
        #P = torch.where(P < 1e-4, torch.tensor(1e-4, dtype=torch.float32), P)
        self.noise_cov= self.c * torch.inverse(P) #just in order to export it
        #if self.testing_period: #in case of testing/evaluation there is no need for the noisy action
        #    return action_value, Q, V, action_value 
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), self.noise_cov)
        action = dist.sample()
        action = torch.clamp(action, min=-self.max_speed, max=self.max_speed) #normillization of the action
        return action, Q, V, action_value
    

    def get_noise_cov(self):
        if self.noise_cov.shape[0] >1:
            return self.noise_cov[0]
        return self.noise_cov

    def set_c(self, c):
        self.c = c
    
    def set_testing_period(self, period):
        self.testing_period = period



UNCERTAINTY = 187.5

class RND_Network(nn.Module):
    def __init__(self, device, input_size = 10, hidden_size = 64, max_speed = 1):
        super(RND_Network, self).__init__()
        self.device = device
        self.max_speed = max_speed
        # Define the layers
        hidden_size *= 2
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.set_uncertainty(UNCERTAINTY)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.uncertainty * self.output(x)
        
        return x
    

    def set_uncertainty(self,uncertainty):
        self.uncertainty = uncertainty
    def get_uncertainty(self):
        return self.uncertainty

class RND_NetworkPred(nn.Module):
    def __init__(self, device, input_size = 10, hidden_size = 64, max_speed = 1):
        super(RND_NetworkPred, self).__init__()
        self.device = device
        self.max_speed = max_speed
        # Define the layers
        hidden_size *= 2
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc4 = nn.Linear(2*hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.set_uncertainty(UNCERTAINTY)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.uncertainty * self.output(x)
        return x
    

    def set_uncertainty(self,uncertainty):
        self.uncertainty = uncertainty
    def get_uncertainty(self):
        return self.uncertainty