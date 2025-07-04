import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torch.nn import init
from torch import Tensor
import torch.nn.functional as F
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='3'
class HyperLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, bias_value:float=0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HyperLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_value=bias_value
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(self.bias_value)).cuda()
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight)+self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class hyperp_generator():
    def __init__(self, fix_hyp):

        self.X = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
        ]).cuda()
        #self.X = torch.tensor(np.random.uniform(low=-20, high=20, size=(8,3))).float().cuda()
        #for verify

        #p=torch.Tensor([-0.3570, 19.9367, -0.3570, -9.4176]).reshape(-1,1)

    # output - 0 and 1 being negative and positive, respectively

        #self.y=torch.FloatTensor([np.random.choice(2,size=(8))]).permute(1,0).cuda()
        #3self.y[3]=1

        #self.y = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]).unsqueeze(1).cuda() # to test invariance to 3rd var X[,2]
        self.y = torch.FloatTensor([int(h) for h in '{0:08b}'.format(fix_hyp)]).unsqueeze(1).cuda()
        #self.hyp=fix_hyp[0]*1.0
        #self.model = torch.nn.Sequential(HyperLinear(3,1,use_bias=True, bias_value=self.hyp),  torch.nn.Sigmoid()).cuda()
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 1, bias=True), torch.nn.Sigmoid()).cuda()
        self.cross_entropy = torch.nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-6)
    def optim(self ):
        for iter in range(int(1e4)):
            self.optimiser.zero_grad()
            pred = self.model(self.X) # bias was dealt in nn.Linear
            loss = self.cross_entropy(pred, self.y)
            loss.backward()
            self.optimiser.step()
        #a_lreg = torch.cat((self.model[0].weight.data.squeeze(), self.hyp.cuda()), dim=0)
        a_lreg = torch.cat((self.model[0].weight.data.squeeze(), self.model[0].bias.data), dim=0)

        # # manual validation
        #p=a_lreg.reshape(-1,1)
        # x=torch.cat((self.X, torch.ones(8,1).cuda()),1)
        # y=torch.mm(x,p)
        # y=torch.round(torch.sigmoid(y))
        # # convolution validation
        # yy=torch.round(self.model(self.X))
        #print(y.squeeze(), self.y.squeeze(), yy.squeeze())
        res_lreg = pred - self.y

        return a_lreg, res_lreg, self.y.data.cpu().numpy()


    #a_lreg = torch.cat([model[0].weight.data.squeeze(), model[0].bias.data],0)
    def generator(self):

        a_lreg, res_lreg, y=self.optim()
        aq=torch.mean(res_lreg ** 2)
        print(aq)
        if aq<0.03:
            return a_lreg, y


def valid():
    space=[]
    hyp=np.load('hyperp.npy')
    X = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ])
    X1=torch.cat([X, torch.ones(8, 1)], dim=1)
    for i in range(len(hyp)):
        yy = torch.round(torch.sigmoid(X1 @ hyp[i]))
        space.append(yy.numpy())
    #with open('sample_Y.csv', 'w') as file:
    #    writer = csv.writer(file)
    #    writer.writerows(space)
    np.save('sample_Y.npy', np.array(space))

if __name__=='__main__':
    import os
    import csv
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    space=[]
    Y=[]
    j=0
    #valid()
    T=True
    while T:
        for i in range(1, np.power(2,8)):
            hyp_g = hyperp_generator(i)
            if hyp_g.generator() is not None:
                j += 1
                space.append(np.array(hyp_g.generator()[0].cpu().data))
                #print(hyp_g.generator())
                Y.append(np.array([int(h) for h in '{0:08b}'.format(i)]))
        #T=False
        if j>500:
            break
    np.save('hyperp_train.npy', np.array(space))
    np.save('sample_train.npy', Y)
