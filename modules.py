

import torch
import torch.nn as nn
from torch.nn import functional as F


class GLU(nn.Module):
    def __init__(self,inp_dim,out_dim):
        super().__init__()
        self.fc = nn.Linear(inp_dim,out_dim*2)
        self.od = out_dim
    def forward(self,x):
        x = self.fc(x)
        return x[:,:self.od]*torch.sigmoid(x[:,self.od:])


class Block(nn.Module):
    def __init__(self,in_features,out_features):
        super(Block, self).__init__()
        self.fc=nn.Linear(in_features,out_features)
        self.fc2=nn.Linear(out_features,out_features)
        self.bn=nn.BatchNorm1d(in_features)
        self.bn2=nn.BatchNorm1d(out_features)

        self.relu=nn.LeakyReLU()
    def forward(self,x):
        out=self.bn(x)
        out=self.fc(out)
        out=self.relu(out)
        out=self.bn2(out)
        out=self.fc2(out)

        return out

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features):
        super(ResBlock, self).__init__()
        self.fc=nn.Linear(in_features,out_features)
        self.fc2=nn.Linear(out_features,out_features)
        self.bn=nn.BatchNorm1d(in_features)
        self.bn2=nn.BatchNorm1d(out_features)
        self.downsample=GLU(in_features,out_features)
        # if in_features != out_features:
        #     self.downsample=nn.Linear(in_features,out_features)
        self.relu=nn.LeakyReLU()
    def forward(self,x):
        identity=x

        out=self.bn(x)
        out=self.fc(out)
        out=self.relu(out)
        out=self.bn2(out)
        out=self.fc2(out)
        if self.downsample:
            identity=self.downsample(identity)
        out=out+identity
        # out=self.relu(out)
        return out

def norm_layer(dim_in,dim_out):
    return nn.Sequential(nn.BatchNorm1d(dim_in),nn.Linear(dim_in,dim_out))
class MultitaskBlock(nn.Module):
    def __init__(self,dim_in,dim_hidden=20,n_hidden=0):
        super().__init__()

        if n_hidden==0:
            self.layers1=norm_layer(dim_in,3)
            self.layers2=norm_layer(dim_in,3)
            self.layers3=norm_layer(dim_in,3)
        else:
            input_modules1 = norm_layer(dim_in,dim_hidden)
            input_modules2 = norm_layer(dim_in,dim_hidden)
            input_modules3 = norm_layer(dim_in,dim_hidden)
            hidden_modules1=nn.Sequential(*[nn.Sequential(nn.ReLU(),norm_layer(dim_hidden,dim_hidden)) for _ in range(n_hidden)])
            hidden_modules2=nn.Sequential(*[nn.Sequential(nn.ReLU(),norm_layer(dim_hidden,dim_hidden)) for _ in range(n_hidden)])
            hidden_modules3 = nn.Sequential(*[nn.Sequential(nn.ReLU(),norm_layer(dim_hidden,dim_hidden)) for _ in range(n_hidden)])
            output_modules1=nn.Sequential(nn.ReLU(),norm_layer(dim_hidden,3))
            output_modules2=nn.Sequential(nn.ReLU(),norm_layer(dim_hidden,3))
            output_modules3=nn.Sequential(nn.ReLU(),norm_layer(dim_hidden,3))
            self.layers1=nn.Sequential(input_modules1,hidden_modules1,output_modules1)
            self.layers2=nn.Sequential(input_modules2,hidden_modules2,output_modules2)
            self.layers3=nn.Sequential(input_modules3,hidden_modules3,output_modules3)


    def forward(self,x):
        output1=self.layers1(x)
        output2=self.layers2(x)
        output3=self.layers3(x)
        return output1,output2,output3



class Net(nn.Module):
    def __init__(self,dim_input,n_hidden_layers=2,dim_hidden_layers=20,dim_out=1,multitask=False,n_hidden_multitask=0):
        super(Net, self).__init__()
        self.n_hidden_layers=n_hidden_layers
        self.dim_hidden_layers=dim_hidden_layers

        self.initial_layer=nn.Sequential(nn.BatchNorm1d(dim_input),
                                             nn.Linear(dim_input, dim_hidden_layers),
                                             )
        self.trunk_layers=nn.ModuleList([ResBlock(dim_hidden_layers,dim_hidden_layers)for _ in range(n_hidden_layers)])
            # self.trunk_layers = nn.ModuleList(
            #     [Block(dim_hidden_layers, dim_hidden_layers) for _ in range(n_hidden_layers)])

        if not multitask:
            self.fc=nn.Linear(dim_hidden_layers,dim_out)
        else:
            self.fc=MultitaskBlock(dim_hidden_layers,dim_hidden_layers,n_hidden=n_hidden_multitask)
        self.relu=nn.LeakyReLU()


        for m in self.modules():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias,0)

    def forward(self,x):
        out=self.relu(self.initial_layer(x))
        for l in self.trunk_layers:
            out=self.relu(l(out))
        out=self.fc(out)

        return out