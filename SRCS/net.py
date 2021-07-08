import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n, n_eofs, v, s, nS, time_mean):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150 ,150)
        self.fc6 = nn.Linear(150, n_eofs)
        self.act = nn.SELU()
        self.lact= nn.Sigmoid()
        self.bn0 = nn.BatchNorm1d(n)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(150)
        self.bn3 = nn.BatchNorm1d(150)
        self.bn4 = nn.BatchNorm1d(150)
        self.bn5 = nn.BatchNorm1d(150)
        self.bn6 = nn.BatchNorm1d(n_eofs)
        self.dropout = nn.Dropout(p=0.15)
        self.v   = v
        self.s   = s
        self.nS  = nS
        self.time_mean = time_mean

        # initialization
        #nn.init.kaiming_normal_(self.fc1.weight)
        #nn.init.kaiming_normal_(self.fc2.weight)
        #nn.init.kaiming_normal_(self.fc3.weight)
        #nn.init.kaiming_normal_(self.fc4.weight)
        #nn.init.kaiming_normal_(self.fc5.weight)
        #nn.init.kaiming_normal_(self.fc6.weight)
    def minmaxNorm(self, dt, minval, maxval):
        normed_dt = dt * (maxval - minval) + minval
        return normed_dt

    def _recompose(self, x):
        U = x.type(torch.float)/torch.sqrt(self.nS-1)
        test_mean = torch.reshape(self.time_mean[0,:], [1, self.time_mean[0,:].shape[0]])
        y_hat = (torch.matmul(U, torch.matmul(torch.diag(self.s), torch.conj(self.v).t()))) *  torch.sqrt(self.nS-1) + test_mean
        return y_hat

    def forward(self, x):
        #x   = self.bn0(x)
        #out = self.bn1(self.act(self.fc1(x)))
        #out = self.bn2(self.act(self.fc2(out)))
        #out = self.bn3(self.act(self.fc3(out)))
        #out = self.bn4(self.act(self.fc4(out)))
        #out = self.bn5(self.act(self.fc5(out)))
        #aux = self.fc6(out)

        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.act(self.fc4(out))
        out = self.act(self.fc5(out))
        aux = self.fc6(out)
        out = self._recompose(aux)
        return out, aux

