import torch
import torch.nn as nn
class AdaptiveReweight(nn.Module):
    def __init__(self, channel, reduction=4,momentum=0.1,index=0):
        self.channel=channel
        super(AdaptiveReweight, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LayerNorm([channel // reduction]),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.register_buffer('running_scale', torch.zeros(1))
        self.momentum=momentum
        self.ind=index
        

    def forward(self, x):
        b, c, _, _ = x.size()
        _x=x.view(b,c,-1)
        x_var=_x.var(dim=-1)

        y = self.fc(x_var).view(b, c)

        if self.training:
            scale=x_var.view(-1).mean(dim=-1).sqrt()
            self.running_scale.mul_(1. - self.momentum).add_(scale.data*self.momentum)
        else:
            scale=self.running_scale
        inv=(y/scale).view(b,c,1,1)
        return inv.expand_as(x)*x  
    
class CE(nn.Module):
    def __init__(self, num_features, pooling=False, num_groups=1, num_channels=64, T=3, dim=4, eps=1e-5, momentum=0.1,
                    *args, **kwargs):
        super(CE, self).__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.dim = dim

        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        self.AR=AdaptiveReweight(num_features)
        self.pool=None
        if pooling:
            self.pool=nn.MaxPool2d(2,stride=2)
    
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))

        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.x_weight = nn.Parameter(torch.zeros(1))
        print(self.num_channels)

    def forward(self, X):
        N,C,H,W=X.size()
        xin=self.AR(X)
        x_pool=self.pool(X) if self.pool is not None else X
        
        x_pool=x_pool.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        x = X.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        _, d, m = x.size()
        
        if self.training:
            mean = x_pool.mean(-1, keepdim=True)
            
            xc = x_pool - mean
            
            P = [None] * (self.T + 1)
            P[0] = torch.eye(d,device=X.device).expand(self.num_groups, d, d)
            Sigma = torch.baddbmm(self.eps, P[0], 1. / m, xc, xc.transpose(1, 2))

            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            Sigma_N = Sigma * rTr
            for k in range(self.T):
                mat_power3=torch.matmul(torch.matmul(P[k],P[k]),P[k])
                P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, mat_power3, Sigma_N)
            
            wm = P[self.T]  

            self.running_mean.mul_(1. - self.momentum).add_(mean.data*self.momentum)
            self.running_wm.mul_((1. - self.momentum)).add_(self.momentum * wm.data)
        else:
            xc = x - self.running_mean
            wm = self.running_wm

        xn = wm.matmul(x)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()

        x_weight=torch.sigmoid(self.x_weight)
        return x_weight*Xn+(1-x_weight)*xin