import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as TorchModels

class WarpLoss(nn.Module):
    def __init__(self, cfg, dataset, num_classes, margin=1.0, dimensions=512, warp=True):
        super(WarpLoss, self).__init__()
        self.Loss_norm = cfg['loss_norm']
        self.num_classes = num_classes
        self.Warp = warp
        self.k1 = cfg[dataset]['k1'] 
        self.k2 = cfg[dataset]['k2']
        self.alpha = cfg[dataset]['alpha']
        self.Temp = cfg[dataset]['temp']
        self.margin = margin

        #Initialize Proxies
        Proxies = torch.randn(num_classes, dimensions)
        nn.init.kaiming_normal_(Proxies, mode='fan_out')
        self.Proxies = torch.nn.Parameter(Proxies)
    
    def _UpdateParameters(self, cfg):
        self.k1 = cfg['k1'] 
        self.k2 = cfg['k2']
        self.alpha = cfg['alpha']
        self.Temp = cfg['temp']
    
    def _Warp(self, Distances, HotLabels, WarpMask):
        ColdLabels = 1.0 - HotLabels
        
        #Outward Warp
        NoMatchWarp = self.k1*HotLabels*Distances + ColdLabels*Distances
        Deltas = Distances - NoMatchWarp
        k1Warp = NoMatchWarp + self.margin*Deltas.detach()
        
        #Inward Warp
        k2Warp = HotLabels*(self.k2*Distances + self.alpha*(1.0 - self.k2)) + ColdLabels*Distances

        #Combining the Inward and Outward Warps
        FinalWarp = self.Temp*WarpMask*k1Warp + self.Temp*(1.0 - WarpMask)*k2Warp
        return FinalWarp

    def _Distances(self, Output, Proxies):
        Proxies = Proxies.permute(1,0)
        #L2-Norm between Embeddings and Centroids
        HyperNorms = list()
        for i in range(Output.shape[0]):
            #HNorm = (Output[i].unsqueeze(-1) - Means[0])**2
            HNorm = (Output[i].unsqueeze(-1) - Proxies)**2
            HyperNorms.append(torch.sqrt(HNorm.sum(dim=0)))
        
        return torch.stack(HyperNorms, dim=0)

    #More optimized way to Calculate Euclidean Distances.
    def _euclidean_dist(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = dist - 2 * torch.matmul(x, y.t())
        # dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _embedding(self, Output, HotLabels):
        #L2-Distances between each Proxy
        #EuclideanNorms = self._Distances(Output, self.Proxies)
        EuclideanNorms = self._euclidean_dist(Output, self.Proxies)
        HotDistances = torch.max(HotLabels*EuclideanNorms, dim=1, keepdim=True)[0]# + 1.0

        if self.Warp:
            #Softmax Warping (More numerically stable way of calculating loss)
            WarpMask = (HotDistances < self.alpha).float().detach()
            WarpedNorms = self._Warp(EuclideanNorms, HotLabels, WarpMask)
            LogProbs = F.log_softmax(-1*WarpedNorms, dim=1)
            CrossEntropy = torch.max(-1*HotLabels*LogProbs, dim=1)[0]
            return CrossEntropy.mean(dim=0)
        
        else:
            Default = HotDistances - EuclideanNorms
            Kernel = self.Temp*Default
        
        #Final Cross-Entropy Loss
        HotProbs = torch.sum(torch.exp(Kernel), dim=1)
        CrossEntropy = torch.log(HotProbs).mean(dim=0)
        return CrossEntropy
    
    def forward(self, Output, Labels):
        #Create batch-wise class masks off of Labels
        HotLabels = F.one_hot(Labels.long(), num_classes=self.num_classes)
        
        #Final-Loss
        FinalLoss = self._embedding(Output, HotLabels)

        return self.Loss_norm*FinalLoss

        
class WarpNet(nn.Module):
    def __init__(self, dataset, pre_trained=True, norm_layer=nn.BatchNorm2d, dimensions=512):
        super(WarpNet, self).__init__()
        #Bulding Blocks of the Model
        self.Backbone = TorchModels.__dict__['resnet50'](pretrained=pre_trained, norm_layer=norm_layer)
        self.Backbone.Embedding = nn.Linear(in_features=2048, out_features=dimensions, bias=True)
        if dataset != 'sop':
            self._initialize_weights() #Empirically, we found that doing this marginally hurt the results on SOP.
        self.Backbone.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.LNorm = nn.LayerNorm(2048, elementwise_affine=False)

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.Backbone.Embedding.weight, mode='fan_out')
        nn.init.constant_(self.Backbone.Embedding.bias, 0)

    def forward(self, Input):       
        #Forward Batch through Network
        x = self.Backbone.conv1(Input)
        x = self.Backbone.bn1(x)
        x = self.Backbone.relu(x)
        x = self.Backbone.maxpool(x)
        x = self.Backbone.layer1(x)
        x = self.Backbone.layer2(x)
        x = self.Backbone.layer3(x)
        x = self.Backbone.layer4(x)
      
        y = self.Backbone.avgpool(x) + self.avgpool(x) #Actually MaxPool + AvgPool
        x = torch.flatten(y, 1)
        x = self.LNorm(x)
        Output = self.Backbone.Embedding(x)
        return Output

if __name__=='__main__':
    print("Just a PyTorch Model. Nothing to see here...")
