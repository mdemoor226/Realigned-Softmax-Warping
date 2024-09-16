import torch
import torch.nn.functional as F

class WarpLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(WarpLoss, self).__init__()
        self.k1 = cfg.k1
        self.k2 = cfg.k2
        self.alpha = cfg.alpha
        self.Temp = cfg.Temp

    def _Followup(self, cfg):
        self.k1 = cfg.fk1
        self.k2 = cfg.fk2
        self.alpha = cfg.falpha
        self.Temp = cfg.fTemp
    
    def _Warp(self, Distances, HotLabels, ColdLabels, alpha=8.0, k1=0.25, k2=2.0):
        Switch = (Distances.detach() <= alpha).float()
        HotTerms = (HotLabels*Switch).max(dim=1, keepdim=True)[0]
        
        #Two separate warps boosting Inward/Outward Forces
        InWarp = HotLabels*(k2*Distances + (alpha*(1.0 - k2))) + ColdLabels*Distances
        OutWarp = k1*HotLabels*Distances + ColdLabels*Distances
        
        #Combination. Alpha determines where to apply each
        FinalWarp = HotTerms*OutWarp + (1.0 - HotTerms)*InWarp
        
        return FinalWarp

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

    def _embedding(self, Output, Proxies, HotLabels):
        ColdLabels = (1 - HotLabels)     
        
        #L2-Distances between each gt_Centroid
        HyperNorms = self._euclidean_dist(Output, Proxies)

        #Useless Comment
        WarpedNorms = self._Warp(HyperNorms, HotLabels, ColdLabels, self.alpha, self.k1, self.k2)
        Deltas = HyperNorms - WarpedNorms
        DeltaMask = torch.max(HotLabels*(WarpedNorms.detach() < self.alpha).float(), dim=1, keepdim=True)[0]
        ProbKernel = WarpedNorms + DeltaMask*Deltas.detach()
        Logits = -1*self.Temp*ProbKernel
        return Logits
    
    def forward(self, Output, Proxies, Labels):
        #Create batch-wise class masks off of Labels
        PairMasks = torch.zeros(Output.shape[0], Proxies.shape[0], device=Output.device)
        PairMasks[torch.arange(PairMasks.shape[0]),Labels[:,0]] = 1
        PairMasks[Labels[:,0] == -1,:] = 0
        
        #Final-Logits
        return self._embedding(Output, Proxies, PairMasks)

