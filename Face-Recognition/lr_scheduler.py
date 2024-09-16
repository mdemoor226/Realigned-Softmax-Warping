from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter

class _PolyScheduler_Original(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, last_epoch, False)

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]

#Modified PolyLRScheduler to work with a Proxy Learning Rate in addition to the normal one
class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, base_plr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.base_plr = base_plr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, last_epoch, False)

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha, self.base_plr * alpha]# for _ in self.optimizer.param_groups]


class MultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, switch=2, gammas=[0.5,0.1], last_epoch=-1, verbose=False):
        assert len(optimizer.param_groups) == len(gammas)
        self.switch_epoch = milestones[switch]
        self.milestones = Counter(milestones)
        self.gammas = gammas
        super(MultiStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch < self.switch_epoch:
            return [group['lr'] * self.gammas[0] ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups]
        #else:
        return [group['lr'] * self.gammas[i] ** self.milestones[self.last_epoch]
                for i,group in enumerate(self.optimizer.param_groups)]
        
        #print(result)
        #input()
        #return result

    def _get_closed_form_lr(self):
        #Should not be called
        raise
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]

