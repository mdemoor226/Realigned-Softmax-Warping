import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image

class Data(torch.utils.data.Dataset):
    def __init__(self, dataset=None, cfg=None, Train=True):
        assert dataset is not None
        assert cfg is not None
    
        crop_size = (cfg['crop_size'],cfg['crop_size'])
        resize_size = cfg['resize_size']
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.Transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size) if Train else lambda x: x,
            transforms.RandomHorizontalFlip() if Train else lambda x: x,
            transforms.Resize(resize_size) if not Train else lambda x: x,
            transforms.CenterCrop(crop_size) if not Train else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        self.Imgs = dataset

    def __len__(self):
        return len(self.Imgs)
    
    def __getitem__(self, idx):
        #Grab image and annotations
        Img = self.Imgs[idx]
        I = Image.open(Img['Image'])
        if len(list(I.split())) == 1 : I = I.convert('RGB') 
        #assert I.shape[-1] == 3

        ID = Img['Label']
        Img = self.Transform(I)
        
        return {'Data': Img, 'Labels': torch.Tensor([ID]).int()}

    def get_labels(self):
        return np.array([Img['Label'] for Img in self.Imgs])
