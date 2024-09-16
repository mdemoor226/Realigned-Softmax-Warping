import torch
import numpy as np
import sklearn.metrics.pairwise
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.io import loadmat
import os
import json


#From the ProxyNCA++ Repository
def load_config(config_name="config.json"):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) == list:
                config[k] = [eval(value) for value in config[k]]
            elif type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])

    eval_json(config)
    return config

def prepare_cub(TrainVal=True):
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    
    meta_data = "./datasets/cub2011/CUB_200_2011/images.txt"
    data = "./datasets/cub2011/CUB_200_2011/images"
    if not os.path.exists(data) or not os.path.exists(meta_data):
        raise ValueError("Error, please download dataset and place into dataset directory.")
    
    print("Preparing CUB...")
    with open(meta_data) as all_images:
        img_list = all_images.read().splitlines()
    
    Train, Val, Test = [], [], []
    for Img in img_list:
        img_idx, img_path = Img.split()
        
        #Organize Metadata into the DataStructure used in the program
        Img_path = os.path.join(data, img_path)
        Label = int(img_path.split(".")[0]) - 1
        if Label < 50:
            Train.append({'Image': Img_path, 'Label': Label})
        elif Label < 100:
            Val.append({'Image': Img_path, 'Label': Label})
        else:
            Test.append({'Image': Img_path, 'Label': Label})
        
    if not TrainVal:
        return Train, Val
    
    TrVal = Train.copy()
    TrVal.extend(Val)
    return TrVal, Test

def prepare_cars(TrainVal=True):
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    
    meta_data = "./datasets/cars196/cars_annos.mat"
    data = "./datasets/cars196/car_ims"
    if not os.path.exists(data) or not os.path.exists(meta_data):
        raise ValueError("Error, please download dataset and place into dataset directory.")
    
    print("Preparing Cars...")
    Anns = loadmat(meta_data)['annotations'][0]
    
    Train, Val, Test, = [], [], []
    for Ann in Anns:
        Img_file = os.path.join("./datasets/cars196", Ann[0][0])
        ID =  Ann[5][0][0] - 1
        if ID < 49:
            Train.append({'Image': Img_file, 'Label': ID})
        elif ID < 98:
            Val.append({'Image': Img_file, 'Label': ID})
        else:
            Test.append({'Image': Img_file, 'Label': ID})
        
    if not TrainVal:
        return Train, Val
    
    TrVal = Train.copy()
    TrVal.extend(Val)
    return TrVal, Test

'''
Ebay_info.txt: it contains the information about the whole dataset. Each line in the file contains "image_id class_id super_class_id path".
   image_id: index of the image (1 ~ 120,053). We have 120,053 images in the dataset in total.
   class_id: index of the item  (1 ~ 22,634). We have 22,634 items/products in the dataset in total.
   super_class_id: index of the category (1 ~ 12). We have 12 categories in the dataset in total.
   path: path of the image.

image_id class_id super_class_id path
1 1 1 bicycle_final/111085122871_0.JPG
2 1 1 bicycle_final/111085122871_1.JPG
3 1 1 bicycle_final/111085122871_2.JPG
'''

def prepare_sop(TrainVal=True):
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    
    trmeta_data = "./datasets/sop/Stanford_Online_Products/Ebay_train.txt"
    tsmeta_data = "./datasets/sop/Stanford_Online_Products/Ebay_test.txt"
    data = "/datasets/sop/Stanford_Online_Products"
    if not os.path.exists(data) or not os.path.exists(trmeta_data) or not os.path.exists(tsmeta_data):
        raise ValueError("Error, please download dataset and place into dataset directory.")
    
    print("Preparing SOP...")
    with open(trmeta_data) as all_images:
        img_list = all_images.read().splitlines()
    
    Metadata = img_list[1:]
    Train, Val = [], []
    for Img in Metadata:
        img_idx, class_id, _, img_path = Img.split()

        #Organize Metadata into the DataStructure used in the program
        Img_path = os.path.join(data, img_path)
        Label = int(class_id) - 1
        if Label < 5659:
            Train.append({'Image': Img_path, 'Label': Label})
        else:
            Val.append({'Image': Img_path, 'Label': Label})
        
    if not TrainVal:
        return Train, Val
    
    TrVal = Train.copy()
    TrVal.extend(Val)

    #Do the same thing but with the Test MetaData
    with open(tsmeta_data) as all_images:
        img_list = all_images.read().splitlines()
    
    Metadata = img_list[1:]
    Test = list()
    for Img in Metadata:
        img_idx, class_id, _, img_path = Img.split()

        #Organize Metadata into the DataStructure used in the program
        Img_path = os.path.join(data, img_path)
        Label = int(class_id) - 1
        Test.append({'Image': Img_path, 'Label': Label})

    return TrVal, Test

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, dataset, n_classes, n_samples, batch_size):
        self.labels = dataset.get_labels()
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            UnusedInds = []
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                
                UnusedInds.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:])
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

#Borrowed from ProxyNCA++ repository
def assign_by_euclidian_at_k(X, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = sklearn.metrics.pairwise.pairwise_distances(X)

    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1] 
    return np.array([[T[i] for i in ii] for ii in indices])

#Borrowed from ProxyNCA++ repository
def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))
 
def CalculateRecall(Outputs, Labels, KVals):
    """
    Outputs : [nb_samples, nb_features] (nb_features = dimension of embedding space)
    Labels : [nb_samples]
    KVals  : [List of k thresholds to evaluate at]
    """
    Preds = assign_by_euclidian_at_k(Outputs, Labels, max(KVals))
    Recalls = list()
    for k in KVals:
        kRecall = calc_recall_at_k(Labels, Preds, k)
        Recalls.append(kRecall)

    return Recalls

def CalculateNMI(Outputs, Labels):
    num_classes = len(np.unique(Labels))
    Clusters = KMeans(num_classes).fit(Outputs).labels_
    return normalized_mutual_info_score(Clusters, Labels, average_method='geometric')

