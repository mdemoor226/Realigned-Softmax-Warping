from easydict import EasyDict as edict

config = edict()
config.network = "r100"
config.resume = True
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.Mwdecay = 0.0000715
config.Lwdecay = 0.000001275
config.batch_size = 256 
config.lr = 0.2
config.plr = 10.0
config.verbose = 10000
config.dali = False

#Enter path to data
#config.rec = "../../../../InsightFace_Pytorch/data/faces_emore/"
config.rec = "./data/faces_emore/"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 25
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", "calfw", "cplfw"]

#Softmax-Warping Parameters
config.k1 = 0.30
config.k2 = 1.075
config.alpha = 15.75
config.Temp = 3.15
#####################
config.fk1 = 0.20
config.fk2 = 1.03
config.falpha = 12.25
config.fTemp = 1.0

