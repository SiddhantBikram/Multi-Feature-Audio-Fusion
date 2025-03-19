import os
from yacs.config import CfgNode 

cfg = CfgNode()
cfg.batch_size = 16
cfg.prec = 'fp16'
cfg.seed = 510
cfg.lr = 1e-5
cfg.weight_decay = cfg.lr
cfg.momentum = 0.9
cfg.num_epochs = 5
cfg.device = 'cuda'
cfg.extension = '.wav' #.wav for ReCANVo
cfg.ratio = (0.85, 0.15) #Train-Test Ratio
cfg.train_path = 'path/to/your/train/dir'
cfg.test_path = 'path/to/your/test/dir'
cfg.n_classes = len(os.listdir(cfg.train_path)) #PyTorch DataFolder format
cfg.w2v2_model = 'facebook/wav2vec2-base-960h'
cfg.sampling_rate = 16000
cfg.img_size = 224
cfg.vit = 'vit_base_patch16_224'
cfg.drop = 0.3
cfg.cv_embed_size = 1000
cfg.augment = False