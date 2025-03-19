import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from configs import cfg

class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()

class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale

class Model(nn.Module):
  def __init__(self):
        super().__init__()

        self.num_labels = cfg.n_classes
        self.w2v2_size = 768
        self.lld_size = 25
        self.mid_size = 24
        
        self.cv_model = timm.create_model(cfg.vit, pretrained=True).to(cfg.device)
        self.cv_embed_size = cfg.cv_embed_size
        self.drop = cfg.drop

        self.w2v2_proj = nn.Linear(self.w2v2_size, self.mid_size)
        self.mel_proj = nn.Linear(self.cv_embed_size, self.mid_size)
        self.lld_proj = nn.Linear(self.lld_size, self.mid_size)
        self.final_proj = nn.Linear(self.mid_size*3, self.mid_size)
        self.drop = nn.Dropout(self.drop)

        self.classifier = CosineClassifier(self.mid_size, self.num_labels)
        
        for param in self.cv_model.parameters():
            param.requires_grad = True

  def forward(self, w2v2_final, mfcc_final, mel_final, lld_final):

        mel_final = self.cv_model(mel_final)

        x1 = self.drop(F.relu(self.w2v2_proj(F.normalize(w2v2_final))))
        x2 = self.drop(F.relu(self.mel_proj(F.normalize(mel_final))))
        x3 = self.drop(F.relu(self.lld_proj(F.normalize(lld_final))))
        x4 = torch.cat([x1, x2,x3], axis = 1)
        x5 = self.final_proj(x4)
    
        logits = self.classifier(x5)
        
        return x5, logits