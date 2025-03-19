import numpy as np
import os
from configs import cfg
from tqdm import tqdm
import random

import torch
import torch.nn as nn

from transformers import AdamW
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,f1_score, ConfusionMatrixDisplay
import timm
import torchvision
import opensmile
from models import Model
from visualizer import visualize
from datasets import augment_dataset, loader
from torchvision.datasets import DatasetFolder

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(cfg.seed)
print('Anchored seed.\n')

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

train_list = DatasetFolder(root=cfg.train_path, loader=loader, extensions=cfg.extension, transform=None)
test_list = DatasetFolder(root=cfg.test_path, loader=loader, extensions=cfg.extension, transform=None)
print('Data loaded from folder.\n')

train_loader, test_loader = augment_dataset(train_list, test_list)
print('Dataloaders initialized.\n')

model = Model().to(cfg.device)
print('Model initialized.\n')

def train(model, train_loader, val_loader):

    highest = 0

    optimizer = AdamW(model.parameters(), lr = cfg.lr, eps=1e-8, weight_decay=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    print('Train and Val Phase.\n')

    for epoch in range(cfg.num_epochs):
        
        train_epoch_loss = 0
        model.train()

        for (w2v2, mfcc, mel, lld, labels) in tqdm(train_loader):

            optimizer.zero_grad()

            y_pred = []
            y_true = []

            _, logits = model(w2v2.to(cfg.device), mfcc.to(cfg.device), mel.to(cfg.device), lld.to(cfg.device))
            
            loss = criterion(logits, labels.cuda())

            loss.backward()
            optimizer.step()

            _, preds = logits.data.max(1)
            y_pred.extend(preds.cpu().detach().tolist())
            y_true.extend(labels.cpu().detach().tolist())
            train_epoch_loss += loss.item() / len(train_loader)

        train_epoch_acc = accuracy_score(y_true, y_pred)

        model.eval()
        y_pred = []
        y_true = []
        val_epoch_loss=0

        for (w2v2, mfcc, mel, lld, labels) in tqdm(val_loader):
            with torch.no_grad():

                _, logits = model(w2v2.to(cfg.device), mfcc.to(cfg.device), mel.to(cfg.device), lld.to(cfg.device))

                loss = criterion(logits, labels.cuda())

                _, preds = logits.data.max(1)
                y_pred.extend(preds.cpu().detach().tolist())
                y_true.extend(labels.cpu().detach().tolist())

                val_epoch_loss += loss.item() / len(val_loader)

        f1 = f1_score(y_true, y_pred, average='macro')
        val_epoch_acc = accuracy_score(y_true, y_pred)

        print(f"Epoch : {epoch+1} - train_loss : {train_epoch_loss:.4f} - train_acc: {train_epoch_acc:.4f} - val_loss : {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.4f} \n ")
        
        if f1 > highest:
            torch.save(model.state_dict(),'model.pth')
            highest = f1
            print('Best model saved.\n')

def test(model, test_loader):
    print('Test Phase.\n')
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    y_pred = []
    y_true = []

    for (w2v2, mfcc, mel, lld, labels) in tqdm(test_loader):
            with torch.no_grad():

                    _, logits = model(w2v2.to(cfg.device), mfcc.to(cfg.device), mel.to(cfg.device), lld.to(cfg.device))
                    _, preds = logits.data.max(1)
                    y_pred.extend(preds.cpu().detach().tolist())
                    y_true.extend(labels.cpu().detach().tolist())

    print('Test Results:\n')
    print(classification_report(y_true, y_pred))
    print('Confusion Matrix:\n')
    print(confusion_matrix(y_true, y_pred))

train(model, train_loader, test_loader)
test(model, test_loader)
visualize(model, test_loader)