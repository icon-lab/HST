# -*- coding: utf-8 -*-
"""
This is the training pipeline for the Ensemble = CNN + AST baseline.
For a detailed information about AST, please check the original implementation https://github.com/YuanGongND/ast.
"""
import argparse
from models import ast_models
import sys
import torch
import torchvision
import torchaudio
import random
import numpy as np
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm.autonotebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
import os
import timm
import torchvision.models as models
from torch.nn.functional import normalize

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default='', help="train or val dataset for the task")
parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

args = parser.parse_args()
img_path = args.dataset

def seed_env(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_env(args.seed)

batch_size = 8
image_size = (224,224)

#for cnn
train_trms = T.Compose([
                        T.Resize(image_size),
                        T.RandomRotation(20),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5])
                        ])

#for ast
train_trms1 = T.Compose([T.Grayscale(num_output_channels=1),
                        T.Resize(image_size),
                        T.RandomRotation(20),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5])
                        ])

train_data = torchvision.datasets.ImageFolder(root = img_path, transform = train_trms)
train_data1 = torchvision.datasets.ImageFolder(root = img_path, transform = train_trms1)


val_size = int(len(train_data)*0.2)
train_size = len(train_data) - val_size

def accuracy(outputs, labels):
    with torch.no_grad():
        score = F.softmax(outputs, dim=1)
        preds = torch.argmax(score,dim=1)
        accuracy_tensor = torch.tensor(torch.sum(preds == labels).item()/len(preds))
        targets = labels.cpu().numpy()
        scores = score.cpu().numpy()[:,1]
        preds = preds.cpu().numpy()
        
        TP = ((preds == 1) & (targets == 1)).sum()
        TN = ((preds == 0) & (targets == 0)).sum()
        FN = ((preds == 0) & (targets == 1)).sum()
        FP = ((preds == 1) & (targets == 0)).sum()
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        AUC = roc_auc_score(targets,scores)
    return accuracy_tensor,p,r,F1,AUC


class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch, batch1):
        images, targets = batch 
        images1, targets1 = batch1 
        images1 = torch.squeeze(images1)
        out = self(images, images1)
        loss = F.cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch, batch1):
        images, targets = batch 
        images1, targets1 = batch1 
        images1 = torch.squeeze(images1)
        out = self(images, images1)
        loss = F.cross_entropy(out, targets)         # Calculate loss
        score,precision,recall,F1,auc = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() , 'precision': precision
               ,'recall': recall,'F1': F1,'auc': auc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        precision = outputs[0]['precision']
        recall = outputs[0]['recall'] 
        F1 = outputs[0]['F1']
        auc =outputs[0]['auc'] 
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()},precision,recall,F1,auc
    
    def epoch_end(self, epoch, result,precision,recall,F1,auc):
        print("Epoch [{}], val_loss: {:.4f}, val_score: {:.4f}, precision: {:.4f}, recall: {:.4f}, F1: {:.4f} \
              ,AUC: {:.4f}".format(epoch, result['val_loss'], result['val_score'],precision,recall,F1,auc))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class AST(MultilabelImageClassificationBase):
    def __init__(self):
        super(AST, self).__init__()
        # Use a pretrained model
        self.model = ast_models.ASTModel(label_dim=2, \
                                         fstride=16, tstride=16, \
                                         input_fdim=224, input_tdim=224, \
                                         imagenet_pretrain=True, audioset_pretrain=False, \
                                         model_size='base224')
        self.featureast = self.model.v.head.in_features
        self.model.v.head = Identity()
        self.model.v.head_dist = Identity()
        self.model.mlp_head = Identity()

        
    def forward(self, xb):
        return normalize(self.model(xb))

class CNN(MultilabelImageClassificationBase):
    def __init__(self):
        super(CNN, self).__init__()
        # Use a pretrained model
        self.network = models.resnet34()
        self.num_ftrs = self.network.fc.in_features
        self.network.fc = Identity()
    
    def forward(self, xb):
        return normalize(self.network(xb), dim=1)


class Ensemble(MultilabelImageClassificationBase):
    def __init__(self, ResNet, Vit):
        super(Ensemble, self).__init__()
        self.CNN = CNN()
        self.AST = AST()
        self.classifier = nn.Linear(self.CNN.num_ftrs+self.AST.featureast, 2)
        self.sig = nn.Sigmoid()
        
    def forward(self, images, images1):
        #images1 = images
        x1 = self.CNN(images)
        x2 = self.AST(images1)
        x3 = torch.cat((x1, x2), dim=1)
        x = self.classifier(self.sig(x3))
        return x

def evaluate(model,val_loader,val_loader1):
    outputs = [model.validation_step(batch,batch1) for batch,batch1 in zip(val_loader,val_loader1)]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, train_loader1, val_loader1, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    val_losses = []
    val_accuracies = []
    torch.cuda.empty_cache()
    history = []
    precisions = []
    recalls = []
    F1s = []
    AUCs = []
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
      
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        lrs = []
        for batch, batch1 in zip(train_loader, train_loader1):
            torch.autograd.set_detect_anomaly(True)
            loss = model.training_step(batch, batch1)
            loss.backward()
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result,precision,recall,F1,auc = evaluate(model, val_loader, val_loader1)
        result['lrs'] = lrs
        model.epoch_end(epoch, result,precision,recall,F1,auc)
        val_losses.append(result['val_loss'])
        val_accuracies.append(result['val_score'])
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        AUCs.append(auc)
        history.append(result)
    return history,val_losses,val_accuracies,precisions,recalls,F1s,AUCs

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device1 = get_default_device()
torch.cuda.empty_cache()

epochs = 100
max_lr = 1e-3
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.AdamW
batch_size = 6

#for cnn
train_ds, val_ds = random_split(train_data, [train_size,val_size])
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, 4, num_workers=0, pin_memory=True)
train_dl = DeviceDataLoader(train_dl, device1)
val_dl = DeviceDataLoader(val_dl, device1)

#for ast
train_ds1, val_ds1 = random_split(train_data1, [train_size,val_size])
train_dl1 = DataLoader(train_ds1, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl1 = DataLoader(val_ds1, 4, num_workers=0, pin_memory=True)
train_dl1 = DeviceDataLoader(train_dl1, device1)
val_dl1 = DeviceDataLoader(val_dl1, device1)

cnn = CNN()
ast = AST()
model = Ensemble(cnn, ast)
model = to_device(model, device1)
history = [evaluate(model, val_dl, val_dl1)]
history, val_loss, val_acc,  precisions, recalls, F1s, AUCs = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, train_dl1, val_dl1, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)
best_epoch = np.argmax(AUCs)
best_AUC = AUCs[best_epoch]
best_acc = val_acc[best_epoch]
best_precision = precisions[best_epoch]
best_recall = recalls[best_epoch]
best_F1 = F1s[best_epoch]
print("Highest AUC of %2f, achieved at epoch %d" % (best_AUC,best_epoch+1))
print("Accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, F1: {:.4f} \
              ,AUC: {:.4f}".format(best_acc,best_precision,best_recall,best_F1,best_AUC))
