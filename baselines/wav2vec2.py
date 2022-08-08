import torch
import argparse
from torch import nn as nn
import sys
from sklearn.metrics import roc_auc_score
import random
import os
import numpy as np
import librosa as lb
from transformers import Wav2Vec2FeatureExtractor
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import Wav2Vec2Model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default='', help="train or val dataset for the task")

args = parser.parse_args()

path_task = args.dataset  # path to task directory

def seed_env(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def metric(preds, targets):
    TP = ((preds == 1) & (targets == 1)).sum()
    TN = ((preds == 0) & (targets == 0)).sum()
    FN = ((preds == 0) & (targets == 1)).sum()
    FP = ((preds == 1) & (targets == 0)).sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    return p, r, F1

def accuracy(outputs, labels):
    score = F.softmax(outputs, dim=0)
    preds = (score > 0.5).float()
    accuracy_tensor = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    targets = labels.cpu().numpy()
    scores = score.cpu().numpy()
    preds = preds.cpu().numpy()

    TP = ((preds == 1) & (targets == 1)).sum()
    TN = ((preds == 0) & (targets == 0)).sum()
    FN = ((preds == 0) & (targets == 1)).sum()
    FP = ((preds == 1) & (targets == 0)).sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    return accuracy_tensor, p, r, F1, scores, preds


class Wav2Vec2BasedModel(nn.Module):
    def __init__(self, proj_size):
        super(Wav2Vec2BasedModel, self).__init__()
        self.transformer = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.projector = nn.Linear(self.transformer.config.hidden_size, proj_size)
        self.classifier = nn.Linear(proj_size, 1)

    def forward(self, input_features):
        transformer_out = self.transformer(input_features)[0]
        hidden_layer_out = self.projector(transformer_out)
        pooled_out = hidden_layer_out.mean(dim=1)
        logits = self.classifier(pooled_out)
        logits = logits.view(logits.size(0))
        return logits


class Label(nn.Module):
    def train(self, train_data, test_data, seed, n_epochs = 50, lr=1e-3, weight_decay=1e-6):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for i in range(n_epochs):
            for (batch_samples, batch_labels) in train_data:
                logits = self.model.forward(batch_samples)
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            if i % 5 == 0:
                print("Epoch", i + 1," Loss:", loss)
                with torch.no_grad():
                    scorez = []
                    all_scores = []
                    all_targets = []
                    all_preds = []
                    for batch in test_data:
                        features, targetz = batch
                        out = self.model.forward(features)  # Generate predictions
                        score, precision, recall, F1, scores, preds = accuracy(out, targetz)
                        targetz = targetz.cpu().numpy()
                        all_scores.append(scores)
                        all_targets.append(targetz)
                        all_preds.append(preds)
                        scorez.append(score)
                    scr = np.concatenate(all_scores, axis=None)
                    trgts = np.concatenate(all_targets, axis=None)
                    prds = np.concatenate(all_preds, axis=None)
                    prec, rec, f1 = metric(prds, trgts)
                    auc = roc_auc_score(trgts, scr)
                    print({'Test RESULTS --> accuracy': np.array(scorez).mean(), 'precision': prec,
                           'recall': rec, 'F1': f1, 'auc': auc})




class CovidClassifier(Label):
    def __init__(self, proj_size = 256):
        super().__init__()
        self.model = Wav2Vec2BasedModel(proj_size=proj_size)

    def forward(self, xb):
        return self.model(xb)


# Default sampling rate for Wav2Vec2
SAMPLING_RATE = 16000


class FeatureDataset(Dataset):
    def __init__(self, data, label, **kwargs):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        _x = self.data[index]
        _y = self.label[index]
        return _x, _y

# Reading the wav files from a directory

wavs_covid = [lb.effects.trim(lb.load(path_task + "/covid/" + filename, sr=SAMPLING_RATE)[0])[0] for filename in os.listdir(path_task + "/covid") if '.DS_Store' not in filename]
wavs_healthy = [lb.effects.trim(lb.load(path_task + "/healthy/" + filename, sr=SAMPLING_RATE)[0])[0] for filename in os.listdir(path_task + "/healthy")]

c_val_size = int(len(wavs_covid)*0.2)
c_train_size = len(wavs_covid) - c_val_size

h_val_size = int(len(wavs_healthy)*0.2)
h_train_size = len(wavs_healthy) - h_val_size

wavs_train = wavs_covid[:c_train_size] + wavs_healthy[:h_train_size]
wavs_test = wavs_covid[c_train_size:] + wavs_healthy[h_train_size:]

# labels
labels_cov = torch.ones(len(wavs_covid))
labels_healthy = torch.zeros(len(wavs_healthy))

labels_train = torch.cat([labels_cov[:c_train_size], labels_healthy[:h_train_size]])
labels_test = torch.cat([labels_cov[c_train_size:], labels_healthy[h_train_size:]])
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
input_features_list_train = processor(wavs_train, do_normalize=True, padding=True, sampling_rate=SAMPLING_RATE).input_values
input_features_list_test = processor(wavs_test, do_normalize=True, padding=True, sampling_rate=SAMPLING_RATE).input_values

sec = 3  # to crop to 3 seconds

for k in range(len(input_features_list_test)):
    input_features_list_test[k] = input_features_list_test[k]
    if len( input_features_list_test[k]) < 16000 * sec:
        input_features_list_test[k] = np.array(list( input_features_list_test[k]) + [0] * (16000 * sec - len( input_features_list_test[k])))
    else:
        input_features_list_test[k] =  (input_features_list_test[k])[:16000 * sec]
input_features_list_test = torch.as_tensor(input_features_list_test)

for n in range(len(input_features_list_train)):
    input_features_list_train[n] = input_features_list_train[n]
    if len(input_features_list_train[n]) < 16000 * sec:
        input_features_list_train[n] = np.array(list(input_features_list_train[n]) + [0] * (16000 * sec - len(input_features_list_train[n])))
    else:
        input_features_list_train[n] = input_features_list_train[n][:16000 * sec]
input_features_list_train = torch.as_tensor(input_features_list_train)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
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


device = get_default_device()

for seed in [1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002]:
    seed_env(seed)

    train_data = DataLoader(FeatureDataset(input_features_list_train, labels_train), batch_size=32, shuffle=True)
    test_data = DataLoader(FeatureDataset(input_features_list_test, labels_test), batch_size=3, shuffle=True)

    classifier = CovidClassifier()
    device = get_default_device()
    classifier = to_device(classifier, device)
    torch.cuda.empty_cache()
    train_dl = DeviceDataLoader(train_data, device)
    val_dl = DeviceDataLoader(test_data, device)
    classifier.train(train_dl, val_dl, seed)
