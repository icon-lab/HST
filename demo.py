#demo

from model import hst_model as Model
import torch
import librosa
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import warnings
warnings.simplefilter("ignore")

class Net(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, xb):
        return self.model(xb)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--audio_path", type=str, default='', help="audio path")
args = parser.parse_args()

print("Configure model and load weights...")
hst_model = Model.HSTModel(d=96, num_blocks=[1,1,9,1], num_attention_heads=[3,6,12,24])
hst_model = Net(hst_model)
hst_model.load_state_dict(torch.load('./trained_hst.pth'))
hst_model.eval()

print("Convert audio file to spectrogram...")
#convert the audio of a cough/breath to spectrogram
cmap = plt.get_cmap('gray')
plt.figure(figsize=(8,8))
SR = 22050  
FRAME_LEN = int(SR / 10) 
HOP = int(FRAME_LEN / 2)  
sample_name = args.audio_path
y, sr = librosa.load(sample_name,sr=SR, mono=True, offset=0.0, duration=None)
y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
duration = librosa.get_duration(y=y, sr=sr)
plt.specgram(y, NFFT=2048, Fs=SR, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
plt.axis('off');
plt.savefig("./spectrogram.jpg")
plt.close()

img_path = "./spectrogram.jpg"
img = Image.open(img_path)
trms = T.Compose([
                        T.Resize((224,224)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5], [0.5])
                        ])
test_input = trms(img)
test_input = test_input.unsqueeze(0)
test_output = hst_model(test_input)
prediction = torch.argmax(test_output,dim=1)
print("Test output: ")
if prediction == 1:
  print("healthy")
else:
  print("covid")
