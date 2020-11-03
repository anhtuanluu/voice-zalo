import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet
from verification import *

root = '/home/tuan/Documents/Train-Test-Data/public-test/submit/'
log_dir = 'model_saved' # Where the checkpoints are saved
thres = 0.975
# Settings
use_cuda = True # Use cuda or not
embedding_size = 256 # Dimension of speaker embeddings
cp_num = 27 # Which checkpoint to use?
n_classes = 400 # How many speakers in training data?
test_frames = 200 # Split the test utterance 

# Load model from checkpoint
model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    
public_file = '/home/tuan/Documents/Train-Test-Data/public-test.csv'
public = pd.read_csv(public_file)
for au1, au2 in zip(public['audio_1'], public['audio_2']):
    au1 = root + au1[:-4] + '.p'
    au2 = root + au2[:-4] + '.p'
    best_spk = perform_verification(use_cuda, model, au1, au2, test_frames, thres)  
    public['label'] = best_spk
print(public)
df = pd.DataFrame(public)
df.to_csv (r'test.csv', index = False, header=True)