from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
import librosa
import numpy as np
from configs import cfg
from torchvision import models, transforms
import opensmile
import torchvision.transforms.functional as T
from tqdm import tqdm
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers import AutoConfig
import torch 
import opensmile

trans_rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

w2v2_config = AutoConfig.from_pretrained(cfg.w2v2_model,finetuning_task="wav2vec2_clf")
wav2vec2 = Wav2Vec2Model(w2v2_config).to(cfg.device)

def loader(path):

    audio_file = librosa.load(path, sr=None)[0]
    return audio_file

def make_embeddings(x):
    
    with torch.no_grad():
        try:
            w2v2_outputs = wav2vec2(torch.Tensor(x).unsqueeze(0).to(cfg.device), attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None)
            w2v2_hidden = w2v2_outputs[0]
            w2v2_final = torch.max(w2v2_hidden, dim=1)[0].squeeze(0).cpu()
        except:
            w2v2_final = torch.zeros(768)
            print('Error with sample; filling tensor with 0.')

        mfccs_imba = librosa.feature.mfcc(y=x, sr=cfg.sampling_rate, n_mfcc=40)
        mfccs_specs = T.resize(torch.tensor(mfccs_imba).unsqueeze(0), size = (cfg.img_size,cfg.img_size), antialias=None)
        mfccs_final = trans_rgb(mfccs_specs)

        mel_imba = librosa.feature.melspectrogram(y=x, n_mels = 128, sr=cfg.sampling_rate)
        mel_specs = T.resize(torch.tensor(mel_imba).unsqueeze(0), size = (cfg.img_size,cfg.img_size), antialias=None)
        mel_final = trans_rgb(mel_specs)

        all_llds = smile.process_signal(x, cfg.sampling_rate).to_numpy()
        llds_maxed = np.max(all_llds, axis = 0)
        lld_final = torch.Tensor(llds_maxed)

    return w2v2_final, mfccs_final, mel_final, lld_final

class MA_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, w2v2s, mfccs, mels, llds, labels):
        self.w2v2s = w2v2s
        self.mfccs = mfccs
        self.mels = mels
        self.llds = llds
        self.labels = labels

    def __getitem__(self, idx):

        w2v2 = self.w2v2s[idx]
        mfccs = self.mfccs[idx]
        mel = self.mels[idx]
        lld = self.llds[idx]
        labels = self.labels[idx]

        return w2v2, mfccs, mel, lld, labels
    
    def __len__(self):
        return len(self.labels)

def noise(data):
    noise_amp = np.random.uniform(0.03,0.04)*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y = data, rate = np.random.uniform(0.8,1.2))

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate):
    return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps = np.random.uniform(-4,4))

def augment_dataset(train_list, test_list):
    print('Creating dataloaders.\n')
    new_arr = []
    new_labels = []

    for i in tqdm(range(len(train_list))):
        if cfg.augment:
            aug_samples = [noise(train_list[i][0]), pitch(train_list[i][0], sampling_rate = cfg.sampling_rate), stretch(train_list[i][0]), shift(train_list[i][0])]
        else:
            aug_samples = []

        new_label_block = []
            
        for j in range(len(aug_samples)):
            new_label_block.append(train_list[i][1])
        
        new_arr.extend(aug_samples)         
        new_labels.extend(new_label_block)
    
    tr_labels_arr = []
    tr_w2v2_arr = []
    tr_mfcc_arr = []
    tr_mels_arr = []
    tr_llds_arr = []

    for sample in tqdm(train_list):

        w2v2_embed, mfcc, mel, lld = make_embeddings(sample[0])
        tr_w2v2_arr.append(w2v2_embed)
        tr_mfcc_arr.append(mfcc)
        tr_mels_arr.append(mel)
        tr_llds_arr.append(lld)
        tr_labels_arr.append(sample[1])
        
    # tr_w2v2_arr = torch.load('path/train_w2v2.pt')
    # tr_mfcc_arr = torch.load('path/train_mfcc.pt')
    # tr_mels_arr = torch.load('path/train_mels.pt')
    # tr_llds_arr = torch.load('path/train_llds.pt')

    tr_w2v2_arr = [np.float32(i) for i in tr_w2v2_arr]
    tr_mfcc_arr = [np.float32(i) for i in tr_mfcc_arr]
    tr_mels_arr = [np.float32(i) for i in tr_mels_arr]
    tr_llds_arr = [np.float32(i) for i in tr_llds_arr]

    for i in tqdm(range(len(new_arr))):
        
        w2v2_embed, mfcc, mel, lld = make_embeddings(new_arr[i])
        tr_w2v2_arr.append(w2v2_embed)
        tr_mfcc_arr.append(mfcc)
        tr_mels_arr.append(mel)
        tr_llds_arr.append(lld)
        tr_labels_arr.append(new_labels[i])

    tr_w2v2_arr = [np.float32(i) for i in tr_w2v2_arr]
    tr_mfcc_arr = [np.float32(i) for i in tr_mfcc_arr]
    tr_mels_arr = [np.float32(i) for i in tr_mels_arr]
    tr_llds_arr = [np.float32(i) for i in tr_llds_arr]

    te_labels_arr = []
    te_w2v2_arr = []
    te_mfcc_arr = []
    te_mels_arr = []
    te_llds_arr = []

    for sample in tqdm(test_list):
    
        w2v2_embed, mfcc, mel, lld = make_embeddings(sample[0])
        te_w2v2_arr.append(w2v2_embed)
        te_mfcc_arr.append(mfcc)
        te_mels_arr.append(mel)
        te_llds_arr.append(lld)
        te_labels_arr.append(sample[1])

    train_dataset = MA_Dataset(tr_w2v2_arr, tr_mfcc_arr, tr_mels_arr, tr_llds_arr, tr_labels_arr)
    test_dataset = MA_Dataset(te_w2v2_arr, te_mfcc_arr, te_mels_arr, te_llds_arr, te_labels_arr)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader

    