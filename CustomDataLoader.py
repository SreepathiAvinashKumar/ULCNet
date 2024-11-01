import torch
from torch.utils.data import Dataset
import os 
import pandas as pd
import librosa 
# import stft_load

class MyDataset(Dataset):
    def __init__(self,csv,train_dir,label_dir):
        self.data = pd.read_csv(csv)
        self.train_dir = train_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        filename = self.data.iloc[index,0]
        train_audio_path = os.path.join(self.train_dir,filename)
        test_audio_path = os.path.join(self.label_dir,filename)
        train_audio_file, sr = librosa.load(train_audio_path)
        label_audio_file,sr = librosa.load(test_audio_path)
        # print(train_audio_file.shape)
        train_audio_file = train_audio_file[:160000]
        label_audio_file = label_audio_file[:160000]

        return train_audio_file,label_audio_file

    def __getname__(self,index):
        filename = self.data.iloc[index,0]
        return filename
    
    def give_path(self,index):
        filename = self.data.iloc[index,0]
        return os.path.join(self.train_dir,filename)
    


