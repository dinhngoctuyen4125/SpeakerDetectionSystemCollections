
import os
import glob
import numpy as np
import librosa
import pyroomacoustics as pra
import scipy.signal as sps

import glob
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    EarlyStopping, 
    Callback
)

from sklearn.metrics import roc_curve

class HardAugmentor:
    def __init__(self, noise_dir=None, sr=16000):
        self.sr = sr
        self.noise_dir = noise_dir
        self.noise_files = glob.glob(os.path.join(noise_dir, '**/*.wav'), recursive=True) if noise_dir else []
        print(f"Đã load {len(self.noise_files)} file MUSAN noise")

    def _normalize(self, wav):
        return wav / (np.max(np.abs(wav)) + 1e-9)
    
    def add_musan_noise(self, wav):
        noise_path = np.random.choice(self.noise_files)
        noise, _ = librosa.load(noise_path, sr=self.sr)
        if len(noise) < len(wav):
            noise = np.tile(noise, int(np.ceil(len(wav) / len(noise))))
        start = np.random.randint(0, len(noise) - len(wav) + 1)
        noise = noise[start:start + len(wav)]
        
        # SNR nhẹ nhàng hơn: 10-20dB thay vì 5-12dB
        snr_db = np.random.uniform(10, 20)
        signal_power = np.mean(wav ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        scale = np.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))
        return self._normalize(wav + scale * noise)

    def add_rir_reverb(self, wav):
        # Giữ nguyên logic cũ nhưng có thể giảm rt60 nếu muốn nhẹ hơn
        room_dim = [np.random.uniform(6, 12) for _ in range(3)]
        rt60 = np.random.uniform(0.3, 0.7) # Nhẹ hơn bản cũ (0.5-1.0)
        src = [room_dim[0]/2, room_dim[1]/2, 1.5]
        mic = [room_dim[0]/2 + np.random.uniform(-2, 2), room_dim[1]/2 + np.random.uniform(-2, 2), 1.5]
        try:
            e_absorption, _ = pra.inverse_sabine(rt60, room_dim)
            room = pra.ShoeBox(room_dim, fs=self.sr, materials=pra.Material(e_absorption), max_order=3)
            room.add_source(src, signal=wav)
            room.add_microphone_array(pra.MicrophoneArray(np.array([mic]).T, self.sr))
            room.compute_rir()
            rir = room.rir[0][0]
            reverb = sps.fftconvolve(wav, rir)[:len(wav)]
            return self._normalize(reverb)
        except: return wav

    def change_device(self, wav):
        # Giữ nguyên tên def
        device = np.random.choice(['mobile', 'cheap_mic'])
        if device == 'mobile':
            wav_8k = librosa.resample(wav, orig_sr=self.sr, target_sr=8000)
            wav_back = librosa.resample(wav_8k, orig_sr=8000, target_sr=self.sr)
            return self._normalize(wav_back)
        return wav

    def change_environment(self, wav):
        return wav # Có thể bypass hoặc làm nhẹ đi

    def apply_augmentation(self, wav, aug_type):
        # Xác suất 50% không augment để giữ chất lượng gốc
        if np.random.rand() > 0.5: return wav
        aug_map = {
            'musan_noise': self.add_musan_noise,
            'rir_reverb': self.add_rir_reverb,
            'device': self.change_device,
            'environment': self.change_environment
        }
        return aug_map.get(aug_type, lambda x: x)(wav)

def compute_fbank(waveform, sample_rate=16000, num_mel_bins=80,frame_length=25,frame_shift=10):
    feat = kaldi.fbank(waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate,
        window_type='hamming')

    feat = feat - torch.mean(feat, 0)
    return feat

class VoxVietDataset(Dataset):
    def __init__(self, data_list, sample_rate=16000, duration_seconds=3, train_mode=True, augmentor=None, aug_dict=None, use_augmentation=True):
        # data_list lúc này là: {label: [path1, path2, ...]}
        self.label_to_paths = data_list
        self.labels = list(data_list.keys())
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_seconds)
        self.augmentor = augmentor
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.labels) * 50 # Giả lập số lượng sample mỗi epoch

    def __getitem__(self, idx):
        label = self.labels[idx % len(self.labels)]
        paths = self.label_to_paths[label]
        
        # Chọn ngẫu nhiên 1 file của speaker này
        wav_path = np.random.choice(paths)
        
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        # Random Crop 3s (Logic: Không bỏ phí data)
        if waveform.size(1) > self.target_len:
            start = np.random.randint(0, waveform.size(1) - self.target_len)
            waveform = waveform[:, start:start + self.target_len]
        else:
            # Nếu ngắn hơn 3s thì Pad (No Waste)
            waveform = F.pad(waveform, (0, self.target_len - waveform.size(1)))

        waveform_np = waveform[0].numpy()
        if self.use_augmentation and self.augmentor:
            aug_type = np.random.choice(['musan_noise', 'rir_reverb', 'device'])
            waveform_np = self.augmentor.apply_augmentation(waveform_np, aug_type)

        waveform = torch.FloatTensor(waveform_np).unsqueeze(0)
        fbank = compute_fbank(waveform, sample_rate=self.sample_rate)
        return fbank, label

class VoxVietTrialsDataset(Dataset):
    def __init__(self, trials_list, val_dir, sample_rate=16000, duration_seconds=3):
        self.trials_list = trials_list
        self.val_dir = val_dir
        self.sample_rate = sample_rate

    def __len__(self): return len(self.trials_list)

    def _process_full_audio(self, rel_path):
        # Fix path theo yêu cầu
        full_path = os.path.join(self.val_dir, "wav", rel_path)
        waveform, sr = torchaudio.load(full_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform

    def __getitem__(self, idx):
        label, p1, p2 = self.trials_list[idx]
        wav1 = self._process_full_audio(p1)
        wav2 = self._process_full_audio(p2)
        return wav1, wav2, torch.tensor(label, dtype=torch.float)

def val_collate_fn(batch):
    return batch # Trả về list vì độ dài audio khác nhau

class VoxVietDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, valid_dir, trials_path, batch_size=32, num_workers=2, aug_ratio=0.5, noise_dir=None, use_augmentation=True): 
        super().__init__()
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.trials_path = trials_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_dir = noise_dir
        self.use_augmentation = use_augmentation
        
    def setup(self, stage=None):
        # Group file theo speaker ID
        self.train_data = {}
        speaker_dirs = sorted([d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))])
        self.spk2id = {spk: i for i, spk in enumerate(speaker_dirs)}
        self.num_classes = len(self.spk2id)

        for spk in speaker_dirs:
            paths = glob.glob(os.path.join(self.train_dir, spk, "**/*.wav"), recursive=True)
            if paths:
                self.train_data[self.spk2id[spk]] = paths

        # Load Trials
        self.trials_list = []
        
        with open(self.trials_path, "r") as f:
            for line in f:
                line = line.strip()
        
                label, rest = line.split(" ", 1)
                p1, p2 = rest.rsplit(" ", 1)
        
                label = int(label)
        
                self.trials_list.append((label, p1, p2))

        print("Example trial:", self.trials_list[0])

    def train_dataloader(self):
        # Mỗi batch sẽ chọn ngẫu nhiên speaker, mỗi speaker lấy >= 2 mẫu nếu batch_size cho phép
        # Đơn giản nhất là shuffle trong Dataset item logic
        augmentor = HardAugmentor(noise_dir=self.noise_dir) if self.use_augmentation else None
        dataset = VoxVietDataset(self.train_data, train_mode=True, augmentor=augmentor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
    
    def val_dataloader(self):
        dataset = VoxVietTrialsDataset(self.trials_list, self.valid_dir)
        return DataLoader(dataset, batch_size=1, collate_fn=val_collate_fn)