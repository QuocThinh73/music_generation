import os
import tqdm
import torch

from torch.utils.data import Dataset
from utils import genres_utils, audio_utils

class AudioDataset(Dataset):
    def __init__(self, data_dir, json_dir, sample_rate, duration, n_mels, testset_amount=10):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mp3")]
        self.json_dir = json_dir
        self.json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = sample_rate * duration
        self.n_mels = n_mels
        self.unique_genres, self.n_genres = genres_utils.get_unique_and_max_genres(json_dir)
        self.genres2idx, self.idx2genres = genres_utils.genres2idx_idx2genres(self.unique_genres)
        self.testset_amount = testset_amount

        audios = []
        for file_path, json_file_path in tqdm(zip(self.files, self.json_files), desc=f"Loading audio files in {data_dir}", unit="file", total=len(self.files)):
            audio, sr = audio_utils.load_and_resample_audio(file_path, target_sr=sample_rate)
            genres_list = genres_utils.load_and_get_genres(json_file_path)
            genres_tokens = genres_utils.tokenize(genres_list, self.genres2idx)
            genres_input = genres_utils.onehot_encode(genres_tokens, self.n_genres)
            genres_input = torch.tensor(genres_input, dtype=torch.long).unsqueeze(0)
            n_samples = len(audio)
            n_segments = n_samples // self.fixed_length

            for i in range(n_segments):
                start = i * self.fixed_length
                end = (i + 1) * self.fixed_length
                segment = audio[start:end]
                mel_spec = audio_utils.audio_to_melspec(segment, sr, self.n_mels, to_db=True)
                mel_spec_norm = audio_utils.normalize_melspec(mel_spec)
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                mel_spec_norm = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0)
                audios.append((mel_spec_norm, genres_input, mel_spec))

        self.audios = audios[:len(audios) - testset_amount]
        self.testset = audios[len(audios) - testset_amount:]

    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        mel_spec_norm, genres_input, mel_spec = self.audios[idx]
        return mel_spec_norm, genres_input, mel_spec