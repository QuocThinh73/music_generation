import os

from torch.utils.data import DataLoader
from dataset import AudioDataset


def get_dataloader(audio_dir, json_dir, sample_rate, duration, n_mels, max_genres, test_amount):
    if not os.path.exists(audio_dir):
        raise ValueError(f"Audio directory {audio_dir} does not exist.")
    if not os.path.exists(json_dir):
        raise ValueError(f"JSON directory {json_dir} does not exist.")

    trainset = AudioDataset(audio_dir, json_dir, sample_rate, duration, n_mels, max_genres, testset_amount=test_amount)
    testset = trainset.testset

    if len(trainset) == 0:
        raise ValueError(f"No audio files found in {audio_dir}.")

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=test_amount, shuffle=False, num_workers=4)
    
    return trainloader, testloader