import librosa
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def load_and_resample_audio(file_path, target_sr=22050):
    """
    Load an audio file and resample it to the target sample rate.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sample rate. Default is 22050 Hz.
        
    Returns:
        tuple: Tuple containing the audio data and the sample rate.
    """
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio, target_sr

def audio_to_melspec(audio, sr, n_mels, n_fft=2048, hop_length=512, to_db=False):
    """
    Convert audio to Mel spectrogram.
    
    Args:
        audio (ndarray): Audio data.
        sr (int): Sample rate of the audio.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size. Default is 2048.
        hop_length (int): Hop length. Default is 512.
        to_db (bool): Whether to convert to decibels. Default is False.
        
    Returns:
        ndarray: Mel spectrogram.
    """
    spec = librosa.feature.melspectrogram(y=audio, 
                                          sr=sr, 
                                          n_fft=n_fft, 
                                          hop_length=hop_length, 
                                          win_length=None, 
                                          window="hann", 
                                          center=True, 
                                          pad_mode="reflect", 
                                          power=2.0, 
                                          n_mels=n_mels)
    
    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max)

    return spec

def normalize_melspec(melspec, norm_range=(0, 1)):
    """
    Normalize the Mel spectrogram to a specified range.
    
    Args:
        melspec (ndarray): Mel spectrogram.
        norm_range (tuple): Range for normalization. Default is (0, 1).
        
    Returns:
        ndarray: Normalized Mel spectrogram.
    """
    scaler = MinMaxScaler(feature_range=norm_range)
    melspec = melspec.T  # Transpose to shape (n_mels, time_steps)
    melspec_normalized = scaler.fit_transform(melspec)

    return melspec_normalized.T  # Transpose back to (time_steps, n_mels)

def denormalize_melspec(melspec_normalized, original_melspec, norm_range=(0, 1)):
    """
    Denormalize the Mel spectrogram to its original range.
    
    Args:
        melspec_normalized (ndarray): Normalized Mel spectrogram.
        original_melspec (ndarray): Original Mel spectrogram for reference.
        norm_range (tuple): Range for normalization. Default is (0, 1).
        
    Returns:
        ndarray: Denormalized Mel spectrogram.
    """
    scaler = MinMaxScaler(feature_range=norm_range)
    melspec = original_melspec.T  # Transpose to shape (n_mels, time_steps)
    scaler.fit(melspec)
    melspec_denormalized = scaler.inverse_transform(melspec_normalized.T)

    return melspec_denormalized.T  # Transpose back to (time_steps, n_mels)

def melspec_to_audio(melspec, sr, n_fft=2048, hop_length=512, n_iter=64):
    """
    Convert Mel spectrogram back to audio.
    
    Args:
        melspec (ndarray): Mel spectrogram.
        sr (int): Sample rate of the audio.
        n_fft (int): FFT window size. Default is 2048.
        hop_length (int): Hop length. Default is 512.
        n_iter (int): Number of iterations for Griffin-Lim algorithm. Default is 64.
        
    Returns:
        ndarray: Reconstructed audio.
    """
    if np.any(melspec < 0):
        melspec = librosa.db_to_power(melspec)

    audio_reconstructed = librosa.feature.inverse.mel_to_audio(melspec,
                                                            sr=sr, 
                                                            n_fft=n_fft, 
                                                            hop_length=hop_length, 
                                                            win_length=None, 
                                                            window="hann", 
                                                            center=True, 
                                                            pad_mode="reflect", 
                                                            power=2.0, 
                                                            n_iter=n_iter)
    
    return audio_reconstructed