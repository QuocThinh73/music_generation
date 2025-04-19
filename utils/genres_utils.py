import json
import os
import numpy as np


def load_and_get_genres(json_path):
    """
    Load genres from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file.
        
    Returns:
        list: List of genres.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data.get('genres', [])


def get_unique_and_max_genres(json_dir):
    """
    Load all genres from JSON files in the specified directory and return unique genres and their count.

    Args:
        json_dir (str): Directory containing JSON files.

    Returns:
        tuple: A set of unique genres and the maximum number of genres.
    """
    all_genres = []

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            genres = load_and_get_genres(json_path)
            all_genres.extend(genres)

    unique_genres = set(all_genres)
    max_genres = len(unique_genres)

    return unique_genres, max_genres

def genres2idx_idx2genres(unique_genres):
    """
    Create mappings from genres to indices and vice versa.

    Args:
        unique_genres (set): Set of unique genres.

    Returns:
        tuple: Two dictionaries, one mapping genres to indices and the other mapping indices to genres.
    """

    genres2idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    idx2genres = {idx: genre for genre, idx in genres2idx.items()}
    
    return genres2idx, idx2genres

def tokenize(genres, genres2idx):
    """
    Tokenize a list of genres using the provided mapping.
    
    Args:
        genres (list): List of genres to tokenize.
        genres2idx (dict): Mapping from genres to indices.
        
    Returns:
        list: List of indices corresponding to the genres.
    """
    return [genres2idx[genre] for genre in genres if genre in genres2idx]

def detokenize_tolist(tokens, idx2genres):
    """
    Detokenize a list of indices back to genres using the provided mapping.
    
    Args:
        tokens (list): List of indices to detokenize.
        idx2genres (dict): Mapping from indices to genres.
        
    Returns:
        list: List of genres corresponding to the indices.
    """
    return [idx2genres[token] for token in tokens if token in idx2genres]

def onehot_encode(tokens, n_genres):
    """
    One-hot encode a list of genre indices.
    
    Args:
        tokens (list): List of genre indices.
        n_genres (int): Number of genres.
        
    Returns:
        np.ndarray: One-hot encoded vector.
    """
    onehot = np.zeros(n_genres)
    onehot[tokens] = 1
    return onehot

def onehot_decode(onehot):
    """
    Decode a one-hot encoded vector to its indices.
    
    Args:
        onehot (np.ndarray): One-hot encoded vector.
        
    Returns:
        list: List of indices where the one-hot vector is 1.
    """
    return [idx for idx, value in enumerate(onehot) if value == 1]