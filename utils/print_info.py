import Ipython.display as ipd
import matplotlib.pyplot as plt


def display_audio_files(reconstructed_audio, sr, title="", original_audio=None):
    if original_audio is not None:
        print("Original Audio:")
        ipd.display(ipd.Audio(original_audio, rate=sr))
        print("Reconstructed Audio:")
    else:
        print(title)

    ipd.display(ipd.Audio(reconstructed_audio, rate=sr))

def show_spectogram(spectogram, title="Mel-Spectogram", denormalize=False, is_numpy=False):
    if not is_numpy:
        spectogram = spectogram.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    if denormalize:
        plt.imshow(spectogram, aspect='auto', origin='lower', cmap='viridis')
    else:
        plt.imshow(spectogram, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar()
    plt.show()