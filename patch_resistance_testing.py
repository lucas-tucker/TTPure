import whisper, torch, torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np
from IPython.display import Audio, display
import torchaudio.functional as F
import torchaudio.transforms as T
from matplotlib.patches import Rectangle
from torchaudio.utils import download_asset
import librosa, random
import matplotlib.pyplot as plt
import glob, os
from jiwer import wer
from tqdm import tqdm
import pandas as pd

WHISPER_MODEL = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
CUTOFF = 30 * 16000  # 30 second cutoff size
NOISE_LEVEL = 100

def apply_tt_patch(audio, verbose=True):

  segment_size = int(16000 * (30 - 0.64)) # Recall we use 16kHz sampling

  # Split the audio into 30s - 0.64s size segments
  num_segments = int(len(audio) // segment_size)
  seek = 0
  adv_audio = torch.zeros((1,)).to(DEVICE)

  # Embed a patch in each part of the audio
  for patch_number in range(num_segments + 1):
      # Embed the patch randomly into each segment
      breakpoint = random.randint(0, min(segment_size, len(audio[seek:])))
      if verbose:
        print(f"Inserting patch {patch_number} at {(seek + breakpoint) / 16000} seconds")
      adv_audio = torch.cat((adv_audio,
                              audio[seek:seek + breakpoint],
                              tt_patch,
                              audio[seek + breakpoint:seek + segment_size]), dim=0)
      seek += segment_size

  # SKIPPED: Embed a final patch in the rest of the audio
  # adv_audio = torch.cat((adv_audio, longest_audio[seek:]), dim=0)
  if verbose:
    with torch.no_grad():
        original_transcription = model.transcribe(audio)["text"]
        adversarial_transcription = model.transcribe(adv_audio)["text"]

    print()
    display(Audio(audio, autoplay=False, rate=16000))
    print("Benign transcription:")
    print(original_transcription)
    print()

    display(Audio(adv_audio, autoplay=False, rate=16000))
    print("Adversarial transcription:")
    print(adversarial_transcription)
    print()

  return adv_audio

print(f"Using device: {DEVICE}")

### Load Whisper model
model = whisper.load_model(WHISPER_MODEL).to(DEVICE)
print(f"Loaded Whisper {WHISPER_MODEL} model")

### Load the patch attack
tt_patch_attack_path = "/home-nfs/lwtucker/TT-Patch/tucker_attacks/tt_patch.th" # Replace as needed
try:
    patch_data = torch.load(tt_patch_attack_path, map_location=torch.device(DEVICE))
    tt_patch = patch_data["audio_attack_segment"]
    print(f"Loaded tt patch attack segment of shape {tt_patch.shape}")
except:
    print("Warning: Could not load embed attack file. Using random noise for demonstration.")
    tt_patch = torch.randn(8000).to(DEVICE) * 0.01

test_dir = "/home-nfs/lwtucker/TT-Patch/tucker_attacks/src/data/librispeech-long/test-clean" # Replace as needed
# Grab the actual audio files, of which there should be 810; assume 16kHz sampling
flac_files = glob.glob(os.path.join(test_dir, "**", "**", "*.flac"), recursive=True)

segment_size = int(16000 * (30 - 0.64))
wers = []

testing_data_size = 5 # 800
THRESHOLD = 15
num_empty_strings = 0
num_under_threshold = 0

for path in flac_files[:testing_data_size]:
    audio_sample, sr = torchaudio.load(path)
    audio_sample = audio_sample.squeeze().to(DEVICE) # Now a whisper-readable 1D tensor
    adv_audio = apply_tt_patch(audio=audio_sample, verbose=False)
    adv_audio += NOISE_LEVEL
    with torch.no_grad():
        original_transcription = model.transcribe(audio_sample)["text"]
        adversarial_transcription = model.transcribe(adv_audio)["text"]

    num_under_threshold += len(adversarial_transcription) <= THRESHOLD
    num_empty_strings += len(adversarial_transcription) == 0

    wers.append(wer(original_transcription.lower().strip(), adversarial_transcription.lower().strip()))
    print(f"Original transcript: {original_transcription}")
    print(f"Adversarial transcript: {adversarial_transcription}")
    print(f"Avg WER thus far is {np.mean(wers)}")

plt.hist(wers, density=True, bins=10)
plt.ylabel("Frequency")
plt.xlabel("WER of Patch-attacked Long Audio Samples")
plt.savefig("WER.png")

print("WERS were")
print(np.mean(wers))
print()

print(f"Number under threshold of {THRESHOLD} characters were")
print(num_under_threshold)
print()

print("Number of empty transcriptions")
print(num_empty_strings)
print()

# wer(original_transcription.lower().strip(), adversarial_transcription.lower().strip()))