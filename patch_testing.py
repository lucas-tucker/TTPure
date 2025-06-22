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
from pydub import AudioSegment
import io
import torchaudio



### Define the model to test
WHISPER_MODEL = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
CUTOFF = 30 * 16000  # 30 second cutoff size
TINY_SNIPPET_LENGTH = 50  # Defense snippet length

print(f"Using device: {DEVICE}")

### Load Whisper model
model = whisper.load_model(WHISPER_MODEL).to(DEVICE)
print(f"Loaded Whisper {WHISPER_MODEL} model")

### Load the patch attack
tt_patch_attack_path = r"C:\Users\judoc\Documents\GitHub\TTPure\tucker_attacks\tt_patch.th" # Replace as needed
try:
    patch_data = torch.load(tt_patch_attack_path, map_location=torch.device(DEVICE))
    tt_patch = patch_data["audio_attack_segment"]
    print(f"Loaded tt patch attack segment of shape {tt_patch.shape}")
except:
    print("Warning: Could not load embed attack file. Using random noise for demonstration.")
    tt_patch = torch.randn(8000).to(DEVICE) * 0.01

test_dir = r"C:\Users\judoc\Documents\GitHub\TTPure\tucker_attacks\librispeech-long\test-other" # Replace as needed
# Grab the actual audio files, of which there should be 810; assume 16kHz sampling
flac_files = glob.glob(os.path.join(test_dir, "**", "**", "*.flac"), recursive=True)

segment_size = int(16000 * (30 - 0.64))
wers = []

################################################
filtered_8k_wers = []
filtered_4k_wers = []
gain_db = 20.0

################################################

testing_data_size = 10
THRESHOLD = 15
num_empty_strings = 0
num_under_threshold = 0

############### TESTING FLAGS ###############
mp3_test = True
aac_test = True
#############################################


def flac_to_mp3(flac_path):
    """Convert a FLAC file to MP3 format."""
    audio = AudioSegment.from_file(flac_path, format="flac")
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)
    return mp3_buffer

# def flac_to_aac(flac_path):
#     """Convert a FLAC file to AAC format."""
#     audio = AudioSegment.from_file(flac_path, format="flac")
#     aac_buffer = io.BytesIO()
#     audio.export(
#         aac_buffer,
#         format="mp4",   # ADTS container for raw AAC
#         codec="aac",
#         bitrate="192k"
#     )
#     aac_buffer.seek(0)
#     return aac_buffer

def adversarial_attack(audio_sample, patch, segment_size):
    """
    Apply the adversarial attack by embedding the patch into the audio sample.
    The audio sample is split into segments of size `segment_size`, and the patch
    is embedded at a random position within each segment.
    """
    # Split the audio into 30s - 0.64s size segments
    num_segments = int(len(audio_sample) // segment_size)
    seek = 0
    adv_audio = torch.zeros((1,)).to(DEVICE)
    for _ in range(num_segments):
        breakpoint = random.randint(0, segment_size)
        adv_audio = torch.cat((adv_audio, 
                               audio_sample[seek:seek + breakpoint], 
                               patch,
                               audio_sample[seek + breakpoint:seek + segment_size]), dim=0)
        seek += segment_size
    return adv_audio


for path in flac_files[:testing_data_size]:
    audio_sample, sr = torchaudio.load(path)
    audio_sample = audio_sample.squeeze().to(DEVICE) # Now a whisper-readable 1D tensor
    adv_audio = adversarial_attack(audio_sample, tt_patch, segment_size)
    with torch.no_grad():
        original_transcription = model.transcribe(audio_sample)["text"]
        adversarial_transcription = model.transcribe(adv_audio)["text"]
        
        wer_original_adv = wer(original_transcription.lower().strip(), adversarial_transcription.lower().strip())
        print(f"WER between original and adversarial: {wer_original_adv}")
        
        if mp3_test:
            waveform_mp3 = flac_to_mp3(path)
            audio_sample_mp3, _ = torchaudio.load(waveform_mp3, format="mp3")
            audio_sample_mp3 = audio_sample_mp3.squeeze().to(DEVICE)
            
            adv_audio_mp3 = adversarial_attack(audio_sample_mp3, tt_patch, segment_size)
            original_transcription_mp3 = model.transcribe(audio_sample_mp3)["text"]
            adversarial_transcription_mp3 = model.transcribe(adv_audio_mp3)["text"]
            
            wer_original_v_mp3 = wer(original_transcription.lower().strip(), original_transcription_mp3.lower().strip())
            print(f"WER between original and mp3: {wer_original_v_mp3}")
            
            wer_mp3_v_adv_mp3 = wer(original_transcription_mp3.lower().strip(), adversarial_transcription_mp3.lower().strip())
            print(f"WER between mp3 and adversarial mp3: {wer_mp3_v_adv_mp3}")
        
        # if aac_test:
        #     waveform_aac = flac_to_aac(path)
        #     audio_sample_aac, _ = torchaudio.load(waveform_aac, format="mp4")
        #     audio_sample_aac = audio_sample_aac.squeeze().to(DEVICE)
            
        #     adv_audio_aac = adversarial_attack(audio_sample_aac, tt_patch, segment_size)
        #     original_transcription_aac = model.transcribe(audio_sample_aac)["text"]
        #     adversarial_transcription_aac = model.transcribe(adv_audio_aac)["text"]
            
        #     wer_original_v_aac = wer(original_transcription.lower().strip(), original_transcription_aac.lower().strip())
        #     print(f"WER between original and aac: {wer_original_v_aac}")
            
        #     wer_aac_v_adv_aac = wer(original_transcription_aac.lower().strip(), adversarial_transcription_aac.lower().strip())
        #     print(f"WER between aac and adversarial aac: {wer_aac_v_adv_aac}")

                    
        # ################################################################################################
        # # 8kHz lowpass
        # if audio_sample.ndim == 1:
        #     audio_sample = audio_sample.unsqueeze(0)
        # #LOWPASS
        # filtered_audio_sample_8k = F.lowpass_biquad(audio_sample, sample_rate=16000, cutoff_freq=8000).squeeze()
        # filtered_transcription_8k = model.transcribe(filtered_audio_sample_8k)["text"]
        # #HIGH GAIN
        # high_gain_audio = T.Vol(gain_db, gain_type='db')(audio_sample).squeeze()
        # high_gain_transcription = model.transcribe(high_gain_audio)["text"]
        # #RESAMPLED
        # resampler = T.Resample(orig_freq=16000, new_freq=8000).to(DEVICE)
        # resampled_8k_audio = resampler(audio_sample).squeeze()
        # resampled_8k_transcription = model.transcribe(resampled_8k_audio)["text"]
        
        # if adv_audio.ndim == 1:
        #     adv_audio = adv_audio.unsqueeze(0)
        # #LOWPASS
        # filtered_adv_audio_8k = F.lowpass_biquad(adv_audio, sample_rate=16000, cutoff_freq=8000).squeeze()
        # filtered_adv_transcription_8k = model.transcribe(filtered_adv_audio_8k)["text"]
        # #HIGH GAIN
        # high_gain_adv_audio = T.Vol(gain_db, gain_type='db')(adv_audio).squeeze()
        # high_gain_adv_transcription = model.transcribe(high_gain_adv_audio)["text"]
        # #RESAMPLED
        # resampled_8k_adv_audio = resampler(adv_audio).squeeze()
        # resampled_8k_adv_transcription = model.transcribe(resampled_8k_adv_audio)["text"]
        

        # # 4kHz lowpass
        # filtered_audio_sample_4k = F.lowpass_biquad(audio_sample, sample_rate=16000, cutoff_freq=4000).squeeze()
        # filtered_adv_audio_4k = F.lowpass_biquad(adv_audio, sample_rate=16000, cutoff_freq=4000).squeeze()

        # filtered_transcription_4k = model.transcribe(filtered_audio_sample_4k)["text"]
        # filtered_adv_transcription_4k = model.transcribe(filtered_adv_audio_4k)["text"]
        
        
        # ################################################################################################

        # num_under_threshold += len(adversarial_transcription) <= THRESHOLD
        # num_empty_strings += len(adversarial_transcription) == 0

        # wers.append(wer(original_transcription.lower().strip(), adversarial_transcription.lower().strip()))
        # print(f"Avg WER thus far is {np.mean(wers)}")

        # filtered_8k_wers.append(wer(filtered_transcription_8k.lower().strip(), filtered_adv_transcription_8k.lower().strip()))
        # print(f"Filtered 8kHz Avg WER thus far is {np.mean(filtered_8k_wers)}")

        # filtered_4k_wers.append(wer(filtered_transcription_4k.lower().strip(), filtered_adv_transcription_4k.lower().strip()))
        # print(f"Filtered 4kHz Avg WER thus far is {np.mean(filtered_4k_wers)}")

        # # Compare original transcription to low-pass filtered transcriptions
        # wer_orig_vs_8k = wer(original_transcription.lower().strip(), filtered_transcription_8k.lower().strip())
        # wer_orig_vs_4k = wer(original_transcription.lower().strip(), filtered_transcription_4k.lower().strip())
        # print(f"WER between original and 8kHz filtered: {wer_orig_vs_8k}")
        # print(f"WER between original and 4kHz filtered: {wer_orig_vs_4k}")
        # # print(f"Original transcription: {original_transcription}")
        # # print(f"8kHz filtered transcription: {filtered_transcription_8k}")
        # # print(f"4kHz filtered transcription: {filtered_transcription_4k}")
        # # Compare resampled to original
        # wer_resampled_vs_orig = wer(original_transcription.lower().strip(), resampled_8k_transcription.lower().strip())
        # print(f"WER between original and resampled 8kHz: {wer_resampled_vs_orig}")
        
        # # compare adversarial resampled to 8k resampled
        # wer_resampled_adv_vs_8k = wer(resampled_8k_transcription.lower().strip(), resampled_8k_adv_transcription.lower().strip())
        # print(f"WER between resampled 8kHz and adversarial resampled 8kHz: {wer_resampled_adv_vs_8k}")
        
        # # Compare high gain to original
        # wer_high_gain_vs_orig = wer(original_transcription.lower().strip(), high_gain_transcription.lower().strip())
        # print(f"WER between original and high gain: {wer_high_gain_vs_orig}")

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
