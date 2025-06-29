'''
    Train adversarial segment
'''

import random
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import whisper
import glob
from src.models.whisper import WhisperModel, WhisperModelEnsemble

from src.tools.args import core_args, attack_args
# from src.data.load_data import load_data
# from src.models.load_model import load_model
# from src.tools.tools import get_default_device, set_seeds
from src.attacker.selector import select_train_attacker
# from src.tools.saving import base_path_creator, attack_base_path_creator_train

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    # set seeds
    # set_seeds(core_args.seed)
    # base_path = base_path_creator(core_args)
    # attack_base_path = attack_base_path_creator_train(attack_args, base_path)

    # Not sure what this is for...
    # # Save the command run
    # if not os.path.isdir('CMDs'):
    #     os.mkdir('CMDs')
    # with open('CMDs/train_attack.cmd', 'a') as f:
    #     f.write(' '.join(sys.argv)+'\n')

    # Get the device
    # if core_args.force_cpu:
    #     device = torch.device('cpu')
    # else:
    #     device = get_default_device(core_args.gpu_id)
    # print(f"device is {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # load training data
    # data, _ = load_data(core_args)
    # base_dir = "/Users/lucastucker/misc-cs/TTPure/tucker_attacks/src/data/LibriSpeech/dev-clean"
    # cache_dir = "/Users/lucastucker/misc-cs/TTPure/tucker_attacks/tucker_saved_segments"
    # saved_universal_prepend_init_segment = "/Users/lucastucker/base.np.npy"
    # base_dir = "/mnt/c/Documents and Settings/judoc/Documents/GitHub/TTPure/src/data/LibriSpeech/dev-clean"
    # cache_dir = "/mnt/c/Documents and Settings/judoc/Documents/GitHub/TTPure/tucker_attacks/10k_saved_segments"
    # saved_universal_prepend_init_segment = "/mnt/c/Documents and Settings/judoc/Documents/GitHub/TTPure/base.np.npy"
    
    base_dir = r"C:/Users/judoc/Documents/GitHub/TTPure/tucker_attacks/data/dev-clean"
    cache_dir = r"C:\Users\judoc\Documents\GitHub\TTPure\tucker_attacks\1k_saved_segments"
    # saved_universal_prepend_init_segment = r"C:/Users/judoc/Documents/GitHub/TTPure/base.np.npy"
    saved_universal_prepend_init_segment = r"C:\Users\judoc\Documents\GitHub\TTPure\tucker_attacks\trained_5k_25epoch.npy"

    # Use glob to recursively find all .flac files
    flac_files = glob.glob(os.path.join(base_dir, "**", "**", "*.flac"), recursive=True)
    # NOTE that batching is currently not supported!
    training_data_size = 2
    data = [{"audio": path} for path in flac_files[:training_data_size]]
    attack_args.attack_init = saved_universal_prepend_init_segment
    print(f"attack args are {attack_args}")
    # data = [{"audio": "/Users/lucastucker/book_audio.flac"}]
    # data = [{"audio": "/Users/lucastucker/lucas.flac"}]
    # print(data)
    # print(f"data size is {len(data)}")
    # data = [{"audio": "/Users/lucastucker/misc-cs/TTPure/tucker_attacks/src/data/LibriSpeech/dev-clean/251/137823/251-137823-0000.flac"}]
    # data = [{"audio": "/Users/lucastucker/misc-cs/TTPure/tucker_attacks/src/data/data/LibriSpeech/dev-clean/3576/138058/3576-138058-0000.flac"}]
    # load model
    model = WhisperModel(core_args.model_name[0], device=device, task=core_args.task, language=core_args.language) # load_model(core_args, device=device)
    attacker = select_train_attacker(attack_args, core_args, model, device=device)
    attacker.train_process(data, cache_dir)
    