import torch
import torch.nn as nn
import numpy as np
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES, load_audio
import random

class AudioAttackModelWrapper(nn.Module):
    '''
        Whisper Model wrapper with learnable audio segment attack prepended to speech signals
    '''
    def __init__(self, tokenizer, attack_size=5120, device=None, attack_init='random'):
        super(AudioAttackModelWrapper, self).__init__()
        self.attack_size = attack_size
        self.tokenizer = tokenizer
        self.device = device
        self.multiple_model_attack = False

        self.sot_ids = self.tokenizer.sot_sequence_including_notimestamps # self.tokenizer.sot_sequence # self.tokenizer.sot_sequence_including_notimestamps
        self.len_sot_ids = len(torch.tensor(self.sot_ids))

        if attack_init == 'random':
            self.audio_attack_segment = nn.Parameter(torch.rand(attack_size))
        else:
            # load init attack from attack_init path
            # loaded_params = torch.load(attack_init)
            # Comment out this loading for now since assumes th format is given

            # if 'audio_attack_segment' in loaded_params:
            #     initial_value = loaded_params['audio_attack_segment']
            #     self.audio_attack_segment = nn.Parameter(initial_value.to(device))
            # else:
            #     raise ValueError("Invalid attack_init path provided.")
            init_segment = np.load(attack_init)
            init_segment_tensor = torch.from_numpy(init_segment).to(device)
            self.audio_attack_segment = nn.Parameter(init_segment_tensor)
    
    def forward(self, audio_vector, whisper_model, decoder_input=None):
        '''
            audio_vector: Torch.tensor: [Batch x Audio Length]
            whisper_model: encoder-decoder model

            Returns the logits
        '''
        # prepend attack segment
        X = self.audio_attack_segment.unsqueeze(0).expand(audio_vector.size(0), -1)
        ### EXPERIMENT ####
<<<<<<< HEAD
        # We fix a cutoff size to segment the audio vector
        cutoff = 30000 + random.randint(-30, 0)*1000 # len(audio_vector[0])
        # print(f"audio vector is itself of shape {audio_vector.shape}")
        attacked_audio_vector = torch.cat((audio_vector[:, :cutoff//2], X, audio_vector[:, cutoff//2:]), dim=1)
=======
        # We fix a max cutoff size to segment the audio vector
        max_cutoff = 35000 # len(audio_vector[0])
        cutoff = torch.randint(max_cutoff, (1,)).item()
   
        attacked_audio_vector = torch.cat((audio_vector[:, :cutoff], X, audio_vector[:, cutoff:]), dim=1)
>>>>>>> 8ee30f27b552b79532aaf676e183f1cb2b5ddf32
        # print(f"attacked audio vector is of shape {attacked_audio_vector.shape}")
        ### END OF EXPERIMENT ###

        # attacked_audio_vector = torch.cat((X, audio_vector), dim=1)
        # attacked_audio_vector = torch.cat((audio_vector, X), dim=1)

        # forward pass through full model
        mel = self._audio_to_mel(attacked_audio_vector, whisper_model)
        return self._mel_to_logit(mel, whisper_model, decoder_input=decoder_input)
    

    def _audio_to_mel(self, audio: torch.Tensor, whisper_model):
        '''
            audio: [Batch x Audio length]
            based on https://github.com/openai/whisper/blob/main/whisper/audio.py
        '''
        if self.multiple_model_attack:
            n_mels = whisper_model.models[0].dims.n_mels
        else:
            n_mels = whisper_model.model.dims.n_mels
        padded_mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)
        # ADDED EXPERIMENT
        seek = 0
        segment_size = 800
        padded_mel = padded_mel[:, :, seek : seek + segment_size]
        # print(f"truncated mel size is {padded_mel.shape}")
        # END OF EXP
        mel = pad_or_trim(padded_mel, N_FRAMES)
        # mel = padded_mel
        return mel
    
    def _mel_to_logit(self, mel: torch.Tensor, whisper_model, decoder_input=None):
        '''
            Forward pass through the whisper model of the mel vectors
            expect mel vectors passed as a batch and padded to 30s of audio length
            mel: torch.Tensor [B x dim x num_vectors]
        '''
        # create batch of start of transcript tokens
        sot_ids = torch.tensor(self.sot_ids)
        sot_ids = sot_ids.to(self.device)
        decoder_input_ids = sot_ids.unsqueeze(0).expand(mel.size(0), -1)
        if decoder_input is not None:
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_input), dim=1)

        if self.multiple_model_attack:
            # pass through each target model
            sf = nn.Softmax(dim=-1)
            pred_probs = []
            for model in whisper_model.models:
                pred_probs.append(sf(model.forward(mel, decoder_input_ids)))
            return torch.mean(torch.stack(pred_probs), dim=0) 
        else:
            ######### ADD THE 5 SECOND TOKEN AS THE LAST LOGIT SLICE FROM THE OUTPUT #########
            # new_tokens = [7835, 538, 3693, 538, 436, 1361, 281, 452, 1159, 11]
            new_tokens = []
            # new_token = 50364
            new_tokens = [50364]
            for new_token in new_tokens:
            # new_token = 50514 # This is the 3 second token
                new_token_tensor = torch.zeros(decoder_input_ids.shape[0], 1, dtype=torch.int64) + new_token
                new_token_tensor = new_token_tensor.to(self.device)
                decoder_input_ids = torch.cat((decoder_input_ids, new_token_tensor), dim=1)
            # decoder_input_ids = torch.zeros(decoder_input_ids.shape[0], 1, dtype=torch.int64) + new_token
            # print(f"decoder_input_ids are {decoder_input_ids}")
            ######### END OF ALTERED CODE #########
            return whisper_model.model.forward(mel, decoder_input_ids)
    
    def transcribe(self,
        whisper_model,
        audio,
        do_attack=True,
        without_timestamps=False
    ):

        '''
            Mimics the original Whisper transcribe functions but prepends the adversarial attack
            in the audio space

                do_attack parameter is a boolean to do the attack or not
        '''
        if do_attack:
            # prepend attack
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio).to(self.device)
            audio = torch.cat((self.audio_attack_segment, audio), dim=0)

        return whisper_model.predict(audio, without_timestamps=without_timestamps)


