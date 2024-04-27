from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
example_speech = dataset[0]["audio"]["array"]

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(
    audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt"
)

# load xvector containing speaker's voice characteristics from a file
import numpy as np
import torch

speaker_embeddings = np.load("xvector_speaker_embedding.npy")
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

speech = model.generate_speech(
    inputs["input_values"], speaker_embeddings, vocoder=vocoder
)

import soundfile as sf

sf.write("speech.wav", speech.numpy(), samplerate=16000)
