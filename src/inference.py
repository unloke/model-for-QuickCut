# src/inference.py
import torch
import librosa
import numpy as np
from . import config
from .model import StutterDetector

def load_model(model_path):
    model = StutterDetector()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    
    # 切成 2 秒段 (可 overlap)
    chunk_samples = int(config.CHUNK_DURATION * config.SAMPLE_RATE)
    chunks = []
    for start in range(0, len(y), chunk_samples // 2):  # 1 秒 overlap
        chunk = y[start:start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
    
    # 轉 Mel Spectrogram
    specs = []
    for chunk in chunks:
        melspec = librosa.feature.melspectrogram(
            y=chunk, sr=config.SAMPLE_RATE, 
            n_mels=config.N_MELS, 
            hop_length=config.HOP_LENGTH
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)
        specs.append(melspec)
    
    return torch.FloatTensor(np.array(specs)).unsqueeze(1)  # (N, 1, n_mels, time)

def infer(model, audio_tensor):
    with torch.no_grad():
        outputs = model(audio_tensor)  # (N, time, 3)
        preds = torch.argmax(outputs, dim=-1)  # (N, time)
    return preds.numpy()

# 範例使用
if __name__ == "__main__":
    model = load_model(config.MODEL_SAVE_PATH)
    audio_tensor = preprocess_audio("path/to/your/audio.wav")
    predictions = infer(model, audio_tensor)
    print("Predictions shape:", predictions.shape)
    # predictions[i][j] = 0/1/2 for chunk i, frame j