# src/config.py
import os

# 路徑設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_SPEECH_DIR = os.path.join(BASE_DIR, "data", "LibriSpeech_dev_clean")
FILLER_DIR = os.path.join(BASE_DIR, "data", "my_fillers")
NOISE_DIR = os.path.join(BASE_DIR, "data", "noises")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "output", "stutter_detector.pth")

# 允許的音訊格式 (防止讀到 .txt, .jpg)
VALID_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# 音訊參數
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0  # 訓練時切成 2 秒一段
N_MELS = 64           # 梅爾頻譜的特徵數
HOP_LENGTH = 320      # 時間解析度: 16000/320 = 50Hz (每20ms一個點)

# 訓練參數
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
CLASSES = 3  # 0: Normal, 1: Filler, 2: Stutter