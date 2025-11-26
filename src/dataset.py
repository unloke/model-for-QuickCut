# src/dataset.py
import os
import glob
import math
import random
import numpy as np
import librosa
import torch
import audiomentations
from torch.utils.data import Dataset
from . import config

class AudioSynthesizer:
    def __init__(self):
        # (Fix 6) 使用副檔名過濾
        self.clean_files = self._scan_files(config.CLEAN_SPEECH_DIR)
        self.filler_files = self._scan_files(config.FILLER_DIR)
        self.noise_files = self._scan_files(config.NOISE_DIR)
        
        if not self.clean_files:
            raise RuntimeError(f"找不到任何有效音訊於: {config.CLEAN_SPEECH_DIR}")
        
        if not self.filler_files:
            print("警告：找不到贅字檔案，將跳過贅字合成訓練")

        # 聲音增強
        self.augmentor = audiomentations.Compose([
            audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
        ])

    def _scan_files(self, dir_path):
        files = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in config.VALID_EXTENSIONS:
                    files.append(os.path.join(root, filename))
        return files

    def _load_and_normalize(self, path, target_len=None, trim=False, min_ratio=0.3):
        """
        讀取、正規化、去除靜音、長度檢查
        """
        try:
            # 強制轉 16k mono
            y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
            
            # 1. 切除靜音
            if trim:
                y, _ = librosa.effects.trim(y, top_db=20)

            # 2. 空檔案檢查
            if len(y) == 0:
                return None

            # (Fix A) 防止過短的片段被 Loop 造成過擬合
            # 只有當指定了 target_len (通常是噪音或 filler 填補背景時) 才檢查
            if target_len and trim:
                if len(y) < target_len * min_ratio:
                    return None

            # 3. RMS 音量標準化
            rms = np.sqrt(np.mean(y**2))
            if rms > 1e-6:
                y = y / rms
            else:
                return None # 全是靜音
            
            # 4. 長度處理 (Looping or Cutting)
            if target_len:
                if len(y) >= target_len:
                    start = random.randint(0, len(y) - target_len)
                    y = y[start : start + target_len]
                else:
                    repeats = math.ceil(target_len / len(y))
                    y = np.tile(y, repeats)[:target_len]
            
            return y
            
        except Exception as e:
            # print(f"Error loading {path}: {e}") # 訓練時太吵可以註解掉
            return None

    def _get_base_audio(self, target_samples, max_retries=10):
        """(Fix 3) 使用迴圈重試取代遞迴，避免 Stack Overflow"""
        for _ in range(max_retries):
            base_path = random.choice(self.clean_files)
            # Base audio 不需要 trim 也不需要 min_ratio 檢查，因為它是背景
            y = self._load_and_normalize(base_path, target_len=target_samples, trim=False)
            if y is not None:
                return y
        raise RuntimeError("無法載入任何基底語音，請檢查數據集路徑或檔案完整性。")

    def synthesize(self):
        target_samples = int(config.CHUNK_DURATION * config.SAMPLE_RATE)
        
        # 取得基底語音
        y = self._get_base_audio(target_samples)
        
        # 隨機音量
        speech_vol = random.uniform(0.1, 0.9)
        y = y * speech_vol

        # 初始化 Mask (Fix 5: 使用 int64)
        n_frames = int(target_samples / config.HOP_LENGTH) + 1
        mask = np.zeros(n_frames, dtype=np.int64)

        # --- 注入背景噪音 ---
        if self.noise_files and random.random() < 0.5:
            noise_path = random.choice(self.noise_files)
            noise_y = self._load_and_normalize(noise_path, target_len=target_samples, trim=True)
            
            if noise_y is not None:
                # SNR 控制：噪音比人聲小
                noise_vol = speech_vol * random.uniform(0.01, 0.3)
                y = y + (noise_y * noise_vol)

        # 決定模式
        mode = random.choices(['normal', 'filler', 'stutter'], weights=[0.4, 0.3, 0.3])[0]

        # --- Filler 模式 ---
        if mode == 'filler' and self.filler_files:
            filler_path = random.choice(self.filler_files)
            # Filler 必須 trim，且不能太短
            f_y = self._load_and_normalize(filler_path, target_len=None, trim=True, min_ratio=0.1)
            
            if f_y is not None:
                # 應用增強
                f_y = self.augmentor(samples=f_y, sample_rate=config.SAMPLE_RATE)

                # (Fix 2) 檢查增強後的長度
                f_len = len(f_y)
                if f_len >= target_samples:
                    # 如果變太長，裁切到比目標稍短，留空間插入
                    f_len = target_samples - 100 
                    f_y = f_y[:f_len]
                
                if f_len > 0:
                    insert_idx = random.randint(0, target_samples - f_len)
                    
                    # 混合：稍微壓低背景
                    y[insert_idx : insert_idx + f_len] *= 0.3
                    y[insert_idx : insert_idx + f_len] += (f_y * speech_vol * random.uniform(0.8, 1.2))
                    
                    # 更新 Mask
                    start_frame = int(insert_idx / config.HOP_LENGTH)
                    end_frame = int((insert_idx + f_len) / config.HOP_LENGTH)
                    # 邊界檢查
                    end_frame = min(end_frame, len(mask))
                    if start_frame < end_frame:
                        mask[start_frame : end_frame] = 1

        # --- (Fix 7) Stutter 口吃模式實作 ---
        elif mode == 'stutter':
            # 使用 librosa 檢測發音起點 (Onset)
            onset_frames = librosa.onset.onset_detect(y=y, sr=config.SAMPLE_RATE, units='samples')
            
            # 只有當找到起點，且起點不在音訊末端時才做
            if len(onset_frames) > 0:
                # 過濾掉太靠近結尾的起點
                valid_onsets = [o for o in onset_frames if o < target_samples - 2000]
                
                if valid_onsets:
                    split_idx = random.choice(valid_onsets)
                    
                    # 決定重複長度 (0.15s - 0.4s)
                    repeat_len = int(random.uniform(0.15, 0.4) * config.SAMPLE_RATE)
                    
                    # 確保擷取不越界
                    if split_idx + repeat_len < len(y):
                        chunk = y[split_idx : split_idx + repeat_len].copy()
                        
                        # 簡單的 Crossfade 處理 (前後 5ms)
                        fade_len = int(0.005 * config.SAMPLE_RATE)
                        if len(chunk) > fade_len * 2:
                            fade_in = np.linspace(0, 1, fade_len)
                            fade_out = np.linspace(1, 0, fade_len)
                            chunk[:fade_len] *= fade_in
                            chunk[-fade_len:] *= fade_out

                        # 模擬口吃：插入重複片段
                        # 原本: [A] [B] [C]
                        # 變為: [A] [chunk] [B] [C] ... 然後裁切回原長度
                        
                        prefix = y[:split_idx]
                        suffix = y[split_idx:]
                        
                        # 拼接 (這裡會讓總長度變長)
                        stuttered_y = np.concatenate([prefix, chunk, suffix])
                        
                        # 裁切回 target_samples
                        y = stuttered_y[:target_samples]
                        
                        # 更新 Mask
                        # 重複的那一段是 chunk，位置在 split_idx 到 split_idx + repeat_len
                        start_frame = int(split_idx / config.HOP_LENGTH)
                        end_frame = int((split_idx + repeat_len) / config.HOP_LENGTH)
                        
                        # 因為這段是「多出來的重複」，我們將其標記為 2
                        end_frame = min(end_frame, len(mask))
                        if start_frame < end_frame:
                            mask[start_frame : end_frame] = 2

        # 最後 Clip 防止爆音
        y = np.clip(y, -1.0, 1.0)
        
        return y, mask


class StutterDataset(Dataset):
    def __init__(self, length=1000):
        self.synthesizer = AudioSynthesizer()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav, mask = self.synthesizer.synthesize()
        
        # 轉 Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=wav, sr=config.SAMPLE_RATE, 
            n_mels=config.N_MELS, 
            hop_length=config.HOP_LENGTH
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # 正規化 (Mean/Std)
        mean = melspec.mean()
        std = melspec.std() + 1e-6
        melspec = (melspec - mean) / std
        
        # (Fix 4) 強制長度對齊
        # 因為 STFT 計算可能會有一兩幀的誤差，這裡以 mask 長度為基準
        target_len = mask.shape[0]
        current_len = melspec.shape[1]
        
        if current_len > target_len:
            melspec = melspec[:, :target_len]
        elif current_len < target_len:
            # 如果 Mel 比較短 (罕見)，裁切 Mask
            mask = mask[:current_len]

        return torch.FloatTensor(melspec).unsqueeze(0), torch.LongTensor(mask)