import torch
import matplotlib.pyplot as plt
import numpy as np
from QuickCut_Trainer.src import config
from QuickCut_Trainer.src.model import StutterDetector
from QuickCut_Trainer.src.dataset import StutterDataset

# 1. 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StutterDetector().to(device)
# 載入最新的權重 (請確認路徑對應你的 output 資料夾)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
model.eval()

# 2. 隨機生成一筆數據
dataset = StutterDataset(length=1)
melspec, target = dataset[0] # 取第一筆
# 增加 Batch 維度: (1, 1, 64, Time)
input_tensor = melspec.unsqueeze(0).to(device)

# 3. 推論
with torch.no_grad():
    output = model(input_tensor) # (1, Time, 3)
    # 取出機率最大的類別
    pred = torch.argmax(output, dim=2).squeeze().cpu().numpy()

# 4. 畫圖比較
target = target.numpy()
melspec_img = melspec.squeeze().numpy()

plt.figure(figsize=(12, 8))

# 畫聲譜圖
plt.subplot(3, 1, 1)
plt.imshow(melspec_img, aspect='auto', origin='lower')
plt.title("Mel Spectrogram (Input)")
plt.ylabel("Frequency")

# 畫真實標籤
plt.subplot(3, 1, 2)
plt.plot(target, label='Ground Truth', color='green', linewidth=2)
plt.ylim(-0.5, 2.5)
plt.yticks([0, 1, 2], ['Normal', 'Filler', 'Stutter'])
plt.title("Ground Truth Mask")
plt.grid(True, alpha=0.3)

# 畫預測結果
plt.subplot(3, 1, 3)
plt.plot(pred, label='Prediction', color='red', linestyle='--', linewidth=2)
plt.ylim(-0.5, 2.5)
plt.yticks([0, 1, 2], ['Normal', 'Filler', 'Stutter'])
plt.title("Model Prediction")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()