# src/train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from . import config
from .model import StutterDetector
from .dataset import StutterDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # 每個 Epoch 動態生成 2000 筆數據
    dataset = StutterDataset(length=2000)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    model = StutterDetector().to(device)
    
    # Loss function 加權
    # 0: Normal (佔大多數)
    # 1: Filler (需要精準)
    # 2: Stutter (需要精準)
    weights = torch.tensor([1.0, 2.5, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("開始訓練...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            # Reshape for Loss
            outputs = outputs.view(-1, config.CLASSES)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_loss:.4f}")
        
        # 覆蓋儲存最新模型
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print(f"訓練完成！模型已儲存至 {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()