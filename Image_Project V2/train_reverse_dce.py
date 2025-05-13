import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dce_model import ReverseDCEUNet  # ReverseDCEUNet aynı dizinde olmalı
from tqdm import tqdm

# 🔹 1. Dataset Sınıfı
class BrightnessDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.filenames = os.listdir(input_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])

        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        input_img = cv2.resize(input_img, (256, 256))
        target_img = cv2.resize(target_img, (256, 256))

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

# 🔹 2. Transform ve Dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
])

dataset = BrightnessDataset(
    input_dir='dataset/input',
    target_dir='dataset/target',
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🔹 3. Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReverseDCEUNet().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 🔹 4. Eğitim Döngüsü
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for input_img, target_img in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        enhanced_img, _ = model(input_img)

        loss = criterion(enhanced_img, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

    # Her 5 epoch'ta bir model kaydet
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'reverse_dce_epoch{epoch+1}.pth')

# 🔹 Final Model Kaydı
torch.save(model.state_dict(), 'reverse_dce.pth')
print("Eğitim tamamlandı. Model kaydedildi: reverse_dce.pth")
