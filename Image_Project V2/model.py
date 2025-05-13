"""import torch
import cv2
import numpy as np
from dce_model import DCE_net
import torchvision.transforms as transforms

def enhance_image(image_path, output_path, model_path='Epoch99.pth'):
    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model yükleme
    model = DCE_net().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Görüntü okuma
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Görüntü okunamadı: {image_path}")
        return False
    
    # Parlaklık ölçümü
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    # Görüntü işleme
    transform = transforms.Compose([transforms.ToTensor()])
    
    if brightness < 80:
        # DCE-Net ile iyileştirme (karanlık görüntüler)
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = input_img.shape
        input_img_resized = cv2.resize(input_img, (512, 512))
        input_tensor = transform(input_img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced_img, _ = model(input_tensor)

        enhanced_img = enhanced_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_img = np.clip(enhanced_img, 0, 1)
        enhanced_img = (enhanced_img * 255).astype(np.uint8)
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        
        # Orijinal boyuta geri döndürme
        enhanced_img = cv2.resize(enhanced_img, (w, h))

        # Ek gamma düzeltmesi
        enhanced_img = adjust_gamma(enhanced_img, gamma=0.7)
        
    elif brightness > 180:
        # Aşırı parlak görüntüler için gamma düzeltmesi
        enhanced_img = adjust_gamma(frame, gamma=2.2)
        
    else:
        # Normal parlaklıkta olan görüntüler
        enhanced_img = frame
    
    # Sonucu kaydetme
    cv2.imwrite(output_path, enhanced_img)
    return True

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Kullanım örneği
enhance_image('karanlik_goruntu.jpg', 'iyilestirilmis_goruntu.jpg')"""