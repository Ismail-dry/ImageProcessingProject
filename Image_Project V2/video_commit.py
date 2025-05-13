import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as transforms
from dce_model import DCE_net, ReverseDCEUNet

try:
    from dce_model import DCE_net
    DCE_AVAILABLE = True
except ImportError:
    print("Warning: DCE_net model could not be imported. Only HDRNet will be available.")
    DCE_AVAILABLE = False


class ImageEnhancer:
    def __init__(self, dce_model_path='Epoch99.pth', hdrnet_model_path='hdrnet_trained.pth'):
        # Cihaz seçimi
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Cihaz: {self.device}")
        
        # DCE_net Model yükleme (düşük ışık)
        self.dce_model = DCE_net().to(self.device)
        self.reverse_dce_model = ReverseDCEUNet().to(self.device)
        self.reverse_dce_model.eval()


        try:
            dce_state_dict = torch.load(dce_model_path, map_location=self.device)
            self.dce_model.load_state_dict(dce_state_dict)
            self.dce_model.eval()
            print(f"DCE-Net modeli başarıyla yüklendi: {dce_model_path}")
        except Exception as e:
            print(f"DCE-Net modeli yüklenirken hata oluştu: {e}")
            print("DCE-Net modeli kullanılamayacak")
            self.dce_model = None
       

        # Görüntü tensor dönüşümü
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Varsayılan parametreler
        self.low_light_threshold = 78
        self.high_light_threshold = 79
        self.low_light_gamma = 0.7
        self.high_light_gamma = 2.2

    def apply_tonemap(self, image):
        """Reinhard tonemap ile patlamış alanları doğal hale getir"""
        img_float = image.astype(np.float32) / 255.0
        tonemap = cv2.createTonemapReinhard(gamma=1.4)
        tonemapped = tonemap.process(img_float)
        tonemapped = np.clip(tonemapped * 255, 0, 255).astype(np.uint8)
        return tonemapped
    


    def apply_retinex(self, image):
        """Basit retinex iyileştirmesi (patlamış beyaz alanlar için kontrast düzeltmesi)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) + 1.0  # log(0)'ı önlemek için
        log_img = np.log(image_float)
        blur = cv2.GaussianBlur(log_img, (15, 15), 0)
        retinex = log_img - blur
        retinex = np.exp(retinex)
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
        retinex = (retinex * 255).astype(np.uint8)
        return cv2.cvtColor(retinex, cv2.COLOR_RGB2BGR)

    def set_parameters(self, low_threshold=80, high_threshold=120, low_gamma=0.7, high_gamma=2.2):
        """Parlaklık eşiklerini ve gamma değerlerini ayarla"""
        self.low_light_threshold = low_threshold
        self.high_light_threshold = high_threshold

        print(high_threshold)
        print("geldi")
        self.low_light_gamma = low_gamma
        self.high_light_gamma = high_gamma
    
    def adjust_gamma(self, image, gamma=1.0):
        """Gamma düzeltme işlemi"""
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def enhance_image(self, image):
        """Tek bir görüntüyü iyileştir"""
        # Parlaklık ölçümü
        if len(image.shape) == 3:  # Renkli görüntü
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:  # Zaten gri tonlamalı
            gray = image
            
        brightness = np.mean(gray)
        
        # Görüntü iyileştirme
        if brightness < self.low_light_threshold:
            # DCE-Net ile iyileştirme (karanlık görüntüler)
            if self.dce_model is None:
                print("DCE-Net modeli yüklü değil, alternatif iyileştirme uygulanıyor...")
                # DCE-Net yoksa, basit bir histogram eşitleme uygula
                enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                enhanced_img[:,:,0] = cv2.equalizeHist(enhanced_img[:,:,0])
                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_YUV2BGR)
                enhanced_img = self.adjust_gamma(enhanced_img, gamma=self.low_light_gamma)
            else:
                print("daddfsd"+"si")
                print(brightness)
                input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                h, w = input_img.shape[:2]
                
                # Modele beslemek için yeniden boyutlandırma
                input_img_resized = cv2.resize(input_img, (512, 512))
                input_tensor = self.transform(input_img_resized).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    enhanced_img, _ = self.dce_model(input_tensor)

                # Tensor'dan numpy'a dönüştürme
                enhanced_img = enhanced_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_img = np.clip(enhanced_img, 0, 1)
                enhanced_img = (enhanced_img * 255).astype(np.uint8)
                
                # Orijinal renk formatına dönüştürme
                if len(image.shape) == 3:
                    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
                else:
                    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
                
                # Orijinal boyuta geri döndürme
                enhanced_img = cv2.resize(enhanced_img, (w, h))

                # Ek gamma düzeltmesi
                enhanced_img = self.adjust_gamma(enhanced_img, gamma=self.low_light_gamma)

        elif brightness > self.high_light_threshold:
            # HDR-Net modeli ile aşırı parlak görüntüyü düzeltme
            if self.reverse_dce_model is None:
                print("Reverse-DCE-net modeli yüklü değil, alternatif iyileştirme uygulanıyor...")
                # HDRNet yoksa, tonemap ve retinex kombinasyonu uygula
                enhanced_img = self.apply_tonemap(image)
                enhanced_img = self.apply_retinex(enhanced_img)
                enhanced_img = self.adjust_gamma(enhanced_img, gamma=self.high_light_gamma)
            else:
                print("issooooooooo")
                pre_processed_img = self.adjust_gamma(image, gamma=self.high_light_gamma)  # gamma < 1, örn: 0.5

                input_img = cv2.cvtColor(pre_processed_img, cv2.COLOR_BGR2RGB) if len(pre_processed_img.shape) == 3 else cv2.cvtColor(pre_processed_img, cv2.COLOR_GRAY2RGB)
                h, w = input_img.shape[:2]
    
                input_img_resized = cv2.resize(input_img, (512, 512))
                input_tensor = self.transform(input_img_resized).unsqueeze(0).to(self.device)

            with torch.no_grad():
                 enhanced_img, _ = self.dce_model(input_tensor)
                 enhanced_img = enhanced_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                 enhanced_img = np.clip(enhanced_img, 0, 1)
                 enhanced_img = (enhanced_img * 255).astype(np.uint8)

            if len(image.shape) == 3:
                    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            else:
                    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)

                    enhanced_img = cv2.resize(enhanced_img, (w, h))
    
        else:
            # Normal parlaklıkta olan görüntüler    
                    enhanced_img = image.copy()
                    print("cds")
                    print(brightness)
        
        return enhanced_img
    
    def process_image_file(self, input_path, output_path):
        """Dosyadan görüntüyü oku, iyileştir ve kaydet"""
        # Görüntü okuma
        image = cv2.imread(input_path)
        if image is None:
            print(f"Hata: Görüntü okunamadı - {input_path}")
            return False
        
        # Görüntüyü iyileştir
        enhanced = self.enhance_image(image)
        
        # Sonucu kaydet
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, enhanced)
        return True
    
    def process_batch(self, input_folder, output_folder, extensions=('.jpg', '.jpeg', '.png')):
        """Klasördeki tüm görüntüleri iyileştir"""
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Tüm desteklenen uzantılı dosyaları bul
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_folder.glob(f"*{ext}")))
            image_files.extend(list(input_folder.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"Uyarı: {input_folder} içinde desteklenen görüntü dosyası bulunamadı.")
            return 0
        
        print(f"Toplam {len(image_files)} görüntü işlenecek...")
        success_count = 0
        
        # İlerleme çubuğu ile tüm görüntüleri işle
        for img_path in tqdm(image_files, desc="Görüntü İşleniyor"):
            output_path = output_folder / img_path.name
            if self.process_image_file(str(img_path), str(output_path)):
                success_count += 1
        
        print(f"{success_count}/{len(image_files)} görüntü başarıyla işlendi.")
        return success_count
    
    def process_video(self, input_video, output_video, show_progress=True):
        """Video dosyasını iyileştir"""
        # Video dosyasını aç
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Hata: Video açılamadı - {input_video}")
            return False
        
        # Video özellikleri
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, ~{frame_count} kare")
        
        # Video yazıcı
        os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        processed_frames = 0
        pbar = tqdm(total=frame_count, desc="Video İşleniyor") if show_progress else None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Kareyi iyileştir
            enhanced_frame = self.enhance_image(frame)
            
            # Kareyi video dosyasına yaz
            out.write(enhanced_frame)
            
            processed_frames += 1
            if pbar:
                pbar.update(1)
        
        # Kaynakları serbest bırak
        cap.release()
        out.release()
        if pbar:
            pbar.close()
        
        print(f"Video işleme tamamlandı. {processed_frames} kare işlendi.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Düşük Işık ve Parlak Görüntü İyileştirme Sistemi')
    parser.add_argument('--input', required=True, help='Giriş dosyası veya klasörü')
    parser.add_argument('--output', required=True, help='Çıkış dosyası veya klasörü')
    parser.add_argument('--dce_model', default='Epoch99.pth', help='DCE-Net Model dosyası (varsayılan: Epoch99.pth)')
    parser.add_argument('--hdrnet_model', default='hdrnet_trained.pth', help='HDRNet Model dosyası (varsayılan: hdrnet_trained.pth)')
    parser.add_argument('--type', choices=['image', 'video', 'batch'], required=True, help='İşlem tipi (image, video, batch)')
    parser.add_argument('--low_threshold', type=int, default=80, help='Düşük ışık eşik değeri (varsayılan: 80)')
    parser.add_argument('--high_threshold', type=int, default=100, help='Yüksek ışık eşik değeri (varsayılan: 180)')
    parser.add_argument('--low_gamma', type=float, default=0.7, help='Düşük ışık gamma değeri (varsayılan: 0.7)')
    parser.add_argument('--high_gamma', type=float, default=0.07, help='Yüksek ışık gamma değeri (varsayılan: 2.2)')
    
    args = parser.parse_args()
    
    # İyileştirici oluştur
    enhancer = ImageEnhancer(args.dce_model, args.hdrnet_model)
    enhancer.set_parameters(args.low_threshold, args.high_threshold, args.low_gamma, args.high_gamma)
    
    # İşlem tipine göre çalıştır
    if args.type == 'image':
        success = enhancer.process_image_file(args.input, args.output)
        if success:
            print(f"Görüntü başarıyla iyileştirildi: {args.output}")
        else:
            print("Görüntü işleme başarısız oldu.")
    
    elif args.type == 'video':
        success = enhancer.process_video(args.input, args.output)
        if success:
            print(f"Video başarıyla iyileştirildi: {args.output}")
        else:
            print("Video işleme başarısız oldu.")
    
    elif args.type == 'batch':
        count = enhancer.process_batch(args.input, args.output)
        print(f"İşlem tamamlandı: {count} görüntü iyileştirildi.")

    # Command for the run
    # python video_commit_updated.py --input footage.jpg --output output.jpg --type image  # tek bir görüntü
    # python video_commit_updated.py --input ./giris_klasoru --output ./cikis_klasoru --type batch  # tüm görüntüler
    # python video_commit_updated.py --input footage7.jpg --output output.jpg --type image --low_threshold 70 --high_threshold 190 --low_gamma 0.6 --high_gamma 2.0  # parametrelerle
    # python video_commit_updated.py --input video.mp4 --output output.mp4 --type video  # video işleme