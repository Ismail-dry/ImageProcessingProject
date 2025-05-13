import sys
import os
import cv2
import numpy as np
import torch # Derin öğrenme modellerini eğitmek ve çalıştırmak için 
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QSlider, QGroupBox, QFormLayout, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms # Görüntüler üzerinde ön işleme (örneğin boyutlandırma, normalize etme, tensöre çevirme) yapmak için

# Import models conditionally to handle potential import errors
try:
    from dce_model import DCE_net, ReverseDCEUNet
    DCE_AVAILABLE = True
except ImportError:
    print("Warning: DCE models could not be imported. Alternative enhancement methods will be used.")
    DCE_AVAILABLE = False

class ImageEnhancerGUI(QWidget):
    def __init__(self, dce_model_path='Epoch99.pth', reverse_dce_model_path='reverse_dce.pth'):
        super().__init__()
        self.setWindowTitle("Görüntü İyileştirme - PyQt5")
        self.resize(1200, 600)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models as None first
        self.dce_model = None
        self.reverse_model = None
        
        # Load DCE model
        if DCE_AVAILABLE:
            try:
                self.dce_model = DCE_net().to(self.device)
                dce_state_dict = torch.load(dce_model_path, map_location=self.device)
                self.dce_model.load_state_dict(dce_state_dict)
                self.dce_model.eval()
                print(f"DCE-Net modeli başarıyla yüklendi: {dce_model_path}")
            except Exception as e:
                print(f"DCE-Net modeli yüklenirken hata oluştu: {e}")
                print("DCE-Net modeli kullanılamayacak")
                self.dce_model = None
        
            # Load Reverse DCE model
            try:
                self.reverse_model = ReverseDCEUNet().to(self.device)
                reverse_dce_state_dict = torch.load(reverse_dce_model_path, map_location=self.device)
                self.reverse_model.load_state_dict(reverse_dce_state_dict)
                self.reverse_model.eval()
                print(f"Reverse-DCE-Net modeli başarıyla yüklendi: {reverse_dce_model_path}")
            except Exception as e:
                print(f"Reverse-DCE-Net modeli yüklenirken hata oluştu: {e}")
                print("Reverse-DCE-Net modeli kullanılamayacak")
                self.reverse_model = None

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.low_light_threshold = 80
        self.high_light_threshold = 180
        self.low_light_gamma = 0.9
        self.high_light_gamma = 0.7

        self.original_img = None
        self.enhanced_img = None

        self.init_ui()

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

    def init_ui(self):
        # Görseller
        self.orig_label = QLabel("Orijinal")
        self.enh_label = QLabel("İyileştirilmiş")
        self.orig_label.setFixedSize(500, 400)
        self.enh_label.setFixedSize(500, 400)
        self.orig_label.setStyleSheet("background-color: black")
        self.enh_label.setStyleSheet("background-color: black")

        # Butonlar
        load_btn = QPushButton("Görüntü Yükle")
        enhance_btn = QPushButton("İyileştir")
        save_btn = QPushButton("Kaydet")

        load_btn.clicked.connect(self.load_image)
        enhance_btn.clicked.connect(self.enhance_image)
        save_btn.clicked.connect(self.save_image)

        # Ayar kutuları
        self.low_thres_spin = QSpinBox()
        self.low_thres_spin.setRange(0, 255)
        self.low_thres_spin.setValue(self.low_light_threshold)

        self.high_thres_spin = QSpinBox()
        self.high_thres_spin.setRange(0, 255)
        self.high_thres_spin.setValue(self.high_light_threshold)

        form_layout = QFormLayout()
        form_layout.addRow("Düşük Işık Eşiği", self.low_thres_spin)
        form_layout.addRow("Yüksek Işık Eşiği", self.high_thres_spin)

        param_box = QGroupBox("Parametreler")
        param_box.setLayout(form_layout)

        # Yerleşim
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.orig_label)
        img_layout.addWidget(self.enh_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(enhance_btn)
        btn_layout.addWidget(save_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(param_box)

        self.setLayout(main_layout)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Image Files (*.png *.jpg *.jpeg)")
        if fname:
            self.original_img = cv2.imread(fname)
            self.display_image(self.original_img, self.orig_label)

    def enhance_image(self):
        if self.original_img is None:
            return

        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        image = self.original_img.copy()

        low = self.low_thres_spin.value()
        high = self.high_thres_spin.value()

        if brightness < low:
            # DCE-Net ile iyileştirme (karanlık görüntüler)
            if self.dce_model is None:
                print("DCE-Net modeli yüklü değil, alternatif iyileştirme uygulanıyor...")
                # DCE-Net yoksa, basit bir histogram eşitleme uygula
                enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                enhanced_img[:,:,0] = cv2.equalizeHist(enhanced_img[:,:,0])
                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_YUV2BGR)
                enhanced_img = self.adjust_gamma(enhanced_img, gamma=self.low_light_gamma)
            else:
                print(f"DCE-Net modeli ile karanlık görüntü iyileştirme (parlaklık: {brightness})")
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

        elif brightness > high:
            # Reverse DCE-Net ile iyileştirme (aşırı aydınlık görüntüler)
            if self.reverse_model is None:
                print("Reverse-DCE-Net modeli yüklü değil, alternatif iyileştirme uygulanıyor...")
                # Reverse-DCE-Net yoksa, tonemap ve retinex kombinasyonu uygula
                enhanced_img = self.apply_tonemap(image)
                enhanced_img = self.apply_retinex(enhanced_img)
                enhanced_img = self.adjust_gamma(enhanced_img, gamma=self.high_light_gamma)
            else:
                print(f"Reverse-DCE-Net modeli ile aydınlık görüntü iyileştirme (parlaklık: {brightness})")
                pre_processed_img = self.adjust_gamma(image, gamma=self.high_light_gamma)  # gamma < 1, örn: 0.5
                
                input_img = cv2.cvtColor(pre_processed_img, cv2.COLOR_BGR2RGB) if len(pre_processed_img.shape) == 3 else cv2.cvtColor(pre_processed_img, cv2.COLOR_GRAY2RGB)
                h, w = input_img.shape[:2]
                
                input_img_resized = cv2.resize(input_img, (512, 512))
                input_tensor = self.transform(input_img_resized).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    enhanced_output = self.reverse_model(input_tensor)
                    if isinstance(enhanced_output, tuple):
                        enhanced_img = enhanced_output[0]
                    else:
                        enhanced_img = enhanced_output
                        
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
            print(f"Normal parlaklık, iyileştirme gerekmedi (parlaklık: {brightness})")

        self.enhanced_img = enhanced_img
        self.display_image(enhanced_img, self.enh_label)

    def save_image(self):
        if self.enhanced_img is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet", "", "JPEG (*.jpg);;PNG (*.png)")
        if fname:
            cv2.imwrite(fname, self.enhanced_img)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pix)

    def adjust_gamma(self, image, gamma):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImageEnhancerGUI()
    gui.show()
    sys.exit(app.exec_())