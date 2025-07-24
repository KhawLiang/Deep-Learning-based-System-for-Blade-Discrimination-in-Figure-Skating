import torch
import time
import sys
from PIL import Image
import torchvision.transforms as T

from get_landmarks_angle import GETLANDMARKS
from blade_discriminator import BladeDiscriminator

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap

def pack_feature(image_path, keypoints, angle):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
        ])
    
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    keypoints = torch.tensor(keypoints, dtype=torch.float32).view(1, -1)
    angle = torch.tensor(angle, dtype=torch.float32).view(1, -1)

    return [image, keypoints, angle]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("内外刃判别系统")
        self.resize(1400, 800)

        # model & detector
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = BladeDiscriminator(device)
        self.feature_detector = GETLANDMARKS()
        self.image_path = None

        # UI elements
        self.image_label = QLabel("无图像显示")
        self.image_label.setFixedSize(1000, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel("预测: N/A")

        load_btn = QPushButton("加载图像")
        load_btn.clicked.connect(self.load_image)

        predict_btn = QPushButton("预测")
        predict_btn.clicked.connect(self.on_predict)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, alignment=Qt.AlignHCenter)
        layout.addWidget(load_btn)
        layout.addWidget(predict_btn)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            pix = QPixmap(path).scaled(600, 600, aspectRatioMode=1)
            self.image_label.setPixmap(pix)
            self.image_label.adjustSize()
            self.result_label.setText("预测: N/A")
    
    def on_predict(self):
        if not self.image_path:
            self.result_label.setText("请先加载图像！！！")
            return
        
        start_time = time.time()
        
        # 提取图像中的人体关键点和冰刀角度
        keypoints, angle = self.feature_detector.get_landmarks(self.image_path)
        
        # 整理 & 打包特征
        features = pack_feature(self.image_path, keypoints, angle)

        # 模型预测
        pred = self.discriminator.predict(features)
        labels = {0: '内刃', 1: '外刃', 2: '平刃'}

        end_time = time.time()

        # 计算执行时间
        elapsed_time = end_time - start_time
        print(f"预测时间：{elapsed_time:.2f}")

        self.result_label.setText(f"预测: {labels[pred.item()]}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())