# Deep-Learning-based-System-for-Blade-Discrimination-in-Figure-Skating

This repository contains the implementation of a deep learning system designed to discriminate between inside and outside blade use in figure skating based on video analysis.


## 🧠 Overview

Figure skating blade discrimination is a challenging task that helps in understanding skater technique and performance. This project leverages computer vision and deep learning techniques to analyze video frames of figure skaters and classify whether the skater is using the **inside** or **outside** edge of their blade.

### 🔍 Key Features

- Pose landmark detection using **Mediapipe**
- Deep learning classification model (ResNet50 + MLP)
- Custom dataset for blade edge classification


## 📁 Repository Structure

```
.
├── code/
|   ├── blade_discriminator.py  # load model weights and can be used for prediction.
|   ├── get_landmarks_angle.py  # Python scipts that used Mediapipe Pose model to extraction landmarks and used angle between two vectors formula to calculate foot-ice surface angle.
|   ├── main.py 
|   └── model.py                # Deep Learning model's structure
|
├── models/ # Trained model checkpoints
|   ├── angle_enc.pth           # foot-ice surface angle encoder weights
|   ├── classifier.pth          # classifier weights
|   ├── image_enc.pth           # image encoder weights
|   └── keypoint_enc.pth        # body landmarks encoder weights
|
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```


## ⚙️ Installation

1. **Clone this repo**

```bash
git clone https://github.com/KhawLiang/Deep-Learning-based-System-for-Blade-Discrimination-in-Figure-Skating.git
cd Deep-Learning-based-System-for-Blade-Discrimination-in-Figure-Skating
```

2. **(Optional) Create a virtual environment**

```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```
pip install -r requirements.txt
```
> ⚠️ **Note:** `torch` and `torchvision` are not included in `requirements.txt` because their installation depends on your system and whether you want GPU acceleration.  
> You can install them by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).


## 📊 Dataset

We use a custom-built dataset to train the model, dataset available at the following repository:

🔗 [SkaterDataset](https://github.com/KhawLiang/SkaterDataset.git)

This dataset contains annotated video frames and corresponding labels indicating whether the skater is using the **inside blade** or **outside blade** during the frame.


# 📬 Contact

Email: [khawliang2109@gmail.com]
GitHub: @KhawLiang