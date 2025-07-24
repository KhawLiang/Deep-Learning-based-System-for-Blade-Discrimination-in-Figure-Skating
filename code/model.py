import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim=256):
        super(ImageEncoder, self).__init__()
        # 加载在 ImageNet 上预训练的 ResNet50 模型
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉最后的全连接层
        # 映射层，将 ResNet50 的结果大小 2046 转成 proj_dim
        self.projection = nn.Linear(2048, proj_dim)
        # 归一化层
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        feat = self.resnet(x)           # [batch_size, 2048]
        emb  = self.projection(feat)    # [batch_size, 256]
        emb  = self.layer_norm(emb)     # 归一化
        
        return emb


# 关键点编码器
class KeypointEncoder(nn.Module):
    def __init__(self, input_dim=57, proj_dim=256):
        super(KeypointEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, keypoints):
        x = F.relu(self.fc1(keypoints))  # [batch_size, 128]
        x = self.fc2(x)                  # [batch_size, 256]
        x = self.layer_norm(x)           # 归一化

        return x
    
# 角度编码器
class AngleEncoder(nn.Module):
    def __init__(self, input_dim=1, proj_dim=256):
        super(AngleEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, proj_dim)
        # 归一化层
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, angle):
        #print(angle.shape)
        x = self.fc1(angle)  # [batch_size, 128]
        x  = self.fc2(x)     # [batch_size, 256]
        emb  = self.layer_norm(x)     # 归一化
        
        return emb


# 融合+分类器
class EdgeClassifier(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128, num_classes=3):
        super(EdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self,img_emb, kp_emb, angle_emb):
        x = torch.cat([img_emb, kp_emb, angle_emb], dim=-1)  # [batch_size, 768]
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)                      # [batch_size, 3]

        return logits