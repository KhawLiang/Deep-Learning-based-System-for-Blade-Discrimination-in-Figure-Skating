import torch
from model import ImageEncoder, KeypointEncoder, AngleEncoder, EdgeClassifier

class BladeDiscriminator:
    def __init__(self, device="cpu"):
        self.device = device
        # 初始化模型
        self.image_encoder    = ImageEncoder().to(device)
        self.keypoint_encoder = KeypointEncoder().to(device)
        self.angle_encoder    = AngleEncoder().to(device)
        self.edge_classifier  = EdgeClassifier().to(device)

        # 加载训练好的模型
        image_enc_state = torch.load(r"model\image_enc.pth", map_location=device, weights_only=True)
        self.image_encoder.load_state_dict(image_enc_state)

        key_enc_state = torch.load(r"model\keypoint_enc.pth", map_location=device, weights_only=True)
        self.keypoint_encoder.load_state_dict(key_enc_state)

        angle_enc_state = torch.load(r"model\angle_enc.pth", map_location=device, weights_only=True)
        self.angle_encoder.load_state_dict(angle_enc_state)

        classifier_state = torch.load(r"model\classifier.pth", map_location=device, weights_only=True)
        self.edge_classifier.load_state_dict(classifier_state)

    def predict(self, features):
        image = features[0].to(self.device)
        keypoints = features[1].to(self.device)
        angle = features[2].to(self.device)
        
        # 评估模式
        self.image_encoder.eval()
        self.keypoint_encoder.eval()
        self.angle_encoder.eval()
        self.edge_classifier.eval()

        with torch.no_grad():
            # forward
            img_emb = self.image_encoder(image)       
            kp_emb = self.keypoint_encoder(keypoints)   
            angle_emb = self.angle_encoder(angle) 

            outputs = self.edge_classifier(img_emb, kp_emb, angle_emb)

            _, pred = outputs.max(1)

        return pred