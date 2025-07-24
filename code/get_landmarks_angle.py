import os
import cv2
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp


class GETLANDMARKS:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)


    def detect_pose_landmarks(self, frame):
        """Detects pose landmarks in a given frame.

        Args:
            frame (np.ndarray): Video frame.

        Returns:
            Optional[np.ndarray]: Array of pose landmarks or None if no landmarks detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_world_landmarks and results.pose_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            landmarks = results.pose_landmarks
            return world_landmarks, landmarks

        return None, None
    

    def get_normal_vect_from_three_3d_points(self, p1, p2, p3):
        """
        get plane equation from 3 3d points
        steps:
        1. Calculate vector p1p2 (as vector_1) and vector p1p3 (as vector_2)
            vector_1 = (x2 - x1, y2 - y1, z2 - z1) = (a1, b1, c1).
            vector_2 = (x3 - x1, y3 - y1, z3 - z1) = (a2, b2, c2).
        2. Get normal vector n of this plane by calculate outer product of vector p1p2 and vector p1p3
            vector_1 X vector_2 = (b1 * c2 - b2 * c1) i 
                        + (a2 * c1 - a1 * c2) j 
                        + (a1 * b2 - b1 * a2) k 
                    = ai + bj + ck.
        3. Get d and sub into ax + by + cz = d. Formula of d is d =  - a * x1 - b * y1 - c * z1
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3

        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1

        a = (b1 * c2) - (b2 * c1)
        b = (a2 * c1) - (a1 * c2)
        c = (a1 * b2) - (b1 * a2)

        d = - a * x1 - b * y1 - c * z1

        normal_vector = np.array([a,b,c])
        # print(f"Normal Vector: {normal_vector}")      
        return normal_vector
    

    def calculate_angle_3d(self, norm_vect):
        """
        This function is to calculate angle between left_foot_plane and horizontal plane.
        So, we set y as 1 since the horizontal plane's normal vector direction is pointing upwards.
        normal vector of the horizontal plane: (0, 1, 0). 
        """
        A1, B1, C1 = norm_vect
        A2, B2, C2 = 0, 1, 0
        numerator = (A1 * A2) + (B1 * B2) + (C1 * C2)
        denominator = math.sqrt(A1**2 + B1**2 + C1**2) * math.sqrt(A2**2 + B2**2 + C2**2)

        angle = math.acos(numerator/denominator)
        angle = round(math.degrees(angle), 2)
        # print(angle)
        return 90 - angle


    def get_landmarks(self, image_path):
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        world_3d_landmarks, landmarks = self.detect_pose_landmarks(img)

        if world_3d_landmarks is not None and landmarks is not None:            
            '''
            人体关键点坐标:
            0: Nose
            2: Left eye
            5: Right eye
            11: Left shoulder
            12: Right shoulder
            13: Left elbow
            14: Right elbow
            15: Left wrist
            16: Right wrist
            23: Left hip
            24: Right hip
            25: Left knee
            26: Right knee
            27: Left ankle
            28: Right ankle
            29: Left heel
            30: Right heel
            31: Left foot index
            32: Right foot index
            '''
            keypoints = []

            # store keypoints information
            for idx in [0,2,5,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]:
                coordinate_xw = world_3d_landmarks[idx].x
                coordinate_yw = world_3d_landmarks[idx].y
                coordinate_zw = world_3d_landmarks[idx].z
                coordinate_w = [coordinate_xw, coordinate_yw, coordinate_zw]
                keypoints.append(coordinate_w)

            left_ankle_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                     world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y,
                                     world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].z])
            left_heel_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].x,
                                    world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].y,
                                    world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].z])
            left_foot_index_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                          world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                                          world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z])
                    

            # 计算左脚法向量    
            norm_vect = self.get_normal_vect_from_three_3d_points(left_foot_index_w, left_ankle_w, left_heel_w)

            # 计算左脚冰刀角度
            angle = self.calculate_angle_3d(norm_vect)

        return keypoints, angle


if __name__ == "__main__":
    image_path = r"system\sample\sample1.jpg"
    feature_detector = GETLANDMARKS()
    keypoints, angle = feature_detector.get_landmarks(image_path)

    print('-'*10 + "Keypoints" + '-'*10)
    print(f"keypoints: {keypoints}")
    print('-'*10 + "Keypoints" + '-'*10)
    print()
    print('-'*10 + "Angle" + '-'*10)
    print(f"Angle: {angle}")
    print('-'*10 + "Angle" + '-'*10)