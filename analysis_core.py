import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from skimage import morphology
from skimage.measure import label


# ==================== 1. Actin 分析逻辑 ====================

def fractal_dimension(img):
    Z = (img > 0).astype(np.uint8) * 255
    p = min(Z.shape)
    if p < 2: return 0
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(np.log2(n), 1, -1).astype(int)
    counts = []
    for size in sizes:
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
                            np.arange(0, Z.shape[1], size), axis=1)
        counts.append(np.sum(S > 0))
    if len(counts) < 2: return 0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def analyze_actin(actin_img, scale=1.61):
    gray = cv2.medianBlur(actin_img, 3)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    binary = morphology.remove_small_objects(thresh.astype(bool), min_size=700).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feat = {k: 0.0 for k in
            ['Area',  'Perimeter',  'Circularity', 'Radius min', 'Radius max', 'Elongation',
             'Eccentricity', 'Fractal dimension']}

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area > 0:
            feat['Area'] = area / (scale ** 2)
            feat['Perimeter'] = peri / scale
            feat['Circularity'] = 4 * np.pi * area / (peri ** 2 + 1e-8)
            feat['Fractal dimension'] = fractal_dimension(binary)
            if len(cnt) >= 5:
                (_, axes, _) = cv2.fitEllipse(cnt)
                feat['Radius min'], feat['Radius max'] = min(axes) / scale, max(axes) / scale
                feat['Elongation'] = max(axes) / (min(axes) + 1e-8)
                feat['Eccentricity'] = np.sqrt(1 - (min(axes) ** 2) / (max(axes) ** 2 + 1e-8))
    return feat


# ==================== 2. YAP 分析逻辑 ====================

def analyze_yap(yap_img, dna_img, actin_img):
    _, t_actin = cv2.threshold(cv2.medianBlur(actin_img, 3), 5, 255, cv2.THRESH_BINARY)
    mask_actin = morphology.remove_small_objects(t_actin.astype(bool), min_size=700).astype(np.uint8) * 255
    _, t_dna = cv2.threshold(dna_img, 5, 255, cv2.THRESH_BINARY)
    mask_dna = morphology.remove_small_objects(t_dna.astype(bool), min_size=200).astype(np.uint8) * 255

    mask_dna = cv2.bitwise_and(mask_dna, mask_actin)
    f_dna = np.sum(cv2.bitwise_and(yap_img, yap_img, mask=mask_dna))
    f_total = np.sum(cv2.bitwise_and(yap_img, yap_img, mask=mask_actin))
    a_dna, a_total = cv2.countNonZero(mask_dna), cv2.countNonZero(mask_actin)

    d_state = f_dna / (a_dna + 1e-8)
    c_state = (f_total - f_dna) / (a_total - a_dna + 1e-8)
    return {
         'yap n/c ratio': d_state / (c_state + 1e-8),
    }


# ==================== 3. 群体分析逻辑 ====================

def analyze_mass(whole_dna, whole_actin, scale=1.61):
    _, t_dna = cv2.threshold(whole_dna, 20, 255, cv2.THRESH_BINARY)
    m_dna = morphology.remove_small_objects(t_dna.astype(bool), min_size=400).astype(np.uint8) * 255
    count = np.max(label(m_dna > 0))
    area_mm2 = (whole_dna.shape[0] * whole_dna.shape[1]) / (scale ** 2)
    density = count / area_mm2 if area_mm2 > 0 else 0

    angles = [cv2.fitEllipse(c)[2] for c in cv2.findContours(m_dna, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if
              len(c) >= 5]
    align = 0
    if angles:
        theta = np.deg2rad(angles)
        align = np.sqrt(np.sum(np.cos(2 * theta)) ** 2 + np.sum(np.sin(2 * theta)) ** 2) / len(theta)

    _, t_actin = cv2.threshold(cv2.medianBlur(whole_actin, 3), 10, 255, cv2.THRESH_BINARY)
    coverage = np.sum(t_actin > 0) / (t_actin.shape[0] * t_actin.shape[1])
    return {'Local cell density': density,  'alignment consistency': align}


# ==================== 4. 命运预测模型 ====================

class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            last = h
        layers.append(nn.Linear(last, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FatePredictor:
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.loaded = False
        self.model = None
        self.sd = None

    def load(self):
        if self.loaded: return
        try:
            self.sd = torch.load(self.scaler_path, map_location='cpu')
            # 确保均值和方差也是 float32，防止运算时自动变为 float64
            self.sd['mean_'] = np.array(self.sd['mean_'], dtype=np.float32)
            self.sd['scale_'] = np.array(self.sd['scale_'], dtype=np.float32)

            self.model = FlexibleMLP(len(self.sd['features']), [128, 64, 32], 3)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            print(f"Model Load Error: {e}")
            raise e

    def predict(self, features):
        self.load()

        # 1. 提取特征并转换为 float32 的 Numpy 数组
        row = [features.get(f.strip(), 0.0) for f in self.sd['features']]
        X = np.array([row], dtype=np.float32)

        # 2. 标准化 (确保运算在 float32 下进行)
        scale_safe = np.where(self.sd['scale_'] == 0, 1.0, self.sd['scale_'])
        X = (X - self.sd['mean_']) / scale_safe

        # 3. 转换为 Tensor 并强制转为 float (float32)
        # 关键修改：添加了 .float()
        tensor_x = torch.from_numpy(X).float()

        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        return str(probs.argmax()), probs
