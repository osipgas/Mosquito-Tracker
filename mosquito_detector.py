# Core detection and motion analysis logic 

import numpy as np
import torch
import torch.nn.functional as F


class AdaptiveThreshold:
    def __init__(self, rate=0.1, percentile=0.95):
        self.rate = rate
        self.percentile = percentile
        self.threshold = None

    # Threshold supposed to be a little bit higher that highest possible noise
    # Which is why we using "percentile", we want to get rid of mosquitos, which values are the biggest
    def calculate_threshold(self, values):
        noise_level = torch.quantile(values, self.percentile)
        resonable_threshold = noise_level * 3
        return resonable_threshold

    def adapt(self, values):
        resonable_threshold = self.calculate_threshold(values)
        # If the threshold is not yet set, initialize it immediately
        if self.threshold is None:
            self.threshold = resonable_threshold
        else:
            self.threshold += (resonable_threshold - self.threshold) * self.rate
        return self.threshold

class MosquitoDetector:
    def __init__(self, adaptive_threshold, kernel_size=12, stride=2, sigma=3.0, device="mps"):
        self.adaptive_threshold = adaptive_threshold
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.fixed_initial_threshold = 3
        self.kernel = self.create_gaussian_kernel(sigma)
            
    def create_gaussian_kernel(self, sigma=3.0):
        # Method creates gaussian kenrel, a circle birght in the center and dim at the corners
        # Mosquitos are oval shaped, gaussian circle supposed to maximize oval mosquito shape and ignore randomly shaped noise
        center = self.kernel_size // 2
        x, y = np.meshgrid(np.arange(self.kernel_size) - center, np.arange(self.kernel_size) - center)
        kernel = np.exp(- (x**2 + y**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)  # Для корреляции в свёртке
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel.to(self.device)

    @staticmethod
    def grayscale(frame):
        gray = frame.float().mean(dim=2)
        return gray
    
    def get_optimized_motion_map(self, frame_buffer):
        # Grascaling 3 last frames
        gray_frame_cur = frame_buffer[-1]
        gray_frame_prev = frame_buffer[-2]
        gray_frame_prev_prev = frame_buffer[-3]
        
        # Calculating motion maps between current frame and previos one, and between current and one before previos
        diff1 = torch.abs(gray_frame_cur - gray_frame_prev)
        diff2 = torch.abs(gray_frame_cur - gray_frame_prev_prev)

        # Choosing the minimum between every motion:
        # Motions from previos frames disappears while motion from current frame remain
        # As a bonus phantom noises fade away because we choosing the minimum
        motion_map = torch.minimum(diff1, diff2)
        motion_map[motion_map < self.fixed_initial_threshold] = 0
        return motion_map
    
    def evaluate_motions(self, motion_map):
        # Evaluating motions with Gaussian kernel, which supposed to maximize mosquito motions due to their oval shape.
        motion_map = motion_map.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            match_map = F.conv2d(input=motion_map, weight=self.kernel, stride=self.stride)

        # Collecting coordinates of motions which passed threshold
        coords = torch.nonzero(match_map > 0, as_tuple=False)  # Сначала все ненулевые координаты
        conf = match_map[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]  # Значения по этим координатам

        # Calibrating threshold
        self.adaptive_threshold.adapt(conf)  # Адаптируем порог только по отфильтрованным значениям
        threshold = self.adaptive_threshold.threshold

        # Фильтруем координаты и значения по порогу
        mask = conf > threshold
        coords = coords[mask]
        conf = conf[mask]

        # y, x
        return coords[:, 2:], conf
    
    def coords_to_boxes(self, coords):
        y1 = coords[:, 0] * self.stride
        x1 = coords[:, 1] * self.stride
        x2 = x1 + self.kernel_size
        y2 = y1 + self.kernel_size
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes
    
    def detect(self, frame_buffer):
        motion_map = self.get_optimized_motion_map(frame_buffer)
        coords, conf = self.evaluate_motions(motion_map)
        boxes = self.coords_to_boxes(coords)
        # boxes, conf = self.apply_nms(boxes.float(), conf.float())
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        return centers, conf