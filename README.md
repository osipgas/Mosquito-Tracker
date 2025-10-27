# 🦟 Intelligent Mosquito Tracking System

A computer vision project for **detecting and tracking mosquitoes** in videos using **motion analysis**, **adaptive thresholding**, and **Gaussian correlation**.  
The system processes video frames, identifies mosquito movements, filters noise, and maintains persistent tracking IDs over time.

---

## 🚀 Features

- 🧮 **Adaptive Thresholding:** Dynamically adjusts the detection threshold based on noise statistics.
- 🎯 **Gaussian Motion Filtering:** Enhances oval mosquito-like motion patterns while suppressing background noise.
- 🧠 **Persistent Object Tracking:** Tracks each mosquito across frames with velocity prediction and ID consistency.
- ⚙️ **GPU Acceleration:** Supports `mps` (Apple Silicon) and `cuda` devices.
- 📹 **Video Integration:** Processes videos directly via `moviepy` and optionally saves annotated outputs.

---

## 🧩 Project Structure

mosquito-tracker/
│
├── mosquito_detector.py     # Detection and motion analysis logic
├── mosquito_tracker.py      # Tracking system with ID persistence
├── visualization_utils.py   # Drawing boxes and tracking visualization
└── demo_usage.ipynb         # Example notebook demonstrating full 

---

## 🧠 System Overview

### 1. Detection (`MosquitoDetector`)
- Converts frames to grayscale tensors.
- Computes motion maps between consecutive frames.
- Uses **Gaussian kernels** to highlight mosquito-shaped motion.
- Applies **adaptive thresholding** to distinguish real mosquito movements from noise.

### 2. Tracking (`MosquitoTracker`)
- Associates detected mosquito centers between frames.
- Predicts motion for temporarily lost targets.
- Removes duplicates and assigns consistent display IDs.
- Handles track aging and deletion of stale entries.

### 3. Visualization
- Draws bounding boxes and tracking IDs over detected mosquitoes.
- Optional video saving with OpenCV (`cv2.VideoWriter`).

---

## 🔧 Installation

```bash
git clone https://github.com/osipgas/Mosquito-Tracker.git
cd mosquito-tracker
pip install -r requirements.txt