# A-Anti-Interference-Real-time-Segmentation-Network

This repository contains the official PyTorch implementation and the **MSD-SII** dataset for the paper:
> **An Edge-Based Online Mandrel Surface Defect Inspection System under Strong Industrial Interference**

## 🌟 Overview
This repository provides the code and dataset for automated mandrel surface defect inspection. At its core lies the Anti-Interference Real-time Segmentation Network (**AIRSegNet**), tailored to robustly segment submillimeter-scale defects from complex backgrounds on edge devices. 

Deployed on Jetson edge nodes, AIRSegNet achieves an inference speed of **371 FPS**, meeting real-time inspection demands, and attains a mean Intersection over Union (IoU) of **0.57** with an ultralow false positive rate (FPR) of **0.0011**.

## 🏗️ Framework
The overall architecture of the proposed AIRSegNet is shown below:

![image](./Other/AIRSegNet.png)

## 📊 MSD-SII Dataset & Preparation
We are releasing the **Mandrel Surface Defect under Strong Industrial Interference (MSD-SII)** benchmark, the first industrial mandrel defect dataset collected under strong interference conditions.
* **Complexity:** Includes severe defect-like interferences (e.g., water droplets, graphite contamination, oxide scale) to reflect real-world manufacturing challenges.

🔗 **Dataset Download Link:** [Insert your dataset link here, e.g., BaiduNetdisk / Google Drive]

### Directory Structure
After downloading the dataset, please organize it strictly according to the following structure:

```text
01Dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## ⚙️ Installation & Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/NEUMandrelInspection/A-Anti-Interference-Real-time-Segmentation-Network.git
cd A-Anti-Interference-Real-time-Segmentation-Network
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Data Preparation:**
Download the MSD-SII dataset and place it in the `01Dataset` directory.

**4. Training & Evaluation:**
```bash
# Train the model
python train.py 

# Evaluate the model
python test.py 
```
