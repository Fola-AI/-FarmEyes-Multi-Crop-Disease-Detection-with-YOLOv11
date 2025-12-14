# 🌱 FarmEyes — Multi-Crop Disease Detection with YOLOv11

FarmEyes is a computer vision project that uses **YOLOv11** to detect and localize **crop diseases** across **Tomato, Cocoa, and Cassava** using bounding-box annotations.  
The goal is to provide an accurate, scalable baseline for **AI-assisted agricultural disease monitoring**.

---

## 📌 Project Overview

Early detection of crop diseases is critical for improving yield and reducing losses. This project trains a **single unified object detection model** capable of identifying multiple disease types across different crops from images captured in real-world conditions.

**Crops covered:**
- 🍅 Tomato
- 🌰 Cocoa
- 🌿 Cassava

**Total disease classes:** 6 (2 per crop)

---

## 🧠 Model & Approach

- **Model:** YOLOv11s (Ultralytics)
- **Task:** Object Detection
- **Annotation type:** Bounding boxes
- **Framework:** PyTorch (Ultralytics)
- **Hardware:** Apple Silicon (MPS)

A single model is trained to learn shared visual features across crops while distinguishing fine-grained disease patterns.

Installation was done using Pytorch, installation instructions can be found here:
**https://github.com/mrdbourke/pytorch-apple-silicon?tab=readme-ov-file**

---

## 📂 Dataset

- ~300 manually annotated images
- Labeled using **Roboflow**
- Offline augmentation applied (horizontal flips only)
- Train / Validation / Test split

Bounding boxes tightly cover disease regions while allowing natural overlap where leaves intersect.

---

## ⚙️ Training Configuration

- **Optimizer:** SGD  
- **Epochs:** 100  
- **Image size:** 640×640  
- **Early stopping:** Disabled  
- **Augmentation:** Light (mosaic + flip)  
- **IoU (training):** 0.5  

---

## 📊 Final Model Performance (Validation Set)

| Metric | Value |
|------|------|
| Precision | 0.77 |
| Recall | 0.73 |
| **mAP@0.50** | **0.74** |
| mAP@0.50–0.95 | 0.45 |

---

## 🧪 Running Inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    "path/to/test/image.jpg",
    conf=0.2,
    iou=0.5,
    save=True
)
```

Predictions are saved to:
```
runs/detect/predict/
```

---

## 📁 Repository Structure

```
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/
│   └── detect/
├── notebooks/
│   └── training_and_evaluation.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Key Takeaways

- Single model handles multi-crop disease detection effectively
- Strong performance achieved with limited data
- Further gains depend mainly on adding more labeled images

---

## 🔮 Future Work

- Add more samples for underrepresented classes
- Explore instance segmentation
- Deploy to mobile or edge devices
- Expand to more crops and diseases

---

## 🧑‍💻 Author

**Afolabi Ajao**  
Focus areas: Computer Vision, Applied AI, Data Analytics

---

## 📄 License

For research and educational purposes.
