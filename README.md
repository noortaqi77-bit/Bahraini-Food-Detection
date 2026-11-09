# Bahraini Food Detection

An object-detection project that recognizes traditional Bahraini foods and drinks in images, built by fine-tuning **YOLO26s** and deployed as a Streamlit web app.

**Live app:** https://bahraini-food-detection.streamlit.app/

## 1. Problem Statement

Tourists visiting Bahrain often can't identify the local dishes they see at restaurants, food markets and traditional gatherings. Searching online or asking someone is slow and inconvenient. This project lets a user upload a photo and get the dishes in it detected and labeled.

### Classes (10)

Arabic coffee, Balaleet, Chicken Majboos, Halwa, Harees, Khanfaroosh, Meat Majboos, Tikka, Zalabiya, Karak

## 2. Dataset

- Images were collected and annotated with bounding boxes in [Roboflow](https://roboflow.com/), then exported in YOLO format.
- Before augmentation, the data was split 70% / 20% / 10% into train, validation and test sets.
- Augmentation was applied to the training set only.

| Split | Images |
|---|---|
| Train (after augmentation) | 546 |
| Validation | 52 |
| Test | 26 |

### Objects per class

| Class | Objects |
|---|---|
| Tikka | 235 |
| Khanfaroosh | 139 |
| Halwa | 39 |
| Zalabiya | 35 |
| Karak | 34 |
| Arabic coffee | 29 |
| Balaleet | 29 |
| Chicken Majboos | 28 |
| Meat Majboos | 27 |
| Harees | 27 |

Total: 622 annotations, 2.4 per image on average.

**Class imbalance:** Tikka and Khanfaroosh have many more objects because each piece was annotated separately. The other classes were annotated with one box per dish.

## 3. Training

- **Base model:** `yolo26s.pt` (pretrained YOLO26s, ~9.5M parameters)
- **v1:** fine-tuned for 5 epochs at 640×640
- **v2:** training continued from the v1 weights for 10 more epochs (15 in total)
- **Hardware:** Google Colab, Tesla T4 GPU
- The final model is saved as `best.pt`.

## 4. Results

### Overall: validation set

| Metric | v1 (5 epochs) | v2 (15 epochs) |
|---|---|---|
| Precision | 0.688 | 0.804 |
| Recall | 0.796 | 0.809 |
| mAP50 | 0.791 | 0.849 |
| mAP50-95 | 0.566 | 0.661 |

Training for longer improved every metric, most of all precision (+0.12) and mAP50-95 (+0.10).

### Final model (v2): test set

| Metric | Value |
|---|---|
| Precision | 0.726 |
| Recall | 0.881 |
| mAP50 | 0.791 |
| mAP50-95 | 0.591 |

### Per class: test set (v2)

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Arabic coffee | 1 | 0.125 | 1.00 | 0.199 | 0.119 |
| Balaleet | 5 | 1.00 | 0.826 | 0.995 | 0.775 |
| Chicken Majboos | 1 | 0.701 | 1.00 | 0.995 | 0.895 |
| Halwa | 4 | 0.936 | 1.00 | 0.995 | 0.773 |
| Khanfaroosh | 17 | 0.990 | 1.00 | 0.995 | 0.734 |
| Meat Majboos | 3 | 0.854 | 1.00 | 0.995 | 0.907 |
| Tikka | 3 | 0.108 | 0.333 | 0.050 | 0.035 |
| Zalabiya | 4 | 1.00 | 1.00 | 0.995 | 0.691 |
| Karak | 6 | 0.820 | 0.766 | 0.901 | 0.384 |

> Harees has no instances in the test split, so it isn't listed above. On the validation set it reached P 0.884, R 1.00 and mAP50 0.926.

## 5. Error Analysis

These findings come from the normalized confusion matrix on the validation set and from the per-class metrics.

- **Tikka is the weakest class.** Validation recall is only 0.25, and test mAP50 is 0.05. Tikka is annotated piece by piece, often as small, overlapping skewers, and the model produces many false-positive Tikka boxes on the background. Most background false positives belong to this class.
- **Chicken Majboos vs. Meat Majboos.** The two dishes look almost the same (rice with meat on top). On the validation set, half of the Chicken Majboos dishes were predicted as Meat Majboos, and a quarter of the Meat Majboos dishes were predicted as Harees or Chicken Majboos.
- **Arabic coffee and Karak** have low precision. Both are drinks served in small cups, and the model sometimes fires on cups that aren't the target drink. The test set also has only one Arabic coffee instance, so its metrics are unstable.
- **Small evaluation sets.** Several classes have only 1–5 instances in validation or test, so per-class numbers can swing a lot from a single prediction.
- **Label quality.** One validation image was skipped because its label mixes segmentation and detection rows, and the test labels contain 4 segment rows. Converting these to boxes would clean up evaluation.

### Possible improvements

- Collect more images, especially of Tikka, Majboos (chicken and meat), Arabic coffee and Karak.
- Annotate Tikka consistently, either one box per plate or tighter boxes per skewer.
- Train for more epochs, or try a larger model (YOLO26m).
- Fix the mixed segment/detection labels in Roboflow.

## 6. Web App

The Streamlit app ([app.py](app.py)) loads `best.pt`, takes an uploaded JPG/PNG, runs detection at a 0.25 confidence threshold, and shows the image with the predicted boxes drawn on it.

### Run locally

```bash
git clone https://github.com/noortaqi77-bit/Bahraini-Food-Detection.git
cd Bahraini-Food-Detection
pip install -r requirements.txt
streamlit run app.py
```

## 7. Project Structure

```
├── Notebook.ipynb     # Data download, fine-tuning and evaluation (Colab)
├── app.py             # Streamlit app
├── best.pt            # Fine-tuned YOLO26s weights (final model)
├── yolo26s.pt         # Pretrained base weights
├── requirements.txt   # Python dependencies
└── README.md
```

## 8. Tech Stack

Python · Ultralytics YOLO26 · Roboflow · Streamlit · OpenCV · Pillow · NumPy
