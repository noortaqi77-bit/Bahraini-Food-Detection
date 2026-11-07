
# Bahraini Food Detection
An object-detection project that recognizes traditional Bahraini foods and drinks in images.


App : https://bahraini-food-detection.streamlit.app/

## 1. Problem statement 
Tourists visiting Bahrain often cannot identify the local dishes they encounter at restaurants, food markets and traditional gatherings. Searching online or asking someone what the dishes are is slow and inconvenient.


### Classes
Arabic coffee, Balaleet, Chicken Majboos, Halwa, Harees, Khanfaroosh, Meat Majboos, Tikka, Zalabiya, Karak


## Dataset
- Images were collected and annotated with bounding boxes in Roboflow, then exported in YOLO format.
- The dataset was split into train, validation and test sets before training (70% / 20% / 10%).
- Augmentation was applied to the training set only.
  

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

#### Class imbalance
Tikka and Khanfaroosh have many more objects because they were annotated piece by piece, while the other classes were annotated as one box per dish.

### Training
- PretrainedvModel: YOLO26s
- Fine-tuned v1: 10 epochs
- Fine-tuned v2: [epochs]

### Results in test set

| Metric | v1 | v2 |
|---|---|---|
| Precision | 0.757 | |
| Recall | 0.769 | |
| mAP50 | 0.801 | |
| mAP50-95 | 0.617 | |

#### for per class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Arabic coffee | 0.266 | 1.00 | 0.332 | 0.166 |
| Balaleet | 1.00 | 0.397 | 0.835 | 0.666 |
| Chicken Majboos | 0.581 | 1.00 | 0.995 | 0.895 |
| Halwa | 1.00 | 0.684 | 0.856 | 0.723 |
| Khanfaroosh | 1.00 | 0.958 | 0.995 | 0.833 |
| Meat Majboos | 0.89 | 1.00 | 0.995 | 0.863 |
| Tikka | 0.342 | 0.333 | 0.346 | 0.208 |
| Zalabiya | 0.968 | 1.00 | 0.995 | 0.741 |
| Karak | 0.765 | 0.55 | 0.859 | 0.463 |

### Error Analysis 

