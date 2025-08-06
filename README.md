# 🍄 Mushroom Classifier

A deep learning project for classifying different types of Vietnamese mushrooms using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

**🏆 Built for OAI 2025 Contest** - This project was developed to participate in the [OAI 2025 Contest](https://oai.hutech.edu.vn)
## 📋 Project Overview

This project implements an image classification system to identify different species of Vietnamese mushrooms. The model uses a CNN architecture to classify mushroom images into 4 distinct categories.

### 🎯 Mushroom Categories

| Category ID | Vietnamese Name | English Name | Description |
|-------------|----------------|--------------|-------------|
| 0 | nấm mỡ | Shiitake Mushrooms | Popular edible mushrooms |
| 1 | bào ngư xám + trắng | Gray & White Oyster Mushrooms | Common oyster mushroom varieties |
| 2 | Đùi gà Baby (cắt ngắn) | Baby Chicken Leg Mushrooms | Small chicken leg mushrooms |
| 3 | linh chi trắng | White Reishi Mushrooms | Medicinal mushrooms |

## 🚀 Features

- **Deep Learning Classification**: CNN architecture with batch normalization and dropout
- **Data Augmentation**: Test Time Augmentation (TTA) for improved predictions
- **Model Persistence**: Automatic model saving and loading
- **Batch Processing**: Efficient batch processing for large datasets
- **Early Stopping**: Prevents overfitting with patience-based callbacks
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## 📁 Project Structure

```
Mushroom-Classifier/
├── data/
│   ├── train/                    # Training dataset
│   │   ├── bào ngư xám + trắng/  # Oyster mushrooms
│   │   ├── linh chi trắng/       # White reishi mushrooms
│   │   ├── nấm mỡ/               # Shiitake mushrooms
│   │   └── Đùi gà Baby (cắt ngắn)/ # Baby chicken leg mushrooms
│   └── test/                     # Test images (001.jpg - 200.jpg)
├── libs/
│   ├── lib1.py                   # Custom utility functions
│   └── lib2.py                   # Additional helper functions
├── output/
│   ├── final_model.keras         # Trained model
│   └── result.csv                # Prediction results
├── main.py                       # Main training and prediction script
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/TuanBew/Mushroom-Classifier.git
   cd Mushroom-Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data structure**
   Ensure your data folders are organized as shown in the project structure above.

## 🔧 Usage

### Training and Prediction

Run the main script to train the model and generate predictions:

```bash
python main.py
```

### Key Parameters

- **Image Size**: 32x32 pixels
- **Batch Size**: 32
- **Epochs**: 80 (with early stopping)
- **Model Architecture**: CNN with 2 convolutional blocks

### Model Architecture

```python
Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])
```

## 📊 Model Features

### Training Features
- **90/10 Train-Validation Split**: Automatic data splitting
- **Data Augmentation**: Horizontal flip, rotation, zoom for TTA
- **Early Stopping**: Monitors loss with patience of 10 epochs
- **Learning Rate Reduction**: Reduces LR by 50% when loss plateaus
- **Model Checkpointing**: Saves best model based on validation loss

### Prediction Features
- **Test Time Augmentation (TTA)**: Improves prediction accuracy
- **Batch Processing**: Efficient processing of large test sets
- **CSV Output**: Results saved in submission format

## 📈 Output

The model generates:

1. **Trained Model**: `output/final_model.keras`
2. **Predictions**: `output/result.csv` with columns:
   - `id`: Image filename (without extension)
   - `type`: Predicted class (0-3)

### Sample Output Format

```csv
id,type
001,1
002,0
003,2
...
```

## 🔍 Performance Optimization

- **Mixed Precision Training**: Uses TensorFlow's automatic optimization
- **Parallel Data Loading**: Utilizes `tf.data.AUTOTUNE` for efficient I/O
- **Memory Management**: Prefetching and batch processing for optimal memory usage

## 🧪 Testing

The model is tested on 200 unlabeled images in the `data/test/` directory. Predictions are generated using Test Time Augmentation for improved accuracy.

## 📝 Dependencies

- **TensorFlow**: ≥2.8.0 (Deep learning framework)
- **NumPy**: ≥1.19.5 (Numerical computing)
- **Pandas**: ≥1.3.0 (Data manipulation)
- **Pillow**: ≥8.2.0 (Image processing)
- **scikit-learn**: ≥1.0.0 (Machine learning utilities)

## 🙏 Acknowledgments

- [OAI 2025 Contest](https://oai.hutech.edu.vn)
- Dataset contributors for providing mushroom images
- TensorFlow team for the excellent deep learning framework
- Vietnamese mushroom research community
- HUTECH University for organizing the OAI contest

---
