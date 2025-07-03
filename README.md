# Image Classification using EfficientNetB5

This project implements a deep learning pipeline for multi-class image classification using transfer learning. The model leverages EfficientNetB5 as the feature extractor, followed by custom dense layers for final classification.

## Dataset

- Total Images: ~4,217
- Classes: 4
- Dataset split:
  - Training: 3,373 images
  - Validation: 422 images
  - Testing: 422 images

The dataset is structured in subdirectories per class (standard Keras format).

## Model Architecture

The classification model is built using EfficientNetB5 as the base model (pretrained on ImageNet, include_top=False). The architecture includes:

Input → EfficientNetB5 → BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Dense(4, softmax)

### Model Summary

Total Parameters: 29,079,675  
Trainable: 28,902,836  
Non-Trainable: 176,839  

## Dependencies

- Python 3.x
- TensorFlow 2.9.1
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- OpenCV (optional for data preprocessing)

Install all dependencies:
```bash
pip install tensorflow==2.9.1 numpy pandas matplotlib seaborn scikit-learn opencv-python
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/IrisScope.git
   cd your-repo-name
   ```

2. Run the notebook:
   ```bash
   jupyter notebook notebookc4481e3d4f.ipynb
   ```

3. Ensure your dataset follows this structure:
   ```
   dataset/
     ├── Class1/
     ├── Class2/
     ├── Class3/
     └── Class4/
   ```

## Results

- EfficientNetB5 significantly boosts performance through pretrained knowledge.
- The notebook includes training curves, accuracy/loss plots, and a confusion matrix.

## Author

Akansha Mulchandani  
BTech CSE, IIIT Pune  
Deep learning researcher | TensorFlow enthusiast
