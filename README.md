# **Pneumonia Detection from Chest X-Rays**

## **Overview**
This project aims to classify chest X-ray images as pneumonia-positive or pneumonia-negative using deep learning techniques. The model is trained on the *"Chest X-Ray Images Dataset"* and includes both a custom CNN and fine-tuning of MobileNetV2 for transfer learning. It also evaluates performance using metrics like sensitivity, specificity, and ROC-AUC, with a comparison between augmented and non-augmented datasets.

---

## **Dataset**
The dataset used is the *"Chest X-Ray Images Dataset"* available on Kaggle. It consists of:
- **Training samples**: 5216
- **Validation samples**: 16
- **Testing samples**: 624

Classes:
- **NORMAL**: Healthy chest X-rays.
- **PNEUMONIA**: X-rays indicating pneumonia.

---

## **Features**
1. **Custom CNN Architecture**: Designed for binary classification with dropout regularization.
2. **Transfer Learning**: Fine-tuning MobileNetV2 pre-trained on ImageNet.
3. **Data Augmentation**: Includes random rotations, shifts, zooms, and histogram equalization to improve generalization.
4. **Performance Metrics**: Sensitivity, specificity, and ROC-AUC score used for evaluation.
5. **Comparison**: Results analyzed for augmented vs. non-augmented datasets.

---

## **Installation and Setup**

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. Install the required libraries:
   ```bash
   pip install tensorflow matplotlib scikit-learn opencv-python kagglehub
   ```

3. Download the dataset:
   The dataset will be automatically downloaded using the KaggleHub library:
   ```python
   path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
   ```

---

## **Usage**

### **Step 1: Preprocess the Data**
- Normalize images to the range [0, 1].
- Perform data augmentation (e.g., rotations, shifts, and histogram equalization).
- Define training, validation, and test data generators.

### **Step 2: Model Training**
- Train the custom CNN architecture or fine-tune MobileNetV2.
- Use binary cross-entropy loss for the custom CNN and categorical cross-entropy for MobileNetV2.
- Save the trained model:
   ```python
   cnn_model.save('cnn_pneumonia_model.h5')
   ```

### **Step 3: Evaluation**
- Evaluate the model on the test dataset using:
  - Sensitivity
  - Specificity
  - ROC-AUC score
- Compare results on augmented vs. non-augmented datasets.

### **Step 4: Visualize Results**
- Plot the ROC curve to analyze performance.
- Visualize augmented data samples to inspect the effects of preprocessing.

---

## **Key Results**

### **Augmented Dataset**
- **Sensitivity**: 92% (high pneumonia detection).
- **Specificity**: 7% (poor normal case detection).
- **ROC-AUC**: 0.50 (random-level performance).

### **Non-Augmented Dataset**
- **Sensitivity**: 88% (slightly lower pneumonia detection).
- **Specificity**: 14% (slightly better normal case detection).
- **ROC-AUC**: 0.49 (poor overall discrimination).

---

## **Challenges**
- **Class Imbalance**: Models were biased towards detecting pneumonia cases, leading to low specificity.
- **Small Validation Set**: Only 16 samples were available for validation.
- **Training Interruptions**: Misconfigurations in data generators caused stability issues.
- **Augmentation Variability**: Noise introduced by excessive augmentation reduced specificity.

---

## **Future Improvements**
1. Address class imbalance using oversampling or weighted loss functions.
2. Increase validation set size for better evaluation.
3. Use additional pre-trained models like ResNet or EfficientNet for improved feature extraction.
4. Fine-tune augmentation techniques to prevent overfitting.

---

## **Conclusion**
This project demonstrates the potential of deep learning in medical imaging for pneumonia detection. While achieving high sensitivity, further optimization is needed to enhance specificity and overall performance for real-world applications.

---
