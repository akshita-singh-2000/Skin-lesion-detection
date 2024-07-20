# Skin Lesion Detection and Classification Using Deep Learning

This project focuses on the application of deep learning techniques for the detection and classification of skin lesions. By leveraging convolutional neural networks (CNNs) trained on medical images, we aim to develop automated systems capable of identifying various types of skin abnormalities. This can significantly improve early detection and treatment planning for conditions such as skin cancer.

## Objectives

- To address the limitations of small and homogeneous datasets in training neural networks for the automated diagnosis of pigmented skin lesions.
- To utilize the HAM10000 dataset, which includes diverse dermatoscopic images from different populations and acquisition modalities, to facilitate more robust machine learning models for accurate diagnosis.

## Dataset

The HAM10000 (Human Against Machine) dataset is a collection of dermatoscopic images used for skin lesion classification. It contains over 10,000 images of pigmented skin lesions from various sources and captured with different modalities.

### Key Points

- **Data Type:** Image Classification
- **Number of Images:** 10,015 
- **Content:** Dermoscopic images of pigmented skin lesions
- **Classes:** 7 diagnostic categories
  - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
  - Basal cell carcinoma (bcc)
  - Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
  - Dermatofibroma (df)
  - Melanoma (mel)
  - Melanocytic nevi (nv)
  - Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)
- **Source:** Multi-source, including different populations and acquisition methods
## Methods

### Data Preprocessing

1. **Exploratory Data Analysis (EDA):**
   - Addressed class imbalance using oversampling and undersampling techniques.
   - Ensured a balanced distribution of male and female data points to mitigate gender bias.
   - Analyzed age distribution and anatomical region frequencies to understand data characteristics.

2. **Image Integration:**
   - Integrated image data with cancer type labels.
   - Resized images to 32x32x3 arrays using OpenCV.
   - Applied Adaptive Synthetic Sampling (ADASYN) to balance the training data.

### Model Implementation

1. **CNN Model:**
   - Designed a sequential architecture with convolutional, max-pooling, batch normalization, and dropout layers.
   - Used Adam optimizer with a learning rate of 0.001.
   - Achieved approximately 72% accuracy on the test set.

2. **VGG16 Model:**
   - Utilized the pre-trained VGG16 model as a feature extractor with additional custom layers for classification.
   - Achieved 89.96% accuracy on the training set and 63.60% accuracy on the test set.

3. **ResNet50 Model:**
   - Employed the pre-trained ResNet50 model with custom top layers for classification.
   - Achieved 24.86% accuracy on the test set, indicating the need for further optimization.

### Evaluation

- **Performance Metrics:**
  - Accuracy, precision, recall, and F1-score for each class.
  - Confusion matrix heatmaps to visualize classification results.
  - Training and validation loss and accuracy plots to assess model learning and convergence.

## Results

- The traditional CNN model exhibited the best performance with a test accuracy of 72%.
- The VGG16 model showed promising results with high training accuracy but lower test accuracy, indicating overfitting.
- The ResNet50 model requires further optimization due to its lower test accuracy.

## Conclusion

The project demonstrates the potential of CNNs in medical image analysis for improving patient outcomes. Despite challenges such as class imbalance and dataset size, the models showed promising performance in classifying skin cancer types from medical images. Future work will focus on optimizing these models to enhance their accuracy and robustness for real-world applications.
