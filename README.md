## 🫁 Lung Cancer Detection Using Hybrid Model

### 📌 Overview

Lung cancer is one of the most lethal forms of cancer globally, largely due to late-stage diagnosis and the subtle nature of early symptoms. To address this challenge, this project presents a **hybrid model** that combines the **deep learning capabilities of Faster R-CNN (FRCNN)** with the **predictive power of traditional machine learning algorithms** to detect and classify lung nodules from CT scan images.

This hybrid system is designed to support radiologists and medical professionals by providing accurate, automated assessments of lung nodules, enabling **early detection** and **timely treatment planning**.

---

### ⚙️ Key Components

#### 🧠 1. Deep Learning – **Faster R-CNN (FRCNN)**

* We use FRCNN, a region-based convolutional neural network, to detect and localize lung nodules in CT scan images.
* FRCNN efficiently identifies regions of interest (ROIs) that may indicate the presence of cancerous growth.
* It leverages convolutional layers for feature extraction and proposes bounding boxes around suspected nodules.

#### 📊 2. Machine Learning Classifiers

After extracting nodule features using FRCNN, the data is passed to multiple machine learning models for classification:

* **Support Vector Machine (SVM)**: For high-margin separation of benign and malignant nodules.
* **Random Forest (RF)**: Ensemble-based model for handling complex patterns and reducing overfitting.
* **Logistic Regression**: Probabilistic classification used as a baseline model.
* **Linear Regression**: Evaluated for performance but more applicable in regression contexts.

---

### 🔍 Workflow

1. **Data Preprocessing**

   * CT scan images are cleaned, resized, and normalized.
   * Ground-truth annotations are used to train the FRCNN model.

2. **Nodule Detection (FRCNN)**

   * The model scans each image, proposing regions that may contain nodules.
   * Detected regions are cropped and features extracted for classification.

3. **Feature Classification**

   * ML models are trained and tested on extracted features.
   * Each model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

4. **Model Comparison and Selection**

   * Results are analyzed to determine the most effective classifier or ensemble method.

---

### 📈 Objective

The primary objective is to build a **highly accurate, robust, and interpretable** system for lung cancer detection that leverages both:

* **Deep learning** for precise image-based localization
* **Machine learning** for efficient classification of the detected nodules

This dual approach ensures better generalization, improved accuracy, and reduced false positives/negatives compared to standalone models.

---

### 🧪 Tools & Technologies

* Python 🐍
* TensorFlow / PyTorch (for FRCNN)
* Scikit-learn (for ML models: SVM, RF, Logistic/Linear Regression)
* OpenCV, NumPy, Pandas
* Matplotlib/Seaborn (for visualizations)

---

### 📊 Performance Evaluation

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Cross-validation is also employed to validate model robustness.

---

### 📁 Dataset

(Include your dataset source here — e.g., LIDC-IDRI or mention if it’s a private dataset)

* CT scan images with labeled nodules
* Annotated ground-truth bounding boxes
* Preprocessed for noise removal and normalization

---

### ✅ Outcomes

* Improved diagnostic accuracy for early-stage lung cancer
* Reduced human error in radiology reports
* Faster analysis and decision-making
* Scalable solution for clinical deployment

---

### 🤝 Contribution

We welcome contributions, feedback, and suggestions! Please feel free to fork the repo, open issues, or submit pull requests.

---

### 📄 License

Apache 2.0
