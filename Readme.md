# Skin Cancer Detection using Hybrid ANN-CNN Model

## Overview
This project aims to predict the presence of skin cancer using a hybrid deep learning model that integrates both tabular data and image data. The dataset used is the ISIC dataset, which contains skin lesion images along with associated metadata. The model achieves an accuracy of **88%** on the test set, demonstrating its effectiveness in combining multiple data modalities for improved prediction.

The hybrid model leverages:
1. **Tabular Data**: Metadata such as patient age, sex, lesion location, and other clinical features.
2. **Image Data**: RGB images of skin lesions resized to 128x128 pixels.

By combining these two data types, the model captures both visual patterns in the images and contextual information from the metadata, leading to robust predictions.

---

## Integration of Tabular and Image Data
The model integrates tabular and image data using a two-branch architecture:
1. **Image Data Pipeline**:
   - A pre-trained **EfficientNetB0** model is used as a feature extractor for the image data.
   - The model is fine-tuned to adapt to the specific characteristics of skin lesion images.
   - The output of EfficientNet is flattened and passed through a dense layer for further processing.

2. **Tabular Data Pipeline**:
   - The tabular data is preprocessed using standardization (scaling to zero mean and unit variance).
   - A simple Artificial Neural Network (ANN) with dense layers processes the tabular data.

3. **Combined Model**:
   - The outputs of the image and tabular pipelines are concatenated.
   - The combined features are passed through additional dense layers to produce the final prediction.

---

## Architecture of the Model
The hybrid model consists of the following components:

### 1. **Image Branch**:
   - **Input**: 128x128x3 RGB images.
   - **Backbone**: Pre-trained EfficientNetB0 (frozen during initial training).
   - **Layers**:
     - EfficientNetB0 (feature extraction).
     - Flatten layer to convert 3D features to 1D.
     - Dense layer (128 units, ReLU activation).
     - Dropout layer (0.5 dropout rate for regularization).

### 2. **Tabular Branch**:
   - **Input**: Tabular data (12 features after preprocessing).
   - **Layers**:
     - Dense layer (64 units, ReLU activation).
     - Dropout layer (0.5 dropout rate for regularization).

### 3. **Combined Branch**:
   - **Concatenation**: The outputs of the image and tabular branches are concatenated.
   - **Layers**:
     - Dense layer (64 units, ReLU activation).
     - Dropout layer (0.5 dropout rate).
     - Output layer (1 unit, sigmoid activation for binary classification).

### 4. **Training**:
   - **Optimizer**: Adam optimizer with a learning rate of 0.001.
   - **Loss Function**: Binary cross-entropy (since it's a binary classification task).
   - **Metrics**: Accuracy.

---

## Results
The model achieves **88% accuracy** on the ISIC dataset, demonstrating its ability to effectively combine tabular and image data for skin cancer detection. The integration of metadata with image data provides additional context, improving the model's predictive performance.

---

## How to Use
1. **Install Dependencies**:
   Ensure you have the required libraries installed:
   ```bash
   pip install tensorflow pandas numpy scikit-learn
   ```

2. **Run the Model**:
   - Clone the repository.
   - Load the dataset and preprocess it as described in the code.
   - Train the model using the provided script.

3. **Evaluate the Model**:
   - Use the test set to evaluate the model's performance.
   - Visualize predictions and analyze results.

---

## Future Work
- Experiment with other pre-trained models (e.g., ResNet, Inception) for image feature extraction.
- Perform hyperparameter tuning to further improve accuracy.
- Explore additional data augmentation techniques for images.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The ISIC dataset for providing high-quality skin lesion images and metadata.
- TensorFlow and Keras for providing the tools to build and train deep learning models.