## CatDogClassifier-GUI
Overview
CatDogClassifier-GUI is a machine learning project that implements a convolutional neural network (CNN) to classify images as either cats or dogs, leveraging transfer learning with the pre-trained VGG16 model from TensorFlow/Keras. The project includes a user-friendly Tkinter-based graphicalFoundational Knowledge: The project uses the pre-trained VGG16 model from TensorFlow/Keras to classify images of cats and dogs with approximately 90% accuracy on a test dataset. Data augmentation techniques enhance the model's robustness and generalization. A Tkinter-based GUI enables real-time image classification, providing confidence scores and out-of-distribution detection for non-cat/dog images.
Features

Model Architecture: Utilizes VGG16 with frozen convolutional layers, followed by custom dense layers, dropout (0.5) for regularization, and a sigmoid output for binary classification.
Data Preprocessing: Applies data augmentation (rotation, zoom, flipping, etc.) to the training set to handle image variations, with rescaling for both training and test sets.
Training: Trained over 20 epochs using the Adam optimizer (learning rate 1e-4) and binary cross-entropy loss on a dataset of 10,000 images (8,000 training, 2,000 testing).
Evaluation: Includes confusion matrix visualization and accuracy metrics for both training and test sets to assess model performance.
GUI: A Tkinter-based interface allows users to upload images (JPG/PNG) and receive predictions with confidence scores, incorporating a threshold (0.85) for out-of-distribution detection (non-cat/dog images).
Dependencies: Requires TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn, Pillow, and Tkinter.

Installation

Download or copy the project files to your local machine.
Install the required dependencies using pip: tensorflow, numpy, matplotlib, seaborn, scikit-learn, pillow.
Ensure Python 3.8+ is installed, as Tkinter is included in the standard library.

Usage

Download and extract the dataset (see Dataset section below).
Place the training_set and test_set folders in the project directory as specified in the notebook (GUI Cat and Dogs.ipynb).
Open the Jupyter Notebook (GUI Cat and Dogs.ipynb) to train or evaluate the model, or to launch the GUI.
To use the GUI, run the final cell in the notebook to launch the Tkinter application, then upload an image to classify it as a cat or dog.

Dataset
The dataset used is the Cats vs. Dogs dataset, containing 10,000 images (8,000 for training, 2,000 for testing). Due to GitHub's storage limitations (100MB per file, ~1-5GB per repository), the dataset is not included in this repository. You can download it from Google Drive or the original source on Kaggle (Dogs vs. Cats). After downloading, extract the dataset and place the training_set and test_set folders in the project directory as referenced in the notebook.
Model Details

Input Size: 128x128 RGB images.
Training Setup: Batch size of 16, 20 epochs, with data augmentation to prevent overfitting.
Performance: Achieves ~90% validation accuracy, with confusion matrix analysis to evaluate true positives/negatives.
Out-of-Distribution Detection: Uses a 0.85 confidence threshold to identify images that are neither cats nor dogs.

Future Improvements

Fine-tune VGG16 layers for improved accuracy.
Extend to multi-class classification (e.g., other animals).
Deploy as a web application using Flask or Streamlit.
Optimize model size for faster inference and smaller storage requirements.

Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub to suggest improvements, report bugs, or add features.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with TensorFlow/Keras and Tkinter.
Dataset sourced from Kaggle's Dogs vs. Cats challenge.
Inspired by practical applications of transfer learning in computer vision.
