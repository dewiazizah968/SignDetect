# SignDetect
SignDetect is a digital image processing project focused on sign language recognition using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers. The system leverages simple preprocessing and data augmentation techniques to achieve high accuracy in gesture classification.

## Project Overview
The sign language recognition system is successfully built using straightforward image augmentation and preprocessing methods, achieving high accuracy rates of:
  
  KNN: 99.73%
  
  SVM: 99.53%

An interactive GUI demonstrates strong performance on test data; however, there is room for improving generalization on real-world images. This approach provides an effective and efficient solution for automatic gesture recognition.

## Live Demo
Try the live web application here:
[[Click here to open SignDetect app](https://signdetect-dip-project.streamlit.app/)]

## Running Locally
To run the application locally, simply execute:
```python
streamlit run main.py
```

## Training From Scratch
If you want to train the models from scratch, follow these steps:
1. Download the dataset here: [[dataset link](https://drive.google.com/drive/folders/1qN-6N_GOYRJ3a_hpC3CBdWrJvusxj-Pr?usp=sharing)]
2. Run the Python scripts in the following order:

    1_Augmentasi.py
    
    2_Pre-processing.py
    
    3_KNN.py
    
    4_Prediksi KNN.py
    
    5_SVM.py
    
    6_Prediksi SVM.py
    
    main.py

## Project Structure
1. 1_Augmentasi.py: Data augmentation to increase dataset variability
2. 2_Pre-processing.py: Image preprocessing (resize, grayscale, thresholding, etc.)
3. 3_KNN.py: Training the KNN classifier
4. 4_Prediksi KNN.py: Predicting sign language gestures using KNN
5. 5_SVM.py: Training the SVM classifier
6. 6_Prediksi SVM.py: Predicting sign language gestures using SVM
7. main.py: Streamlit app to run the interactive GUI
