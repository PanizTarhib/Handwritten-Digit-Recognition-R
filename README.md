# Handwritten Digit Recognition in R

A machine learning project for recognizing handwritten digits (0-9) using R.  
Models: KNN, MLP (nnet + PCA), Multi-Layer Neural Network (Keras), CNN (Keras).

## Results
- KNN: ~41.67% accuracy  
- MLP: ~51.67% accuracy  
- Multi-Layer NN: ~55% accuracy  
- CNN: ~76.67% accuracy (best)

## Project Files
- digit_recognition.R → All code in one file  
- report/number_recognition.pdf → Full English report  
- data/README.md → How to prepare your dataset

## Important: Dataset
Images are not uploaded (large size). Please follow data/README.md to create your own images.

## How to Run
1. Prepare images folder (see data/README.md).  
2. Update paths in the R script.  
3. Install packages:  
   ```r
   install.packages(c("imager", "caret", "tidymodels", "kknn", "nnet", "keras", "tensorflow", "reticulate"))

Run the script in R or RStudio.
Report (English) is in report/ folder.
Suggestions: Add data augmentation, thresholding, hyperparameter tuning.
Made for learning purposes.
