# Dataset Instructions

Images are NOT included in this repository due to size and upload limits.

To run the project, create your own dataset:

1. Handwrite digits 0-9 on paper (10-20 samples per digit recommended).  
2. Take clear photos with your mobile phone camera.  
3. Save images in a folder, for example: D:/nums  
4. Name files like this: 0_1.JPG, 0_2.JPG, ..., 9_20.JPG  
   (The number before '_' is the label/digit)  
5. For testing, create a separate folder like D:/test with sample files (e.g., 8.jpg).

Then update the path in digit_recognition.R:
image_folder <- "D://nums"   # Change to your actual path
