# CNN_CancerDetector
Histopathologic Cancer Detection using CNN Network

This project is based on the Kaggle Competition Histopathologic Cancer Detection. https://www.kaggle.com/c/histopathologic-cancer-detection

I have created a Convolutional Deep Neural Network using Tensorflow-Keras to identify metastatic cancer in small image patches taken from larger digital pathology scans. This model is able to identify Cancer cells with ~93% accuracy and ~97% AUC-ROC.

The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset. In the dataset, we are provided with 220K small sized images(96x96 pixels). A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

Notes 
1) Since the dataset is large and the CNN is deep, I had used the GPU-enabled tensorflow to get the machine to learn.
Set Up Instructions for GPU-enabled Tensorflow & Keras : https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1
2) I have only used the train-dataset from Kaggle to create this network, which I have divided accordingly to train/val/test datasets. 

Method

1) Data Preparation
   - Labels are read from the CSV file and encoded to binary
   - Train/Val/Test data labels are created in the proportion 90/10/10
   - A new directory structure is created to use Keras ImageDataGenerator
   
   /dataset/train/Class_NoTumor ;
   /dataset/train/Class_Tumor ;
   
   /dataset/val/Class_NoTumor ;
   /dataset/val/Class_Tumor ;
   
   /dataset/test/Class_NoTumor ;
   /dataset/test/Class_Tumor.
   
   - Images are copied into these folders based on the train/val/test dataset           
     
2) Building the CNN Model
    - Four sets of below struture is added to the model after its initialization (the last one w/o MaxPooling)
      - Convolution2D (Multiply 3 times) - Dropout - MaxPooling 
    - Add Flatten layer
    - Add Dense (128 output) - Dropout - Dense (1 output)
    - The model is compiled with Adam optimizer on accuracy metric
  
3) Data Generation
   - Keras ImageDatagenerators are used to create batches of tensor image data for the machine to learn/evaluate and test

4) Training the model
   - GPU session is initialized
   - Training starts, the best model is saved and the learning process is logged

5) Model Assessment 
    - Plot training & validation loss and Accuracy.
    - Evaluate on the validation dataset to understand the Accuracy and loss
    
6) Performance Measurement
    - Predict on the test set
    - Metrics used to evaluate the model
        - AUC - ROC
            AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. 
            ROC curve tells how much model is capable of distinguishing between classes. 
            Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. 
            By analogy, Higher the AUC, better the model is at distinguishing between patients with Tumor and No_Tumor.
         - The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.
     - Plot Confusion Matrix
     - Classification report is generated to understand overall model performance
     
 Reference kernels
 
 https://www.kaggle.com/vbookshelf/cnn-how-to-use-160-000-images-without-crashing/data
 https://www.kaggle.com/soumya044/histopathologic-cancer-detection
    
