# Biometric-Systems-project-Federici-Fornetti
Gender and age recognition systems. Project for "Biometric systems" course at "Sapienza Università di Roma". 

- Group members: Federici Tommaso, Lucia Fornetti

### Age and Gender Recognition 
We developed two architectures for the Age and Gender recognition tasks.  

#### Architecture 1
The first architecture is a multitask model obtained through the fine-tuning technique, using Google's FaceNet as the base model. The UTKFace dataset was used for retraining.  

The repository includes training and testing files for this architecture, the 'best_model.pth' file containing the weights obtained during training, and a small demo file to test the model on images in the 'demo_images' folder.  

#### Architecture 2  
The second architecture combines two single-task models, both based on the CaffeNet model.  
This architecture is an adaptation of an existing project ([https://github.com/MohammedNayeem9/Age-and-Gender-Detection-using-OpenCV-in-Python](https://github.com/MohammedNayeem9/Age-and-Gender-Detection-using-OpenCV-in-Python)), which uses CaffeNet networks for age and gender recognition. The repository includes a testing file and a folder containing the files required to load the models.  

#### Testing  
The UTKFace dataset was split into training, validation, and testing sets with percentages of 70%, 10%, and 20% respectively. Both models were tested on the same test set using different metrics for each task. The results can be found in the project report.  

##### Credits:
