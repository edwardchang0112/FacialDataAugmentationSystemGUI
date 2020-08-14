# FacialDataAugmentationSystemGUI
In this project, there is a GUI interface that make you can do facial data augmentation based on pre-trained GAN model and weather information, in order to produce more qualified data for trainning ML/AI model. The following figure shows you how the GUI looks like.

![image](https://github.com/edwardchang0112/FacialDataAugmentationSystemGUI/blob/master/GUI_Fig.png)

# Preparation

1. The following link shows you the code to train a custom CGAN model, https://github.com/edwardchang0112/CRNNCGAN_OnFacialData. You can also use the pre-trained example model in this project to have a try.

2. The weather information used in this project is extracted from https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp. You can dowload the weather information directly and just remember to change the file name/format to fit this project

# Steps

Remember that the PATH of all of the following step need to be changed to fit your machine.

1. Load Trained Model: Press "Browse Trained Model" to load your pre-trained model, then press the button "Load Trained Model"
2. Load Weahter Info.: Press "Browse  Weahter File" to load the weather that you just dowload from https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp, then press the button "Load Weahter Info."
3. 

  (a) First Gen. File No.: Type the user No. you want, but make it in 3-number format (1:001, 10:, 010, ...)
  
  (b) Numbers of Gen. File: Type the number of users with data that you want to generate.
  
  (c) Prediction Days: The numbers of data that generated for 1 user. For now, please enter 15.
  
  (d) Start Date (YYYY-MM-DD)
  
  Then press "Start Generating New File"
  
4. Based on the existing user to extend the data, so don't delete or make any changes on the file "User_No_Code_Table.xlsx"

  (a) Choose one existing file: Choose one existing file generating from step 3.
  
  (b) Prediction Days: The numbers of data that generated for the user. For now, please enter 15.
  
  (c) Start Date (YYYY-MM-DD)

### All the project just provide you a basic structure, you need to make some changes to fit your applications.
