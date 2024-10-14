# Knowledge-Management-Lab
## Lab1
In the attached folder (Task), you can find 3 folders (Train, Validation, and Test) that contain subject folders. Each of the subject folders includes 1 folder named Face, which contains cropped faces from the subject video, each 25 consecutive frames representing 1 second, and a file (gt_SpO2.csv) that contains oxygen saturation values for each second of the video.

Take the following steps:
1. Pre-process the dataset.
2. Extract the photoplethysmogram signal by spatial averaging of the pixels over time (you can extract the signal from one channel, or you can take the average signal from the 3 channels.)
3. Break down the signals you got into non intersecting 5 seconds windows (125 frames), and take the ground truth of each window as the mean of the consecutive 5 seconds
4. Build a CNN model that estimates the oxygen saturation of the input which is a piece of the PPG signal with length of 5 seconds. The model should include 1-D convolutional layer, max pooling layer, average pooling layer, global average pooling layer, flatten layer, dense layer

Requirements:
The loss function is MSE
The metric is MAE

## Lab2
The files that you need to deal with are:
1. VPTD_Dataset.csv :: contains the ground truth for personality traits and sales estimation
2. smiles.csv :: contains the smiles detected in each video from the dataset
3. head_movements.json :: contains the info about the head movments (roll, pitch, yaw) for each frame

Take the following steps:
1. Read the files and check their structure for further processing. You might use pandas and json to help you read the files
2. Show the correlation between smiles and Extraversion. You might use a confusion matrix with correlation tests like (Pearson and Spearman). Try also finding any other relationship between smiles and (sales estimation and personality traits)
3. Show the correlation between head movements and personality traits. You should try first finding features like (mean, std, min, max, ...) and check any correlation. You might use a confusion matrix. Find important events and then try to prove the correlation. Try also finding any other relationship (use your way for analyzing)
4. Prove the correlation you found between non-verbal cues (smiles and head movements) with personality traits, by training some models to predict the correlated traits depending on the extracted features. You might train some models like (Random Forest, XGBoost, MLP, ...). Train you models on 70% of the data and test on the rest. Show your results and also compersion between the models.
