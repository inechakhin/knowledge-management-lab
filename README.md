# Knowledge-Management-Lab
## Lab1
In the attached folder (Task), you can find 3 folders (Train, Validation, and Test) that contain subject folders. Each of the subject folders includes 1 folder named Face, which contains cropped faces from the subject video, each 25 consecutive frames representing 1 second, and a file (gt_SpO2.csv) that contains oxygen saturation values for each second of the video.

Take the following steps:
1- Pre-process the dataset.
2- Extract the photoplethysmogram signal by spatial averaging of the pixels over time (you can extract the signal from one channel, or you can take the average signal from the 3 channels.)
3- Break down the signals you got into non intersecting 5 seconds windows (125 frames), and take the ground truth of each window as the mean of the consecutive 5 seconds
4- Build a CNN model that estimates the oxygen saturation of the input which is a piece of the PPG signal with length of 5 seconds. The model should include 1-D convolutional layer, max pooling layer, average pooling layer, global average pooling layer, flatten layer, dense layer

Requirements:
The loss function is MSE
The metric is MAE
