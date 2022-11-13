# Task

In this project we classified different hand movements with machine learning tools.
The subjects EEG signals has been recorded during the experiment and this signals was processed with  Deep Learning (convolutional neural network)

&nbsp;
# Paradigm
The object of the project was to classify the following 3 different tasks:

    * Rest    (reference state)
    * Left    (move the left hand)
    * Right   (move the right hand)  

&nbsp;
### Device:

The EEG signals was recorded with the gTec Unicorn:

[Official Site](https://www.unicorn-bi.com/)


&nbsp;
### Details of recording:


We recorded 6 channels with 250hz sampling rate.

The expriment contained separated sessions from the same subject, recorded in different times.




Each task was recorded with the following timeline:
    - 1 second stand by time 
    - The instructions appeared
    - 2 second of recording period
    
&nbsp;
# Dataset
The raw EEG signals has been stored, all filtering and preprocessing applied before the train. Each session stored in separated folder.
The 3 class has been stored in separated npy files. Each file contains 20 samples. So the files has the following dimension:

(20,6,1000) --> N_sample, N_channel, Time_points(2sec*500hz)

&nbsp;
# Training
The Train.ipynb contains an implementaion of a  "traditional" feed forward training method. 
