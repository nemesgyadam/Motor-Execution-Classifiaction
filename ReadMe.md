# Task

In this project we classified different hand movements with machine learning tools.\
The subjects EEG signals has been recorded during the experiment and this signals was processed with  Deep Learning (convolutional neural network)

&nbsp;
# Paradigm
The object of the project was to classify the different tasks\
The tasks can be changed in the config file.\
Default classes are:

    * Rest    (reference state)
    * Left    (move the left hand)
    * Right   (move the right hand)  

&nbsp;
# Devices:
The repository supports the following devices:

## **gTec Unicorn**
[Official Site](https://www.unicorn-bi.com/)

### Unicorn setup:
1. ) Install Unicorn **Suite Hybrid Black** (ask maintener for installer)
2. ) Activate the **Unicorn Python Hybrid Black** in the licences tab of Suite (ask maintener for licence key)
3. ) Add the following Environment Variable to System Variables:
    - Name: PYTHONPATH
    - Value: C:/Users/[User]/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Python/Lib
4. ) Import UnicornPy package in your python file


## **Minrove Arc**
[Official Site](https://mindrove.com/)

[Github Repo](https://github.com/MindRove/SDK_Public)

[User Manual](https://mindrove.com/wp-content/uploads/2021/11/UserManual_v2_0_0.pdf)

### Arc setup:
1. ) Download SDK: [Mindrove SDK](https://github.com/MindRove/SDK_Public)
2. ) Install python package from [SDK]/win64/python using:
    - pip install -e [path]
    - [path] => [directory where 'setup.py' can be found]
3. ) (Optional temporal fix) If you get error related to NDArray, remove all type definitions 
4. ) Connect Mindrove on WiFi (default password: "#mindrove")
&nbsp;\
<br>
<br>
## Details of recording:



The expriment contained separated sessions from the same subject, recorded in different times.




Each task was recorded with the following timeline:
    - 1 second stand by time 
    - The instructions appeared
    - 2 second of recording period
    
&nbsp;
# Dataset
The raw EEG signals has been stored, all filtering and preprocessing applied before the train. Each session stored in separated folder.
The 3 class has been stored in separated npy files. Each file contains 20 samples. So the files has the following dimension:


(20,6,2*sample_rate) --> N_sample, N_channel, Time_points

&nbsp;
# Training
The Train.ipynb contains an implementaion of a  "traditional" feed forward training method. 
