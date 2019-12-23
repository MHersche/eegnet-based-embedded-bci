# Results

---

## EEGNet on BCI Competition IV
We have trained and tested EEGNet on BCI Competition 2b dataset.
This dataset includes Motor Imagery (MI) from 9 subjects.
Below are the acquired results for each individual subject with 500 epochs:


Classification accuracy of 1  : 0.779359

Classification accuracy of 2  : 0.579505

Classification accuracy of 3  : 0.930403

Classification accuracy of 4  : 0.618421

Classification accuracy of 5  : 0.615942

Classification accuracy of 6  : 0.502326

Classification accuracy of 7  : 0.815884

Classification accuracy of 8  : 0.793358

Classification accuracy of 9  : 0.738636

General accuracy: 0.708204


With 100 epochs we reached a general accuracy of 0.692

## EEGNet on Physionet Dataset
---
The dataset consists of 109 subjects Motor Imagery and Motor Movement Runs.
Each subject has 14 runs total with 2 of them just eye movement 6 of them motor movement and 6 of them motor imagery.
There are two different approaches:
1) Intra-subject training and testing.
2) Inter-subject training and testing.

## Intra-Subject Training
The data acquisition was different for some subjects, therefore we excluded 5 subjects and used 104 subjects for the purpose of the project. We used left hand, right hand, feet and rest data and did 4 class classification. The rest data is taken from the base run with eyes open. We have taken 3 seconds data after the cue.
Data is then seperated to 3 parts where each part includes 2 of the motor imagery runs each subject has.
Then the model hyperparameters are tuned. 
* Dropout Rate: **0.1**, 0.2, 0.3, 0.4
* Learning Rate: 10^-1, 10^-2, 10^-3, **10^-4**, 10^-5

The results are in "hyperParameterSearch"

Hyperparameter Tuning is done with 500 epochs using 5-fold Stratified CV so that there are equal number of classes in each fold.
After we chose the hyperparameters we have done testing with 1000 epochs as the model does not overfit.
We have reached an accuracy of 68,5% on our test set. The results are in "3_fold_testing".
Further training our model on data from individual subjects and testing with the data from individual subjects with the same parameters using 10 and 50 epochs. 10 epochs resulted in an average of 70,3% accuracy. The results are in "global_model_subject_specific".

We also validated the results by doing 3 fold cross validation using the combination of different runs as folds and reached an average accuracy of 68,1%. 

After that we have done 5 fold stratified cross validation to the whole dataset with 800 epochs and reached an accuracy of 69,8% averaged on all folds. The results are in "5_fold_cv_whole_dataset"

## Inter-Subject Training
---

## Hyperparameter Tuning
Data: 4/5 of 105 subjects' data is used for 4-Fold Cross-Validation. 
This means that 21 subjects' data is left untouched, while 
63 subjects' data is used as training data and 21 subjects' data is
used as validation data for each of the 4 folds.
4-class data is used exclusively.

Hyperparameter search was done on: 
* Dropout Rate: 0.1, 0.2, 0.3, 0.4
* Learning Rate: 10^-1, 10^-2, 10^-3, 10^-4, 10^-5
* Number of Epochs for different Learning Rates
* Constant and Sparse Learning Rate Schedulers for the [Adam Optimizer](https://arxiv.org/abs/1412.6980) 

Results:
* Dropout Rate=0.2
    * [DR=0.1](/results/hp_tuning/global_trainer_hp_dr/stats/valid_accu_dr_01_avg.csv): 62.54%
    * [DR=0.2](/results/hp_tuning/global_trainer_hp_dr/stats/valid_accu_dr_02_avg.csv): 63.75%
    * [DR=0.3](/results/hp_tuning/global_trainer_hp_dr/stats/valid_accu_dr_03_avg.csv): 62.54%
    * [DR=0.4](/results/hp_tuning/global_trainer_hp_dr/stats/valid_accu_dr_04_avg.csv): 61.41%
* For a single Learning Rate: LR=10^-4
    * [LR=10^-1](/results/hp_tuning/global_trainer_hp_lr/stats/valid_accu_lr_1e--1_avg.csv): 41.75%
    * [LR=10^-2](/results/hp_tuning/global_trainer_hp_lr/stats/valid_accu_lr_1e--2_avg.csv): 56.90%
    * [LR=10^-3](/results/hp_tuning/global_trainer_hp_lr/stats/valid_accu_lr_1e--3_avg.csv): 57.85%
    * [LR=10^-4](/results/hp_tuning/global_trainer_hp_lr/stats/valid_accu_lr_1e--4_avg.csv): 62.54%
    * [LR=10^-5](/results/hp_tuning/global_trainer_hp_lr/stats/valid_accu_lr_1e--5_avg.csv): 61.39%
* For Learning Rate Scheduler: A Sparse LR Scheduler with:
    * LR=10^-3 for 20 epochs
    * LR=10^-4 for 30 epochs
    * LR=10^-5 for 50 epochs

With the help of the sparse Learning Rate Scheduler, we avoid the fluctuations and overfitting of [LR=10^-3](/results/hp_tuning/global_trainer_hp_lr/plots/accu_lr_1e--3_avg.pdf), and surpass the accuracies of [LR=10^-4](/results/hp_tuning/global_trainer_hp_lr/plots/accu_lr_1e--4_avg.pdf), while reducing the total number of epochs to just 100.

## Global Models
Data: The 105 subjects are split into 5 Folds. For each fold, 4/5 of the subjects' data is used as training data. Validation is done using the rest 1/5 of the subjects' data for 5-Fold Global Classsifier accuracy.
Global classifier models were trained for 2-, 3-, and 4-class data.

Results include:
* Individual classifier Accuracy/Loss
* 5-Fold Cross-Validation Accuracy/Loss
* Confusion Matrices

## Subject Specific
Global classifier models were further retrained on individual subject data
using 2-, 3-, and 4-class data.

Results include:
* Per Subject 4-Fold Cross-Validation Accuracy/Loss
* Averaged among source global model 4-Fold CV Accuracy/Loss
* Averaged among all subjects 4-Fold CV Accuracy/Loss

## MCU
After creating our models we wanted to implement our model in a STM32 M4 microcontroller. Therefore whe have used STMCubeAI tool to pass our model. Our model was too big to fit in the RAM of the device. It required 1 MB ram and we had 128 KB. Therefore made the input of our model smaller. We have tested several different input shapes such as downsampling the frequency, decreasing the channel size and decreasing the duration of the data. At the end we have chosen to decrease the channel size to 19. We have selected the electrodes in 10-20 system from international standard exluding A1 and A2. We have downsampled the input by a factor of 5 and kept the duration of 3 seconds. 
Validation on the microcontroller and the C-model didn't result in any accuracy decrease.

We have tested our implementation on microcontroller with the first fold of the inter-subject 5 fold cross validation. Decreasing the size of the input resulted in an accuracy decrease from 67% to 62,6%.
The results are in the folder "mcu". Output of the validation from STMCubeAI can be found in mcu/outputs. The channels we have used can be found in mcu/channels.

