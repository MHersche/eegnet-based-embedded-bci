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
There are two different approaches:
1) Intra-subject training and testing
2) Inter-subject training and testing

## Hyperparameter Tuning
Data: 4/5 of 105 subjects' data is used for 4-Fold Cross-Validation. 
This means that 21 subjects' data is left untouched, while 
63 subjects' data is used as training data and 21 subjects' data is
used as validation data for each of the 4 folds.
4-class data is used exclusively.

Hyperparameter search was done on: 
* Dropout Rate: 0.1, **0.2**, 0.3, 0.4
* Learning Rate: 10^-1, 10^-2, 10^-3, **10^-4**, 10^-5
* Number of Epochs for different Learning Rates
* Constant and Sparse Learning Rate Schedulers for the [Adam Optimizer](https://arxiv.org/abs/1412.6980) 

Results:
* Dropout Rate=0.2
* For a single Learning Rate: LR=10^-4
* For Learning Rate Scheduler: A Sparse LR Scheduler with:
    * LR=10^-3 for 20 epochs
    * LR=10^-4 for 30 epochs
    * LR=10^-5 for 50 epochs
With the help of the sparse Learning Rate Scheduler, we avoid the fluctuations and overfitting of LR=10^-3 ![Learning Rate 1e-3](/hp_tuning/global_trainer_hp_lr/plots/accu_lr_1e--3_avg.pdf)

## Global Models
Global classifier models were trained for 2-, 3-, and 4-class data:
* 5-Fold Cross-Validation Accuracy
* 5-Fold Cross-Validation Loss
* Confusion Matrices

## Subject Specific
Global classifier models were further retrained on individual subject data
for 2-, 3-, and 4-class data
* Per Subject 4-Fold Cross-Validation Accuracy/Loss
* Averaged among source global model 4-Fold CV Accuracy/Loss
* Averaged among all subjects 4-Fold CV Accuracy/Loss
