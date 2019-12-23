# Results

---

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

