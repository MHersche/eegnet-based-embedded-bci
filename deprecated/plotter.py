import pandas as pd
import matplotlib.pyplot as plt

for alpha in range(1,5):
    #Names of directoriesto access
    directory = f'lr_10e{alpha}/'
    for split_str in range(0,5):
        print(f'Learning Rate: {10**(-alpha)}, Split: {split_str}')
        #Names of files to access
        training_loss_str   = f'train_lr_loss[{10**(-alpha)}]_dr[0.2]_split[{split_str}].csv'
        validation_loss_str = f'valid_lr_loss[{10**(-alpha)}]_dr[0.2]_split[{split_str}].csv'
        training_acc_str    = f'train_lr[{10**(-alpha)}]_dr[0.2]_split[{split_str}].csv'
        validation_acc_str  = f'valid_lr[{10**(-alpha)}]_dr[0.2]_split[{split_str}].csv'
        #Read Data
        training_loss_df    = pd.read_csv(f'{directory}{training_loss_str}')
        validation_loss_df  = pd.read_csv(f'{directory}{validation_loss_str}')
        training_acc_df     = pd.read_csv(f'{directory}{training_acc_str}')
        validation_acc_df   = pd.read_csv(f'{directory}{validation_acc_str}')
        ###########Plot, label, save###########
        # LOSS
        plt.plot(training_loss_df,label='Training')
        plt.plot(validation_loss_df,label='Validation')
        plt.title(f'Loss: LR={10**(-alpha)}, Split={split_str}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'lr_10e_{alpha}_split{split_str}_loss.pdf')
        plt.show()
        # ACCURACY
        plt.plot(training_acc_df,label='Training')
        plt.plot(validation_acc_df,label='Validation')
        plt.title(f'Accuracy: LR={10**(-alpha)}, Split={split_str}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'lr_10e_{alpha}_split{split_str}_acc.pdf')
        plt.show()