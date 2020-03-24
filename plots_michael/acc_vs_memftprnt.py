import numpy as np
import matplotlib.pyplot as plt
import os
import pdb


#Names of directoriesto access
directory = f'../../results/large_scale_test/'

fs = 160
T_vec = [1,2,3]
ds_vec = [1,2,3]
nch_vec = [19,27,38,64]

KB_scale = 1000/1024

memftprnt = np.zeros(36)
acc = np.zeros(36)
cnt = 0
num_classes = 4

y_min = 50 
y_max = 66

ram_M4 = 90
ram_M7 = 290

ram_base = (480*64+480*64*40)*4/1000*KB_scale
acc_base = 58.59
print(ram_base)


markers = {1:"s", 2:"P", 3:"o"}
configurations = []



plt.rcParams.update({'font.size': 14})
# ACCURACY
plt.figure()
# plot baseline 
plt.scatter(ram_base,acc_base,color = 'C6',marker = 'D',s = 75,label='Dose et al. [18]')
cnt_last = 0
for T in  T_vec:

    for n_ds in ds_vec:
        for n_ch in nch_vec: 
            #Names of files to access
            config_name = f'ds{n_ds}_nch{n_ch}_T{T}'
            configurations.append(config_name)
            validation_acc_str  = f'{directory}stats/valid_accu_class_{num_classes}_{config_name}_avg.csv'


            if os.path.isfile(validation_acc_str):

                # compute memory footprint
                Ns = np.ceil(T*fs/n_ds) # number of input samples 
                poolLength = np.ceil(8/n_ds) 
                #Np = Ns*n_ch+Ns*n_ch*8+ np.floor(Ns/poolLength)*16 + np.floor(np.floor(Ns/poolLength)/8)*16+4 
                Np = Ns*n_ch*9 # np.floor(Ns/poolLength)*16 + np.floor(np.floor(Ns/poolLength)/8)*16+4
                memftprnt[cnt] = Np*4/1e3*KB_scale # memory footprint in kiB
                #pdb.set_trace()
                #Read Data
                validation_acc_df   = np.loadtxt(validation_acc_str)
                
                acc[cnt] = validation_acc_df[-1]


                if config_name =="ds3_nch38_T1": 
                    plt.scatter(memftprnt[cnt],acc[cnt]*100,color = 'blue',s=150,linewidth=1.5,facecolors='none')
                elif config_name =="ds3_nch38_T2": 
                    plt.scatter(memftprnt[cnt],acc[cnt]*100,color = 'tab:cyan',s=150,linewidth=1.5,facecolors='none')


                cnt +=1




    plt.plot(memftprnt[cnt_last:cnt],acc[cnt_last:cnt]*100,marker=markers[T],linewidth=0,
        color = f'C{T}',label=f'T={T}s') #illstyle='none'
    plt.xlabel('RAM requirements (log-scale) [KB]')
    plt.ylabel('Accuracy [%]')
    plt.ylim([y_min,y_max])

    cnt_last = cnt
    
    plt.grid()

index = np.argsort(memftprnt)

for idx in index:
    print("{:}\t{:}\t{:.2f}".format(configurations[idx],memftprnt[idx],acc[idx]*100))


#plt.savefig(f'lr_10e_{alpha}_split{split_str}_acc.pdf')

plt.ylim([y_min,y_max])
plt.xlim([30,10000])

# plot limits 
plt.plot([ram_M4,ram_M4],[y_min,y_max],color = 'blue',linestyle=':',label='RAM limit M4')
plt.plot([ram_M7,ram_M7],[y_min,y_max],color = 'tab:cyan',linestyle='--',label='RAM limit M7')



plt.legend()
plt.xscale('log')

plt.savefig(f'./acc_vs_memftprnt.pdf')

plt.show()