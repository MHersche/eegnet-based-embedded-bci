#!/usr/bin/env python3

''' 
Hyperdimensional (HD) classifier packager 
'''

import numpy as np

# plots 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import csv
import scipy.stats as stat
import os





def write_to_csv(file,data):


	# compute wilcoxon statistics p-value 
	first_column = ['2 classes', '3 classes', '4 classes']

	
	################################################ Write Shallow and EEGNet results 
	#Assuming res is a flat list
	
	f = open(file+'table_reduction.txt',"w")

	f.write('\\begin{tabular}{lcccccccc} \n')
	f.write('\\cmidrule(r){1-9} \n')
	f.write('& Standard & \\multicolumn{2}{c}{Downsampling} & \\multicolumn{3}{c}{Channels} & \\multicolumn{2}{c}{Time window} \\\\ \n')
	f.write("\\cmidrule(r){2-2} \\cmidrule(r){3-4} \\cmidrule(r){5-7} \\cmidrule(r){8-9} \n")
	f.write('&  & $ds$=2 & $ds$=3 & $N_{ch}$=38 & $N_{ch}$=19 & $N_{ch}$=8 & $T$=2\\,s & $T$=1\\,s \\\\ \n')
	f.write("\\cmidrule(r){3-3} \\cmidrule(r){4-4} \\cmidrule(r){5-5} \\cmidrule(r){6-6} \\cmidrule(r){7-7} \\cmidrule(r){8-8} \\cmidrule(r){9-9} \n")

	for row in range(data.shape[0]): 
		f.write('{}'.format(first_column[row]))

		for column in range(8):			
			f.write('&\t{:.2f}'.format(data[row,column]))

		f.write('\\\\ \n')

	f.write('\\cmidrule(r){1-9}  \n')

	f.write("\\end{tabular}")
	f.close()








def main():


	# load_path = '/home/herschmi/Documents/Projects/03_HD_superpos/DATE2020/results/'

	# with open(load_path + 'results_shallowEEGNet.csv', newline='') as csvfile:
	# 	data = list(csv.reader(csvfile))
	data = np.zeros((3,8))

	path_4classtest = '../../results/large_scale_test/'
	path_channel_test = '../../results/channel_test/'
	path_ds_test = '../../results/dstest/'
	path_time_test = '../../results/channel_test/'

	data[0,0] = 82.43
	data[1,0] = 75.07


	# search for results 
	n_ds = 1
	n_ch = 64
	T = 3
	#standard
	for clas in range(3): 
		name = f'stats/valid_accu_class_{clas+2}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv'
		if os.path.isfile(path_4classtest+name):
			validation_acc_df   = np.loadtxt(path_4classtest+name)
			data[clas,0] = validation_acc_df[-1]*100

	# ds reduction
	for clas in range(3):
		ch_cnt = 1
		for n_ds in [2,3]:
			name = f'stats/valid_accu_class_{clas+2}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv'
			if os.path.isfile(path_4classtest+name):
				validation_acc_df   = np.loadtxt(path_4classtest+name)
				data[clas,ch_cnt] = validation_acc_df[-1]*100
			elif os.path.isfile(path_ds_test+name):
				validation_acc_df   = np.loadtxt(path_ds_test+name)
				data[clas,ch_cnt] = validation_acc_df[-1]*100
			ch_cnt +=1


	n_ds = 1
	# channel reduction
	for clas in range(3):
		ch_cnt = 3
		for n_ch in [38,19,8]:
			name = f'stats/valid_accu_class_{clas+2}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv'
	
			if os.path.isfile(path_4classtest+name):
				validation_acc_df   = np.loadtxt(path_4classtest+name)
				data[clas,ch_cnt] = validation_acc_df[-1]*100
			elif os.path.isfile(path_channel_test+name):

				validation_acc_df   = np.loadtxt(path_channel_test+name)
				data[clas,ch_cnt] = validation_acc_df[-1]*100
			ch_cnt +=1

	n_ch = 64
	# channel reduction
	for clas in range(3):
		ch_cnt = 6
		for T in [2,1]:
			name = f'stats/valid_accu_class_{clas+2}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv'
			if os.path.isfile(path_4classtest+name):
				validation_acc_df   = np.loadtxt(path_4classtest+name)
				data[clas,ch_cnt] = validation_acc_df[-1]*100
			elif os.path.isfile(path_time_test+name):
				validation_acc_df   = np.loadtxt(path_time_test+name)
				data[clas,ch_cnt] = validation_acc_df[-1]*100
			ch_cnt +=1





	write_to_csv('./',np.array(data))
	#np.savetxt(load_path+ 'tot_res.csv', Tot_res, fmt='%.2f', delimiter=',', header="1,2,3,4,5,6,7,8,9,m,s,1,2,3,4,5,s,m")

	# 


if __name__ == '__main__':
	main()

