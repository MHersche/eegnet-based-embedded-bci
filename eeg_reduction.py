#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Authors: Batuhan Toemekce, Burak Kaya, Michael Hersche                     *
#*----------------------------------------------------------------------------*

#!/usr/bin/env python3

__author__ = "Michael Hersche"
__email__ = "herschmi@iis.ee.ethz.ch"

import numpy as np
import scipy.signal as scp
import pdb


def eeg_reduction(x, n_ds = 1, n_ch = 64, T = 3, fs = 160):
	'''
	Inputs
	------
	x : np array ()
		input array 
	n_ds: int 
		downsampling factor 
	n_ch: int 
		number of channels
	T: float 
		time [s] to classify
	fs: int 
		samlping frequency [Hz]


	Outputs
	-------
	'''


	if n_ch ==64: 
		channels = np.arange(0,n_ch)
	elif n_ch == 38: 
		channels = np.array([0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,29,31,33,35,37,40,41,42,43,46,48,50,52,54,55,57,59,60,61,62,63])
	elif n_ch == 27: 
		channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,38,39,40,41,44,45])
	elif n_ch == 19: 
		channels = np.array([8,10,12,21,23,29,31,33,35,37,40,41,46,48,50,52,54,60,62])
	elif n_ch ==8: 
		channels = np.array([8,10,12,25,27,48,52,57])


	n_s_orig = int(T*fs)
	n_s = int(np.ceil(T*fs/n_ds)) # number of time samples
	n_trial = x.shape[0]

	# channel selection 
	if n_ds >1: 
		x = x[:,channels]
		y = np.zeros((n_trial, n_ch,n_s))
		for trial in range(n_trial):	
			for chan in range(n_ch): 
				# downsampling 
				#pdb.set_trace()
				y[trial,chan] = scp.decimate(x[trial,chan,:n_s_orig],n_ds)
	else: 
		y = x[:,channels]
		y = y[:,:,:n_s_orig]

	return y





