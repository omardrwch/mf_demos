"""
Convert file from Bonn seizure datset to .mat format
"""
import numpy as np
import pandas as pd
from  scipy.io import loadmat, savemat
import os


FOLDER = 'C:\\Users\\omard\\Documents\\git_projects\\for_report\\seizure_mf\\bonn\\data\\setC'

def read_file(filename):
    _data = pd.read_csv(filename, sep='\n', header = None)
    return np.array(_data.values, dtype = float).squeeze()


for ii in range(10):
	filename = os.path.join(FOLDER, 'N%s.txt'%(str(ii+1).zfill(3)))
	out_filename = 'example_data/N%s.mat'%(str(ii+1).zfill(3))

	data = read_file(filename)


	data_mat = {'data':data}

	savemat(out_filename,data_mat)


