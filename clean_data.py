import numpy as np 
import pandas as pd 
import argparse 
import os 
from natsort import os_sorted

parser = argparse.ArgumentParser(description='For further argument specifications, do python train.py --help')
parser.add_argument('-d', metavar='DATA', type=str, help="The folder containing the data", required=True)
args = parser.parse_args()

# Currently assumes that all data files were created during the same date. This should never be a problem!

def get_idx(filename):
    idx_start = len(filename) - filename[::-1].find('_')
    idx_end = filename.find('.')
    idx = 0
    if idx_start != -1 and idx_end != -1 and idx_end - idx_start != 0:
        try:
            idx = int(filename[idx_start:idx_end])
        except:
            pass
    return idx

idx_max = 1
for file in os_sorted(os.listdir(args.d)):
    filename = f"{args.d}/{file}"
    idx = get_idx(filename)
    if idx > idx_max:
        idx_max = idx

for idx in range(idx_max):
    length_dicts = {}
    min_len = np.inf
    for file in os_sorted(os.listdir(args.d)):
        filename = f"{args.d}/{file}"
        idx_new = get_idx(filename)
        if idx_new != 0 and idx_new == idx:
            if file.find('.clell') == -1 and file.find('.txt') == -1:
                data = pd.read_table(filename, sep='\t', header=None)
                cur_len = data.values.shape[0]
                length_dicts[filename] = cur_len
                if cur_len < min_len:
                    min_len = cur_len
                #print(f"File {filename} has length {cur_len}")
    for key, N_lines in length_dicts.items():
        if N_lines > min_len:
            N_cut_lines = N_lines - min_len
            print(f"Removing {N_cut_lines} lines from {key}")
            data = np.loadtxt(f"{key}", delimiter=" \t", ndmin=2)
            np.savetxt(f"{key}", data[:-N_cut_lines,:], delimiter=" \t")


# Collect files created with several chains
for file in os_sorted(os.listdir(args.d)):
    filename = f"{args.d}/{file}"
    idx_start = len(file) - file[::-1].find('_')
    idx_end = file.find('.')
    if idx_start != -1 and idx_end != -1 and idx_end - idx_start != 0:
        try:
            idx = int(file[idx_start:idx_end])
        except:
            print(f"passing on file {file}")
            idx = 0 # don't do anything in this case 
        if idx > 1:
            if file.find('.cl') != -1:
                if not file.endswith('.clell'):
                    spectrum_type = file[file.find('.cl') + 3:]
                    data = pd.read_table(filename, sep='\t', header=None)
                    first_name = f"{file[:idx_start]}1.cl{spectrum_type}"
                    with open(f"{args.d}/{first_name}", "ab") as f:
                        f.write(b"\n")
                        np.savetxt(f, data.values, delimiter=" \t ", fmt='%s')
                        #np.savetxt(f, data.values, delimiter=" \t ", fmt='%.18e')
                        print(f"Appended data from chain {idx} of spectrum {spectrum_type} to {first_name}.")
                    os.remove(filename)
                else:
                    os.remove(filename)
            elif file.find('.txt') != -1:
                data = pd.read_table(filename, sep='\t', header=None)
                first_name = f"{file[:idx_start]}1.txt"
                with open(f"{args.d}/{first_name}", "ab") as f:
                        f.write(b"\n")
                        np.savetxt(f, data.values, delimiter=" \t ", fmt='%s')
                        print(f"Appended data from chain {idx} to {first_name}.")
                os.remove(filename)
            elif file.find('.nuisance') != -1:
                data = pd.read_table(filename, sep='\t', header=None)
                first_name = f"{file[:idx_start]}1.nuisance"
                with open(f"{args.d}/{first_name}", "ab") as f:
                    f.write(b"\n")
                    np.savetxt(f, data.values, delimiter=" \t ", fmt='%.18e')
                    print(f"Appended data from chain {idx} of nuisance files to {first_name}.")
                os.remove(filename)
            else: # should never happen unless some strange files are in the folder 
                print(f"passed on {file} even though idx={idx}")

# filter NaN entries
have_cleaned = False
for file in os_sorted(os.listdir(args.d)):
    filename = f"{args.d}/{file}"
    if file.find('.cl') != -1:
        if not file.endswith('.clell'):
            #print(f"Removing non-uniform entries in {file}")
            #data = pd.read_table(filename, sep='\t', header=None, dtype=np.float64)
            data = pd.read_table(filename, sep='\t', header=None)
            nan_rows = data.index[data.isna().any(axis=1)]
            if len(nan_rows) > 0:
                print(f"Dropping the following rows from {file}:\n   {nan_rows}")
                data.drop(nan_rows)
                np.savetxt(filename, data.values, delimiter=" \t ", fmt='%s')
                #np.savetxt(filename, data.values, delimiter=" \t ", fmt='%.18e')
                have_cleaned = True
# filter the same rows in the nuisance parameter file
if have_cleaned:
    for file in os_sorted(os.listdir(args.d)):
        filename = f"{args.d}/{file}"
        if file.find('.nuisance') != -1:
            print(f"Dropping the following rows from {file}:\n   {nan_rows}")
            data = pd.read_table(filename, sep='\t', header=None)
            data.drop(nan_rows)
            np.savetxt(filename, data.values, delimiter=" \t ", fmt='%s')

