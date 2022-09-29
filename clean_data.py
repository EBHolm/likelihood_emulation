import numpy as np 
import pandas as pd 
import argparse 
import os 
from natsort import os_sorted

parser = argparse.ArgumentParser(description='For further argument specifications, do python train.py --help')
parser.add_argument('-d', metavar='DATA', type=str, help="The folder containing the data", required=True)
args = parser.parse_args()

# Could also add a line cutter if this keeps being an issue!
# Currently assumes that all data files were created during the same date!

for file in os_sorted(os.listdir(args.d)):
    filename = f"{args.d}/{file}"
    if file.find('.cl') != -1:
        if not file.endswith('.clell'):
            # Collect files created with several chains
            spectrum_type = file[file.find('.cl') + 3:]
            idx = int(file[file.find('.cl') - 1])
            if idx > 1:
                data = pd.read_table(filename, sep='\t', header=None)
                first_name = f"{file[:file.find('.cl') - 1]}1.cl{spectrum_type}"
                with open(f"{args.d}/{first_name}", "ab") as f:
                    f.write(b"\n")
                    np.savetxt(f, data.values, delimiter=" \t ")
                    print(f"Appended data from chain {idx} of spectrum {spectrum_type} to {first_name}.")
                os.remove(filename)
        else:
            idx = int(file[file.find('.cl') - 1])
            if idx > 1:
                os.remove(filename)
    elif file.find('.txt') != -1:
        # Assumes no more than 9 chains!
        idx = int(file[-5])
        if idx > 1:
            data = pd.read_table(filename, sep='\t', header=None)
            first_name = f"{file[:file.find('.txt') - 1]}1.txt"
            with open(f"{args.d}/{first_name}", "ab") as f:
                    f.write(b"\n")
                    np.savetxt(f, data.values, delimiter=" \t ", fmt='%s')
                    print(f"Appended data from chain {idx} to {first_name}.")
            os.remove(filename)

# filter NaN entries
for file in os_sorted(os.listdir(args.d)):
    filename = f"{args.d}/{file}"
    if file.find('.cl') != -1:
        if not file.endswith('.clell'):
            print(f"Cleaning {file}")
            #data = pd.read_table(filename, sep='\t', header=None, dtype=pd.Float64Dtype())
            data = pd.read_table(filename, sep='\t', header=None)
            nan_rows = data.index[data.isna().any(axis=1)]
            if len(nan_rows) > 0:
                print(f"Dropping the following rows from {file}:\n   {nan_rows}")
                data.drop(nan_rows)
                np.savetxt(filename, data.values, delimiter=" \t ", fmt='%s')