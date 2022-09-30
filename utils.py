import numpy as np
import os 

def load_data(folder):
    data_indices = {} # keys are spectrum types, values point to indices in data vector where they start
    labels_found = False
    bad_points = []
    data_list = np.array([])
    for file in os.listdir(folder):
        filename = f"{folder}/{file}"
        if file.find('.cl') != -1:
            if file.endswith(".clell"):
                print(f"Reading {file}")
                ell = np.loadtxt(filename, dtype=int)[2:-2]
            else:
                print(f"Reading {file}")
                data = np.genfromtxt(filename)
                if not labels_found:
                    loglkl = data[:, 0]
                    labels_found = True
                if np.any(np.isnan(data)):
                    pass
                spectrum_type = file[file.find('.cl')+3:]
                if data_list.shape[0] > 0:
                    data_list = np.concatenate([data_list, data[:, 3:-2]], axis=1)
                    data_indices[spectrum_type] = data_list.shape[1]
                else:
                    data_list = data[:, 3:-2]
                    data_indices[spectrum_type] = 0
        elif file.find('.nuisance') != -1:
            print(f"Reading {file}")
            nuisance_data = np.loadtxt(filename, ndmin=2)
            if data_list.shape[0] > 0:
                data_list = np.concatenate([data_list, nuisance_data], axis=1)
                data_indices['nuisance'] = data_list.shape[1]
            else:
                data_list = nuisance_data
                data_indices['nuisance'] = 0
    # remove eventual buggy values 
    loglkl    = np.delete(loglkl, np.where(np.isnan(data_list))[0], axis=0)
    data_list = np.delete(data_list, np.where(np.isnan(data_list))[0], axis=0)
    return data_list, loglkl, data_indices, ell

def symlog(x, a):
    out = np.zeros(x.shape)
    out[x > a] = np.log(x[x > a])
    out[np.abs(x) < a] = x[np.abs(x) < a]
    out[x < -a] = -np.log(-x[x < -a])
    return out

def normalize(data):
    # normalization on a per-spectra, per-ell basis 
    # to invert, we need to store the means and sigmas here 
    means, sigmas = np.zeros([data.shape[1]]), np.zeros([data.shape[1]])
    for idx_l in range(data.shape[1]):
        means[idx_l] = np.mean(data[:, idx_l])
        sigmas[idx_l] = np.std(data[:, idx_l])
        data[:, idx_l] = (data[:, idx_l] - means[idx_l])/sigmas[idx_l]
    return data, means, sigmas

def normalize_inv(data, means, sigmas):
    raise Exception("Not yet implemented!")
    return data

def loglkl_norm(loglkl):
    mean, std = np.mean(loglkl), np.std(loglkl)
    return (loglkl - mean)/std, mean, std

def loglkl_norm_inv(loglkl_norm, mean, std):
    return std*loglkl_norm + mean