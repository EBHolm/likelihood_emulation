import numpy as np
import os 

def load_data(folder):
    cls = []
    spectrum_types = []
    labels_found = False
    for file in os.listdir(folder):
        filename = f"{folder}/{file}"
        if file.find('.cl') != -1:
            if file.endswith(".clell"):
                print(f"Reading {file}")
                ell = np.loadtxt(filename, dtype=int)[2:-2]
            #elif file.endswith(".cltt") or file.endswith(".clte") or file.endswith(".clee") or file.endswith(".clbb"):
            #elif file.endswith(".cltt"):
            else:
                print(f"Reading {file}")
                data = np.genfromtxt(filename)
                if not labels_found:
                    loglkl = data[:, 0]
                    labels_found = True
                spectrum_types.append(file[file.find('.cl')+3:])
                cls.append(data[:, 1:])
    cls = np.array(cls)[:, :, 2:-2] # shape (N_spectra, N_data_points, N_ell)
    cls = np.swapaxes(cls, 0, 1) # shape (N_data_points, N_spectra, N_ell)
    return cls, loglkl, spectrum_types, ell

def symlog(x, a):
    out = np.zeros(x.shape)
    out[x > a] = np.log(x[x > a])
    out[np.abs(x) < a] = x[np.abs(x) < a]
    out[x < -a] = -np.log(-x[x < -a])
    return out

def normalize(data, method=1):
    N_spectra = data.shape[1]
    if method == 1:
        # normalization on a per-spectra basis, uniformly over cosmologies and ells 
        for i in range(N_spectra):
            symlog_thresh = 1e-40
            data[:, i, :] = symlog(data[:, i, :], symlog_thresh)
            data[:, i, :] = (data[:, i, :] - np.mean(data[:, i, :]))/np.std(data[:, i, :])
    if method == 2:
        # normalization on a per-spectra, per-ell basis 
        # to invert, we need to store the means and sigmas here 
        means, sigmas = np.zeros([data.shape[1], data.shape[2]]), np.zeros([data.shape[1], data.shape[2]])
        for idx_spectrum in range(N_spectra):
            for idx_l in range(data.shape[2]):
                means[idx_spectrum, idx_l] = np.mean(data[:, idx_spectrum, idx_l])
                sigmas[idx_spectrum, idx_l] = np.std(data[:, idx_spectrum, idx_l])
                data[:, idx_spectrum, idx_l] = (data[:, idx_spectrum, idx_l] - means[idx_spectrum, idx_l])/sigmas[idx_spectrum, idx_l]
                #data[:, idx_spectrum, idx_l] = np.random.rand(data[:, idx_spectrum, idx_l].shape[0]) # check with random data! :)
    return data, means, sigmas

def normalize_inv(data, means, sigmas, method=2):
    # assume method 2
    if method == 1:
        raise Exception("Method 1 not implemented yet!")
    elif method == 2:
        raise Exception("Method 2 not implemented yet!")
    return data

def loglkl_norm(loglkl):
    mean, std = np.mean(loglkl), np.std(loglkl)
    return (loglkl - mean)/std, mean, std

def loglkl_norm_inv(loglkl_norm, mean, std):
    return std*loglkl_norm + mean