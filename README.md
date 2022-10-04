# likelihood_emulation

Data generation is done by running an MCMC with the MontePython using the additional flag `--clik-write-cls` from the branch `clik-write-cls` on https://github.com/AarhusCosmology/montepython_public. That is,

`python /path/to/montepython/MontePython.py run -p /path/to/parameterfile.param -o /path/to/outputfolder --clik-write-cls`

This will save the Cl spectra produced for the `clik` likelihoods requested in the MontePython parameter file.

Once the output folder from a run like the above is obtained, place it in the `/data/` folder of this repository. First format the data by running:

`python clean_data.py -d data/outputfolder`

An emulator for this likelihood is then trained by running:

`python train.py -d data/outputfolder`

The hyperparameters for the training, as well as the architecture etc. are set in the first several of `train.py`. 
