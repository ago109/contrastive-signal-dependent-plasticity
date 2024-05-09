from jax import numpy as jnp, random, nn, jit
import numpy as np
import sys, getopt as gopt, optparse
from csdp_model import load_model
## bring in ngc-learn analysis tools
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from ngclearn.utils.model_utils import measure_ACC, measure_CatNLL, measure_BCE
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, extract_pca_latents, plot_latents

"""
################################################################################
CSDP Exhibit File:

Evaluates CSDP on the MNIST database.

Usage:
$ python analyze_csdp.py --dataX="/path/to/data_patterns.npy" \
                         --dataY="/path/to/labels.npy" \
                         --verbosity=0 \
                         --modelDir=exp/

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 3)

## program-specific co-routine
def eval_model(model, Xdev, Ydev, mb_size, verbosity=1): ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0]/mb_size)

    latents = []
    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    G = 0. ## goodness
    bce = 0.
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size,:]
        Yb = Ydev[idx: idx + mb_size,:]
        ## run model inference
        yMu, yCnt, _R1, _R2, _R3, xMu = model.process(Xb, Yb, dkey=dkey, adapt_synapses=False,
                                                     collect_rate_codes=True,
                                                     lab_estimator="softmax")
        ## record metric measurements
        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0] ## un-normalize score
        _acc = measure_ACC(yMu, Yb) * Yb.shape[0] ## un-normalize score
        _bce = measure_BCE(xMu, Xb) * Xb.shape[0] ## un-normalize score
        nll += _nll
        acc += _acc
        bce += _bce
        latents.append(_R2)

        n_samp_seen += Yb.shape[0]
        if verbosity > 0:
            print("\r Acc = {}  NLL = {}  BCE = {} ({} samps)".format(acc/n_samp_seen,
                                                                    nll/n_samp_seen,
                                                                    bce/n_samp_seen,
                                                                    n_samp_seen), end="")
    if verbosity > 0:
        print()
    #G = G/(Xdev.shape[0]) ## calc full dev-set goodness
    bce = bce/(Xdev.shape[0])
    nll = nll/(Xdev.shape[0]) ## calc full dev-set nll
    acc = acc/(Xdev.shape[0]) ## calc full dev-set acc
    latents = jnp.concatenate(latents, axis=0)
    return nll, acc, latents

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=", "verbosity=",
                                                    "modelDir=", "paramDir="])
modelDir = "exp/"
paramDir = "/best_params1234"
dataX = "../data/mnist/validX.npy"
dataY= "../data/mnist/validY.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--modelDir"):
        modelDir = arg.strip()
    elif opt in ("--paramDir"):
        paramDir = "/{}".format(arg.strip())
print("=> Data X: {} | Y: {}".format(dataX, dataY))

## load dataset
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
n_batches = _X.shape[0]
patch_shape = (28, 28)

dkey = random.PRNGKey(1234)

learn_recon = True
algo_type = "supervised"
T = 50
dt = 3.
model = load_model(dkey, "{}/snn_csdp".format(modelDir), algo_type=algo_type,
                   learn_recon=learn_recon, dt=dt, T=T,
                   save_subdir=paramDir) ## load in pre-trained SNN model

## evaluate performance
nll, acc, latents = eval_model(model, _X, _Y, mb_size=1000)
print("------------------------------------")
print("=> NLL = {}  Acc = {}".format(nll, acc))

## extract latents and visualize via the t-SNE algorithm
print("latent.shape = ",latents.shape)
codes = extract_tsne_latents(np.asarray(latents), perplexity=30, n_pca_comp=400)
print("code.shape = ",codes.shape)
alpha=0.325
plot_latents(codes, _Y, plot_fname="exp/snn_codes.png", alpha=alpha)
