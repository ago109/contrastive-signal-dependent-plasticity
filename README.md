[![DOI](https://zenodo.org/badge/798347724.svg)](https://zenodo.org/doi/10.5281/zenodo.11165561)

# Contrastive-Signal-Dependent-Plasticity (CSDP)

This is the code for the paper:
"Contrastive-Signal-Dependent-Plasticity: Self-Supervised Learning in Spiking Neural Circuits"
a preprint of which can be found here:
https://arxiv.org/abs/2303.18187 <br>
Note that this code was written on/run on an Ubuntu 22.04.2 LTS and 
assumes that you have Python 3.10.6, jax/jaxlib 0.4.28 (for Cuda-12), and 
ngclearn 1.2.b3 (with ngcsimlib 0.3.b4) successfully installed on your system.

## Installation

If you have Python 3.10.6 installed, you can automatically configure the needed dependencies 
via pip. It is recommended that you first create a separate Python virtual environment (VM) to 
act as a playground for this code (and protect your working environment) like so:

```consolve
python3.10 -m venv env_csdp  ## create a playground VM
source env_csdp/bin/activate  ## activate/enter the playground VM
```

You can then install the required libraries/modules in your Python VM via the pip package installer like so:

```console
pip install -r requirements.txt  ## install required libraries in your playground VM
```

<i>Note:</i> Running the above pip command will ensure that you have the GPU-enabled variants of 
JAX and NGC-Sim-Lib/NGC-Learn.

## Running the Model Simulation 

In order to run the simulation, make sure you unzip the mnist data prepared for you in 
the `/data/` folder (unzip `/data/mnist.zip` and place it inside of `/data/`).
To train a CSDP SNN model (with `3000` neuronal cells in the first layer and `600` cells 
in the second one), run the following prepared BASH script:
```console
./sim_csdp.sh 0
```

This will train a CSDP SNN model on the MNIST database for you (on the GPU with identifier 0; 
if you want to run a different GPU, choose another appropriate integer identifier).
Furthermore, the script will generate the model structure (in ngc-learn JSON format) as well as
store NPZ files containing your best found parameters during training. All of this
will be stored, if you run the script in its default mode (i.e., w/o modifying
its arguments) to a folder `exp_supervised_mnist/` which contains your saved
ngc-learn CSDP SNN model.

<i>Note:</i> You can safely ignore the warnings collected in auto-generated `logging/` directoy as these 
are simply where ngc-learn/sim-lib store library messages.

## Running the Model Evaluation/Analysis

To evaluate your CSDP model after training it as above, run the following analysis BASH script:
```console
./eval_csdp.sh 0
```

This script will run your CSDP SNN model (inference-only) on the test subset of the MNIST database. 
Inside your model directory, e.g., `exp_supervised_mnist/`, the analysis script above also creates a 
sub-directory called `/tsne/`. It is in here that you will find a t-SNE plot of your model's 
extracted latent codes (as well as a numpy array containing the tSNE embedding codes).


If you use this code or model mathematics in any form in your project(s), please cite its source
paper:
<pre>
@article{ororbia2023contrastive,
  title={Contrastive-signal-dependent plasticity: Forward-forward learning of spiking neural systems},
  author={Ororbia, Alexander},
  journal={arXiv preprint arXiv:2303.18187},
  year={2023}
}
</pre>
