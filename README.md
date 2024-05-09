[![DOI](https://zenodo.org/badge/798347724.svg)](https://zenodo.org/doi/10.5281/zenodo.11165561)

# Contrastive-Signal-Dependent-Plasticity (CSDP)

This is the code for the paper:
"Contrastive-Signal-Dependent-Plasticity: Self-Supervised Learning in Spiking Neural Circuits"
a preprint of which can be found here:
https://arxiv.org/abs/2303.18187

Make sure you unzip the mnist data prepared for you in the `/data/` folder
(unzip `/data/mnist.zip` and place it inside of `/data/`).
To train a CSDP SNN model (with `1024` neuronal cells in each layer), run the
following prepared BASH script:
```console
./sim_csdp.sh
```

This will train a CSDP SNN model on the MNIST database for you.
Furthermore, the script will generate the model structure (in ngc-learn JSON format) as well as
store NPZ files containing your best found parameters during training. All of this
will be stored, if you run the script in its default mode (i.e., w/o modifying
its arguments) to a folder `exp_supervised_mnist/` which contains your saved
ngc-learn CSDP SNN model.

To evaluate your CSDP model after training it, run the following analysis script
```console
python analyze_csdp.py  --dataX=data/mnist/testX.npy \
                        --dataY=data/mnist/testY.npy \
                        --modelDir=exp_supervised_mnist/ \
                        --paramDir=best_params1234
```

For the analysis script, the parameter sub-directory can be toggled by changing
the "paramDir" argument, which is simply the name of the sub-directory within
your model directory "modelDir" that contains the saved NPZ synaptic arrays.
Inside the output directory it creates `/exp/`, you will find a t-SNE plot
of your model's extracted latent codes.
