from ngcsimlib.component import Component
from jax import random, numpy as jnp, jit, nn
from functools import partial
from ngclearn.utils.model_utils import initialize_params
from ngclearn.utils.optim import SGD, Adam
import time

@jit #@partial(jit, static_argnums=[6,7,8])
def calc_update(pre, x_pre, post, x_post, mod, W, w_decay, w_bound, signVal=1.):
    """
    Compute a tensor of adjustments to be applied to a synaptic value matrix.

    Args:
        pre: pre-synaptic statistic to drive CSDP update

        x_pre:

        post: post-synaptic statistic to drive CSDP update

        x_post:

        mod:

        W: synaptic weight values (at time t)

        w_decay:

        w_bound: maximum value to enforce over newly computed efficacies

        signVal: multiplicative factor to modulate final update by (good for
            flipping the signs of a computed synaptic change matrix)

    Returns:
        an update/adjustment matrix, a decay matrix
    """
    dW = jnp.matmul(pre.T, x_post)
    ## FIXME / NOTE: fix decay to be abs value of synaptic weights
    W_dec = -(jnp.matmul((1. - pre).T, (post)) * w_decay) ## post-syn decay
    db = jnp.sum(x_post, axis=0, keepdims=True)
    #if w_bound > 0.:
    #    dW = dW * (w_bound - jnp.abs(W))
    return dW * signVal, W_dec, db * signVal

@jit
def apply_soft_bound(dW, W, w_bound):
    dW = dW * (w_bound - jnp.abs(W))
    return dW

@partial(jit, static_argnums=[2,3])
def apply_decay(W, Wdecay, w_bound, is_nonnegative=True):
    ## TODO: shouldn't mask be applied to this too? (for neg samps)
    _W = W + Wdecay
    if w_bound > 0.:
        if is_nonnegative == True:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

@partial(jit, static_argnums=[2,3,4])
def adjust_synapses(dW, W, w_bound, eta, is_nonnegative=True):
    """
    Evolves/changes the synpatic value matrix underlying this synaptic cable,
    given a computed synaptic update.

    Args:
        dW: synaptic adjustment matrix to be applied/used

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        eta: global learning rate to apply to the Hebbian update

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    _W = W + dW * eta
    if w_bound > 0.:
        if is_nonnegative == True:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

@jit
def apply_hollow_mask(W, M):
    return W * (1. - M)

@jit #@partial(jit, static_argnums=[2,3]) #@jit
def compute_layer(inp, weight, sign=1., scale=1.):
    """
    Applies the transformation/projection induced by the synaptic efficacie
    associated with this synaptic cable

    Args:
        inp: signal input to run through this synaptic cable

        weight: this cable's synaptic value matrix

        sign: constrained/fixed sign to apply to synapses before transform

        scale: transformation output fixed rescale

    Returns:
        a projection/transformation of input "inp"
    """
    factor = scale * sign
    #if sign < 0.:
    #    factor = factor * scale
    return jnp.matmul(inp, weight) * factor

class CSDPSynapse(Component):
    """
    A synaptic cable that adjusts its efficacies via contrastive-signal-dependent
    plasticity (CSDP).

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        elg_tau: if > 0., triggers the use of an eligibility trace where this value
            serves as its time constant

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        signVal: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent schemes
            are to be applied (as Hebbian rules typically yield adjustments for
            ascent)

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'in'

    @classmethod
    def outputCompartmentName(cls):
        return 'out'

    @classmethod
    def modulatorName(cls):
        return 'mod'

    @classmethod
    def presynSpikeName(cls):
        return 'pre'

    @classmethod
    def postsynSpikeName(cls):
        return 'post'

    @classmethod
    def presynTraceName(cls):
        return 'x_pre'

    @classmethod
    def postsynTraceName(cls):
        return 'x_post'

    ## Bind Properties to Compartments for ease of use
    @property
    def modulator(self):
        return self.compartments.get(self.modulatorName(), None)

    @modulator.setter
    def modulator(self, x):
        self.compartments[self.modulatorName()] = x

    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, x):
        self.compartments[self.inputCompartmentName()] = x

    @property
    def outputCompartment(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @outputCompartment.setter
    def outputCompartment(self, x):
        self.compartments[self.outputCompartmentName()] = x

    @property
    def presynTrace(self):
        return self.compartments.get(self.presynTraceName(), None)

    @presynTrace.setter
    def presynTrace(self, x):
        self.compartments[self.presynTraceName()] = x

    @property
    def postsynTrace(self):
        return self.compartments.get(self.postsynTraceName(), None)

    @postsynTrace.setter
    def postsynTrace(self, x):
        self.compartments[self.postsynTraceName()] = x

    @property
    def presynSpike(self):
        return self.compartments.get(self.presynSpikeName(), None)

    @presynSpike.setter
    def presynSpike(self, x):
        self.compartments[self.presynSpikeName()] = x

    @property
    def postsynSpike(self):
        return self.compartments.get(self.postsynSpikeName(), None)

    @postsynSpike.setter
    def postsynSpike(self, x):
        self.compartments[self.postsynSpikeName()] = x

    # Define Functions
    def __init__(self, name, shape, eta, wInit=("uniform", 0., 0.3), bInit=None,
                 w_bound=1., elg_tau=0., is_nonnegative=False, w_decay=0., signVal=1.,
                 is_hollow=False, Rscale=1., w_sign=1., optim_type="sgd",
                 soft_bound=False, key=None, useVerboseDict=False, directory=None,
                 **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##params
        self.shape = shape
        self.w_bounds = w_bound ## soft thresholding function to apply to synaptic updates
        self.w_decay = w_decay ## synaptic decay
        self.soft_bound = soft_bound
        self.eta = eta
        self.wInit = wInit
        self.bInit = bInit
        self.is_nonnegative = is_nonnegative
        self.signVal = signVal
        self.is_hollow = is_hollow
        self.Rscale = Rscale
        self.w_sign = w_sign

        self.opt = None
        if optim_type == "adam":
            self.opt = Adam(learning_rate=self.eta)
        else: ## default is SGD
            self.opt = SGD(learning_rate=self.eta)

        if directory is None:
            self.key, subkey = random.split(self.key)
            self.weights = initialize_params(subkey, wInit, shape)
            self.weightMask = jnp.eye(N=shape[0], M=shape[1])
            if self.is_hollow == True:
                self.weights = apply_hollow_mask(self.weights, self.weightMask)
            if self.bInit is not None:
                self.key, subkey = random.split(self.key)
                self.biases = initialize_params(subkey, bInit, (1, shape[1]))
        else:
            self.load(directory)

        self.dW = None
        self.lW = None
        self.db = None

        ##Reset to initialize stuff
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, **kwargs):
        self.outputCompartment = compute_layer(self.inputCompartment, self.weights,
                                               sign=self.w_sign, scale=self.Rscale)

    def evolve(self, t, dt, **kwargs):
        d_z = self.postsynTrace
        d_z = d_z * self.Rscale #0.1
        dW, lW, db = calc_update(self.presynSpike, self.presynTrace, self.postsynSpike,
                                 d_z, 1., self.weights, self.w_decay,
                                 self.w_bounds, self.signVal)
        if self.soft_bound == True:
            dW = apply_soft_bound(dW, self.weights, self.w_bounds)
        self.dW = dW
        self.lW = lW
        self.db = db
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if self.bInit != None:
            theta = [self.weights, self.biases]
            self.opt.update(theta, [dW, db])
            self.weights = theta[0]
            self.biases = theta[1]
        else:
            # ignore db since no biases configured
            theta = [self.weights]
            self.opt.update(theta, [dW])
            self.weights = theta[0]
        self.weights = apply_decay(self.weights, lW, self.w_bounds,
                                   is_nonnegative=self.is_nonnegative)
        if self.is_hollow == True:
            self.weights = apply_hollow_mask(self.weights, self.weightMask)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = None
        self.presynTrace = None
        self.postsynTrace = None
        self.presynSpike = None
        self.postsynSpike = None
        self.modulator = None
        self.dW = None
        self.lW = None
        self.db = None

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bInit != None:
            jnp.savez(file_name, weights=self.weights, weightMask=self.weightMask,
                      biases=self.biases)
        else:
             jnp.savez(file_name, weights=self.weights, weightMask=self.weightMask)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
        self.weightMask = data['weightMask']
        if "biases" in data.keys():
            self.biases = data['biases']
