from ngcsimlib.controller import Controller
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit, nn
import time, sys
from ngclearn.utils.model_utils import softmax
from img_utils import csdp_deform #vrotate, rand_rotate,

## SNN model co-routines
def load_model(dkey, model_dir, algo_type, learn_recon, exp_dir="exp", model_name="snn_csdp",
               dt=3., T=50, batch_size=1, save_subdir="/custom"):
    #_key = random.PRNGKey(time.time_ns())
    ## load circuit from disk
    circuit = Controller()
    circuit.load_from_dir(directory=model_dir, custom_folder=save_subdir)
    _model_dir = "{}/{}/{}".format(exp_dir, model_name, save_subdir)
    print("Load from: ",_model_dir)

    model = CSDP_SNN(dkey, in_dim=1, out_dim=1, batch_size=batch_size, dt=dt, T=T,
                     save_init=False, algo_type=algo_type, learn_recon=learn_recon)
    model.circuit = circuit
    model.exp_dir = exp_dir
    model.model_dir = _model_dir
    return model

@jit
def _add(x, y): ## jit-i-fied vector-matrix add
    return x + y

class CSDP_SNN():
    def __init__(self, dkey, in_dim, out_dim, hid_dim=1024, hid_dim2=1024, batch_size=1,
                 eta_w=0.002, T=40, dt=3., exp_dir="exp", model_name="snn_csdp",
                 save_init=True, algo_type="supervised", learn_recon=False, **kwargs):
        self.exp_dir = exp_dir
        makedir(exp_dir)
        self.algo_type = algo_type

        #dkey = random.PRNGKey(1234)
        dkey, *subkeys = random.split(dkey, 16)

        self.T = T ## num discrete time steps to simulate
        self.dt = dt ## integration time constant

        ## hard-coded model meta-parameters

        self.learn_recon = learn_recon
        # spiking cell parameters
        tau_m = 100. # ms ## membrane time constant (paper used 2 ms)
        vThr = 0.055 #0.8 ## membrane potential threshold (to emit a spike)
        R_m = 0.1 #1. ## input resistance (to sLIF cells)
        inh_R = 0.01 #0.035 ## inhibitory resistance
        rho_b = 0.001 ## adaptive threshold sparsity constant
        tau_tr = 13.

        # Synaptic initialization conditions
        weightInit = ("uniform", -1., 1.)
        biasInit = None #("constant", 0., 0.) ## init biases from zero values

        # CSDP-specific meta-parameters
        self.use_rot = False ## for unsupervised csdp only
        self.alpha = 0.5 ## for unsupervised csdp only
        optim_type = "adam"
        goodnessThr = 10.
        nonneg_w = False ## should non-lateral synapses be constrained to be positive-only?

        if batch_size >= 200: ## heuristic learning problem scaling
            eta_w = 0.002
            w_decay = 0.00005
        elif batch_size >= 100:
            eta_w = 0.001
            w_decay = 0.00006
        elif batch_size >= 50:
            eta_w = 0.001
            w_decay = 0.00007
        elif batch_size >= 20: #25:
            eta_w = 0.00075
            w_decay = 0.00008
        elif batch_size >= 10:
            eta_w = 0.00055
            w_decay = 0.00009
        else:
            eta_w = 0.0004
            w_decay = 0.0001
        soft_bound = False

        if algo_type == "unsupervised":
            R_m = 0.1
            goodnessThr = 10.
            inh_R = 0.01
            soft_bound = False
            self.use_rot = True
        ## else, supervised setting as above

        ################################################################################
        ## Create/configure model and simulation object
        circuit = Controller()
        #z0 = model.add_component("poiss", name="z0", n_units=in_dim, max_freq=63.75, key=subkeys[0])
        z0 = circuit.add_component("bernoulli", name="z0", n_units=in_dim, key=subkeys[0])
        z0.batch_size = batch_size * 2
        W1 = circuit.add_component("csdp", name="W1", shape=(in_dim, hid_dim),
                                 eta=eta_w, wInit=weightInit, bInit=biasInit, w_bound=1.,
                                 is_nonnegative=nonneg_w, elg_tau=0., w_decay=w_decay,
                                 Rscale=R_m, optim_type=optim_type,
                                 soft_bound=soft_bound, key=subkeys[1])
        ## layer 1
        z1 = circuit.add_component("SLIF", name="z1", n_units=hid_dim, tau_m=tau_m, R_m=1.,
                                 thr=vThr, inhibit_R=0., refract_T=0., thrGain=0.,
                                 thrLeak=0., rho_b=rho_b, sticky_spikes=False,
                                 thr_jitter=0.025, key=subkeys[2])
        z1.batch_size = batch_size * 2
        W2 = circuit.add_component("csdp", name="W2", shape=(hid_dim, hid_dim2),
                                 eta=eta_w, wInit=weightInit, bInit=biasInit, w_bound=1.,
                                 is_nonnegative=nonneg_w, elg_tau=0., w_decay=w_decay,
                                 Rscale=R_m, optim_type=optim_type,
                                 soft_bound=soft_bound, key=subkeys[3])
        ## layer 2
        z2 = circuit.add_component("SLIF", name="z2", n_units=hid_dim2, tau_m=tau_m, R_m=1.,
                                 thr=vThr, inhibit_R=0., refract_T=0., thrGain=0.,
                                 thrLeak=0., rho_b=rho_b, sticky_spikes=False,
                                 thr_jitter=0.025, key=subkeys[4])
        z2.batch_size = batch_size * 2
        V2 = circuit.add_component("csdp", name="V2", shape=(hid_dim2, hid_dim),
                                 eta=eta_w, wInit=weightInit, w_bound=1.,
                                 is_nonnegative=nonneg_w, w_decay=w_decay,
                                 Rscale=R_m, optim_type=optim_type,
                                 soft_bound=soft_bound, key=subkeys[5])
        M1 = circuit.add_component("csdp", name="M1", shape=(hid_dim, hid_dim),
                                 eta=eta_w, wInit=weightInit, w_bound=1.,
                                 is_nonnegative=True, w_decay=w_decay, is_hollow=True,
                                 Rscale=inh_R, w_sign=-1., optim_type=optim_type,
                                 soft_bound=soft_bound, key=subkeys[6])
        M2 = circuit.add_component("csdp", name="M2", shape=(hid_dim2, hid_dim2),
                                 eta=eta_w, wInit=weightInit, w_bound=1.,
                                 is_nonnegative=True, w_decay=w_decay, is_hollow=True,
                                 Rscale=inh_R, w_sign=-1., optim_type=optim_type,
                                 soft_bound=soft_bound, key=subkeys[7])

        ## layer 3
        ### output prediction
        zy = circuit.add_component("SLIF", name="zy", n_units=out_dim, tau_m=tau_m, R_m=1.,
                                 thr=vThr, inhibit_R=0., refract_T=0., thrGain=0.,
                                 thrLeak=0., rho_b=rho_b, sticky_spikes=False,
                                 thr_jitter=0.025, key=subkeys[4])
        ey = circuit.add_component("error", name="ey", n_units=out_dim)
        C2 = circuit.add_component("hebbian", name="C2", shape=(hid_dim, out_dim),
                                   eta=eta_w, wInit=weightInit, bInit=biasInit,
                                   w_bound=1.,
                                   is_nonnegative=nonneg_w, w_decay=0., signVal=1.,
                                   Rscale=R_m, post_wght=R_m, optim_type=optim_type, key=subkeys[8])
        C3 = circuit.add_component("hebbian", name="C3", shape=(hid_dim2, out_dim),
                                   eta=eta_w, wInit=weightInit, bInit=biasInit,
                                   w_bound=1.,
                                   is_nonnegative=nonneg_w, w_decay=0., signVal=1.,
                                   Rscale=R_m, post_wght=R_m, optim_type=optim_type, key=subkeys[9])
        if self.learn_recon == True:
            zR = circuit.add_component("SLIF", name="zR", n_units=in_dim, tau_m=tau_m,
                                     R_m=1., thr=vThr, inhibit_R=0., refract_T=0., thrGain=0.,
                                     thrLeak=0., rho_b=0., sticky_spikes=False,
                                     thr_jitter=0.025, key=subkeys[4])
            eR = circuit.add_component("error", name="eR", n_units=in_dim)

            R1 = circuit.add_component("hebbian", name="R1", shape=(hid_dim, in_dim),
                                       eta=eta_w, wInit=weightInit, bInit=biasInit,
                                       w_bound=1.,
                                       is_nonnegative=nonneg_w, w_decay=0., signVal=-1.,
                                       Rscale=R_m, post_wght=R_m, optim_type=optim_type,
                                       key=subkeys[10])

        ### context/class units (only used if model is supervised)
        z3 = circuit.add_component("bernoulli", name="z3", n_units=out_dim, key=subkeys[8])
        z3.batch_size = batch_size * 2
        if algo_type == "supervised":
            V3y = circuit.add_component("csdp", name="Y3", shape=(out_dim, hid_dim2),
                                      eta=eta_w, wInit=weightInit, w_bound=1.,
                                      is_nonnegative=nonneg_w, w_decay=w_decay,
                                      optim_type=optim_type, soft_bound=soft_bound,
                                      Rscale=R_m, key=subkeys[10])
            # no ablation applied here since context is an "input" at least to hidden layer
            V2y = circuit.add_component("csdp", name="Y2", shape=(out_dim, hid_dim),
                                      eta=eta_w, wInit=weightInit, w_bound=1.,
                                      is_nonnegative=nonneg_w, w_decay=w_decay,
                                      optim_type=optim_type, soft_bound=soft_bound,
                                      Rscale=R_m, key=subkeys[11])
        ## create goodness modulators (1 per layer)
        g1 = circuit.add_component("goodness", name="g1", threshold=goodnessThr)
        g2 = circuit.add_component("goodness", name="g2", threshold=goodnessThr)
        # no g3 since layer 3 is classification layer if applicable
        ########################################################################
        ## static (state) recordings of spike values at t - dt
        z0_prev = circuit.add_component("rate", name="z0_prev", n_units=in_dim,
                                        tau_m=0., leakRate=0.)
        z1_prev = circuit.add_component("rate", name="z1_prev", n_units=hid_dim,
                                        tau_m=0., leakRate=0.)
        z2_prev = circuit.add_component("rate", name="z2_prev", n_units=hid_dim2,
                                        tau_m=0., leakRate=0.)
        z3_prev = circuit.add_component("rate", name="z3_prev", n_units=out_dim,
                                        tau_m=0., leakRate=0.)
        ########################################################################
        ### add trace variables
        tr0 = circuit.add_component("trace", name="tr0", n_units=in_dim, tau_tr=tau_tr,
                                    decay_type="lin", a_delta=0., key=subkeys[12])
        tr1 = circuit.add_component("trace", name="tr1", n_units=hid_dim, tau_tr=tau_tr,
                                    decay_type="lin", a_delta=0., key=subkeys[12])
        tr2 = circuit.add_component("trace", name="tr2", n_units=hid_dim2, tau_tr=tau_tr,
                                    decay_type="lin", a_delta=0., key=subkeys[13])
        tr3 = circuit.add_component("trace", name="tr3", n_units=out_dim, tau_tr=tau_tr,
                                    decay_type="lin", a_delta=0., key=subkeys[14])
        tr0.batch_size = batch_size * 2
        tr1.batch_size = batch_size * 2
        tr2.batch_size = batch_size * 2
        tr3.batch_size = batch_size * 2

        ## wire nodes to their respective traces
        circuit.connect(z1.name, z1.outputCompartmentName(),
                      tr1.name, tr1.inputCompartmentName())
        circuit.connect(z2.name, z2.outputCompartmentName(),
                      tr2.name, tr2.inputCompartmentName())
        circuit.connect(zy.name, zy.outputCompartmentName(),
                      tr3.name, tr3.inputCompartmentName())
        ## wire traces of nodes to their respective goodness modulators
        circuit.connect(tr1.name, tr1.traceName(),
                      g1.name, g1.inputCompartmentName())
        circuit.connect(tr2.name, tr2.traceName(),
                      g2.name, g2.inputCompartmentName())
        ## wires nodes to their respective previous time-step recordings
        circuit.connect(z0.name, z0.outputCompartmentName(),
                      z0_prev.name, z0_prev.inputCompartmentName())
        circuit.connect(z1.name, z1.outputCompartmentName(),
                      z1_prev.name, z1_prev.inputCompartmentName())
        circuit.connect(z2.name, z2.outputCompartmentName(),
                      z2_prev.name, z2_prev.inputCompartmentName())
        circuit.connect(z3.name, z3.outputCompartmentName(),
                      z3_prev.name, z3_prev.inputCompartmentName())
        ########################################################################
        ### Wire layers together

        ## z1 and z2 to zy classifier
        circuit.connect(z3.name, z3.outputCompartmentName(), ey.name, ey.meanName())
        circuit.connect(tr3.name, tr3.traceName(), ey.name, ey.targetName())

        circuit.connect(z1.name, z1.outputCompartmentName(), C2.name, C2.inputCompartmentName())
        circuit.connect(C2.name, C2.outputCompartmentName(), zy.name, zy.inputCompartmentName())
        circuit.connect(z2.name, z2.outputCompartmentName(), C3.name, C3.inputCompartmentName())
        circuit.connect(C3.name, C3.outputCompartmentName(), zy.name, zy.inputCompartmentName(),
                        bundle="fast_add")

        if self.learn_recon == True:
            circuit.connect(z1.name, z1.outputCompartmentName(), R1.name, R1.inputCompartmentName())
            circuit.connect(R1.name, R1.outputCompartmentName(), zR.name, zR.inputCompartmentName())
            circuit.connect(zR.name, zR.outputCompartmentName(), tr0.name, tr0.inputCompartmentName())
            #circuit.connect(tr0.name, tr0.traceName(), eR.name, eR.meanName())
            circuit.connect(zR.name, zR.outputCompartmentName(), eR.name, eR.meanName())
            circuit.connect(z0.name, z0.outputCompartmentName(), eR.name, eR.targetName())
            ## set up R1 plasticity rule
            circuit.connect(z1.name, z1.outputCompartmentName(), R1.name, R1.presynapticCompartmentName())
            circuit.connect(eR.name, eR.derivMeanName(), R1.name, R1.postsynapticCompartmentName())


        ## layer 0 to 1
        circuit.connect(z0.name, z0.outputCompartmentName(), W1.name, W1.inputCompartmentName())
        circuit.connect(W1.name, W1.outputCompartmentName(), z1.name, z1.inputCompartmentName())
        ## layer 1 to 1 (lateral)
        circuit.connect(z1_prev.name, z1_prev.outputCompartmentName(), M1.name, M1.inputCompartmentName())
        circuit.connect(M1.name, M1.outputCompartmentName(), z1.name, z1.inputCompartmentName(),
                      bundle="fast_add")
        ## layer 1 to 2
        circuit.connect(z1.name, z1.outputCompartmentName(), W2.name, W2.inputCompartmentName())
        circuit.connect(W2.name, W2.outputCompartmentName(), z2.name, z2.inputCompartmentName())
        ## layer 2 to 2 (lateral)
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), M2.name, M2.inputCompartmentName())
        circuit.connect(M2.name, M2.outputCompartmentName(), z2.name, z2.inputCompartmentName(),
                      bundle="fast_add")
        ## layer 2 to 1
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), V2.name, V2.inputCompartmentName())
        circuit.connect(V2.name, V2.outputCompartmentName(), z1.name, z1.inputCompartmentName(),
                      bundle="fast_add")
        if algo_type == "supervised":
            ## connect context to layer 2
            circuit.connect(z3_prev.name, z3_prev.outputCompartmentName(), V3y.name, V3y.inputCompartmentName())
            circuit.connect(V3y.name, V3y.outputCompartmentName(), z2.name, z2.inputCompartmentName(),
                          bundle="fast_add")
            ## connect context to layer 1
            circuit.connect(z3_prev.name, z3_prev.outputCompartmentName(), V2y.name, V2y.inputCompartmentName())
            circuit.connect(V2y.name, V2y.outputCompartmentName(), z1.name, z1.inputCompartmentName(),
                          bundle="fast_add")

        ## wire relevant compartment stats to trigger plasticity rules
        circuit.connect(z1_prev.name, z1_prev.outputCompartmentName(), C2.name, C2.presynapticCompartmentName())
        circuit.connect(ey.name, ey.derivMeanName(), C2.name, C2.postsynapticCompartmentName())
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), C3.name, C3.presynapticCompartmentName())
        circuit.connect(ey.name, ey.derivMeanName(), C3.name, C3.postsynapticCompartmentName())

        ## update to W1
        circuit.connect(z0_prev.name, z0_prev.outputCompartmentName(), W1.name, W1.presynSpikeName())
        circuit.connect(z1.name, z1.outputCompartmentName(), W1.name, W1.postsynSpikeName())
        circuit.connect(z0_prev.name, z0_prev.outputCompartmentName(), W1.name, W1.presynTraceName())
        circuit.connect(g1.name, g1.modulatorName(), W1.name, W1.postsynTraceName())
        ## update to M1
        circuit.connect(z1_prev.name, z1_prev.outputCompartmentName(), M1.name, M1.presynSpikeName())
        circuit.connect(z1.name, z1.outputCompartmentName(), M1.name, M1.postsynSpikeName())
        circuit.connect(z1_prev.name, z1_prev.outputCompartmentName(), M1.name, M1.presynTraceName())
        circuit.connect(g1.name, g1.modulatorName(), M1.name, M1.postsynTraceName())
        ## update to W2
        circuit.connect(z1_prev.name, z1_prev.outputCompartmentName(), W2.name, W2.presynSpikeName())
        circuit.connect(z2.name, z2.outputCompartmentName(), W2.name, W2.postsynSpikeName())
        circuit.connect(z1_prev.name, z1_prev.outputCompartmentName(), W2.name, W2.presynTraceName())
        circuit.connect(g2.name, g2.modulatorName(), W2.name, W2.postsynTraceName())
        ## update to M2
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), M2.name, M2.presynSpikeName())
        circuit.connect(z2.name, z2.outputCompartmentName(), M2.name, M2.postsynSpikeName())
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), M2.name, M2.presynTraceName())
        circuit.connect(g2.name, g2.modulatorName(), M2.name, M2.postsynTraceName())
        ## update to V2
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), V2.name, V2.presynSpikeName())
        circuit.connect(z1.name, z1.outputCompartmentName(), V2.name, V2.postsynSpikeName())
        circuit.connect(z2_prev.name, z2_prev.outputCompartmentName(), V2.name, V2.presynTraceName())
        circuit.connect(g1.name, g1.modulatorName(), V2.name, V2.postsynTraceName())
        if algo_type == "supervised":
            ## update to Y3
            circuit.connect(z3_prev.name, z3_prev.outputCompartmentName(), V3y.name, V3y.presynSpikeName())
            circuit.connect(z2.name, z2.outputCompartmentName(), V3y.name, V3y.postsynSpikeName())
            circuit.connect(z3_prev.name, z3_prev.outputCompartmentName(), V3y.name, V3y.presynTraceName())
            circuit.connect(g2.name, g2.modulatorName(), V3y.name, V3y.postsynTraceName())
            ## update to Y2
            circuit.connect(z3_prev.name, z3_prev.outputCompartmentName(), V2y.name, V2y.presynSpikeName())
            circuit.connect(z1.name, z1.outputCompartmentName(), V2y.name, V2y.postsynSpikeName())
            circuit.connect(z3_prev.name, z3_prev.outputCompartmentName(), V2y.name, V2y.presynTraceName())
            circuit.connect(g1.name, g1.modulatorName(), V2y.name, V2y.postsynTraceName())
        ########################################################################
        ## make key commands known to model
        if self.algo_type == "supervised":
            exec_path = [z0_prev.name, z1_prev.name, z2_prev.name, z3_prev.name,
                         W1.name, V2.name, W2.name, V3y.name, V2y.name, M1.name,
                         M2.name, C2.name, C3.name,
                         z0.name, z1.name, z2.name, z3.name, zy.name,
                         tr1.name, tr2.name, tr3.name,
                         g1.name, g2.name, ey.name]
            evolve_path = [W1.name, V2.name, W2.name, V2y.name, V3y.name,
                           M1.name, M2.name, C2.name, C3.name]
            save_path = [W1.name, V2.name, W2.name, V2y.name, V3y.name,
                         M1.name, M2.name, C2.name, C3.name,
                         z1.name, z2.name, zy.name]
        else: ## unsupervised
            exec_path = [z0_prev.name, z1_prev.name, z2_prev.name, z3_prev.name,
                         W1.name, V2.name, W2.name, M1.name,
                         M2.name, C2.name, C3.name,
                         z0.name, z1.name, z2.name, z3.name, zy.name,
                         tr1.name, tr2.name, tr3.name,
                         g1.name, g2.name, ey.name]
            evolve_path = [W1.name, V2.name, W2.name,
                           M1.name, M2.name, C2.name, C3.name]
            save_path = [W1.name, V2.name, W2.name, M1.name, M2.name, C2.name,
                         C3.name, z1.name, z2.name, zy.name]
        if self.learn_recon == True:
            recon_path = [R1.name, zR.name, tr0.name, eR.name]
            exec_path = exec_path + recon_path
            evolve_path = evolve_path + [R1.name]
            save_path = save_path + [zR.name, R1.name]
        circuit.add_command("reset", command_name="reset",
                            component_names=exec_path,
                            reset_name="do_reset")
        circuit.add_command("advance", command_name="advance",
                            component_names=exec_path)
        circuit.add_command("evolve", command_name="evolve",
                            component_names=evolve_path)
        circuit.add_command("clamp", command_name="clamp_input",
                            component_names=[z0.name], compartment=z0.inputCompartmentName(),
                            clamp_name="x")
        circuit.add_command("clamp", command_name="clamp_target",
                            component_names=[z3.name], compartment=z3.inputCompartmentName(),
                            clamp_name="y")
        circuit.add_command("clamp", command_name="clamp_mod_labels",
                            component_names=[g1.name, g2.name, ey.name, eR.name],
                            compartment="lab",
                            clamp_name="mod_labels")
        circuit.add_command("save", command_name="save",
                           component_names=save_path,
                           directory_flag="dir")
        ########################################################################

        self.cell_components = ["z0", "z1", "z2", "z3", "zy",
                                "tr1", "tr2", "tr3", "ey",
                                "z0_prev", "z1_prev", "z2_prev", "z3_prev",
                                "g1", "g2"]

        ## tell model the order in which to run automatic commands
        circuit.add_step("advance")

        if save_init == True: ## save JSON structure to disk once
            circuit.save_to_json(directory=exp_dir, model_name=model_name)
        self.model_dir = "{}/{}/custom".format(exp_dir, model_name)
        self.model_name = model_name
        self.circuit = circuit # embed circuit to model construct
        ################################################################################

    def save_to_disk(self, save_dir):
        """
        Saves current model parameter values to disk

        Args:
            save_subdir:
        """
        model_dir = "{}/{}/{}".format(self.exp_dir, self.model_name, save_dir)
        makedir(model_dir)
        self.circuit.save(dir=model_dir)

    def load_from_disk(self, model_directory="exp"):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit.load_from_dir(self, model_directory)

    def process(self, Xb, Yb, dkey, adapt_synapses=False, collect_spikes=False,
                collect_rate_codes=False, lab_estimator="softmax"):
        dkey, *subkeys = random.split(dkey, 2)
        if adapt_synapses == True:
            ## create negative sensory samples
            if self.algo_type == "supervised":
                Yb_neg = random.uniform(subkeys[0], Yb.shape, minval=0., maxval=1.) * (1. - Yb)
                Yb_neg = nn.one_hot(jnp.argmax(Yb_neg, axis=1), num_classes=Yb.shape[1], dtype=jnp.float32)
                Xb_neg = Xb
            else: ## algo_type is unsupervised
                bsize = Xb.shape[0]
                _Xb = jnp.expand_dims(jnp.reshape(Xb, (bsize, 28, 28)), axis=3)
                Xb_neg, Yb_neg = csdp_deform(subkeys[0], _Xb, Yb, alpha=self.alpha,
                                             use_rot=self.use_rot)
                Xb_neg = jnp.reshape(jnp.squeeze(Xb_neg, axis=3), (bsize, 28 * 28))
            ## concatenate the samples
            _Xb = jnp.concatenate((Xb, Xb_neg), axis=0)
            _Yb = jnp.concatenate((Yb, Yb_neg), axis=0)
            mod_signal = jnp.concatenate((jnp.ones((Xb.shape[0],1)),
                                          jnp.zeros((Xb_neg.shape[0],1))), axis=0)
        else:
            _Yb = Yb * 0 ## we nix the labels during inference/test-time
            _Xb = Xb
            mod_signal = jnp.ones((Xb.shape[0],1))

        for comp_name in self.cell_components: ## configure dynamic batch-size
            batch_size = Xb.shape[0]
            if adapt_synapses == True:
                batch_size = batch_size * 2
            self.circuit.components[comp_name].batch_size = batch_size

        s0_mu = _Xb * 0
        y_count = 0. # prediction spike train
        self.circuit.reset(do_reset=True)
        self.circuit.components["z3"].outputCompartment = _Yb
        self.circuit.components["z3_prev"].outputCompartment = _Yb
        r1 = 0.
        r2 = 0.
        _S1 = []
        _S2 = []
        _S3 = []
        T = self.T + 1
        for ts in range(1, T):
            self.circuit.clamp_input(x=_Xb)
            self.circuit.clamp_target(y=_Yb)
            self.circuit.clamp_mod_labels(mod_labels=mod_signal)
            self.circuit.runCycle(t=ts*self.dt, dt=self.dt)
            if adapt_synapses == True:
                self.circuit.evolve(t=self.T, dt=self.dt)

            if self.learn_recon == True:
                #_s0_mu = self.circuit.components["zR"].spikes
                _s0_mu = self.circuit.components["tr0"].trace
                s0_mu = _s0_mu + s0_mu

            if collect_spikes == True:
                s1 = self.circuit.components["z1"].spikes #[0:1,:]
                _S1.append(s1)
                s2 = self.circuit.components["z2"].spikes #[0:1,:]
                _S2.append(s2)
                s3 = self.circuit.components["zy"].spikes #[0:1,:]
                _S3.append(s3)
            elif collect_rate_codes == True:
                r1 = self.circuit.components["z1"].spikes + r1
                r2 = self.circuit.components["z2"].spikes + r2
                #r3 = self.circuit.components["z3"].spikes + r3

            y_count = self.circuit.components["zy"].spikes + y_count

        ## estimate total goodness
        s0_mu = s0_mu/T
        ## estimate output distribution
        if lab_estimator == "softmax":
            y_hat = softmax(y_count)
        else:
            y_hat = y_count

        if collect_spikes == True:
            _S1 = jnp.concatenate(_S1,axis=0) ## turn into jnp array of spikes
            _S2 = jnp.concatenate(_S2,axis=0)
            _S3 = jnp.concatenate(_S3,axis=0)
        elif collect_rate_codes == True:
            _S1 = r1/T
            _S2 = r2/T
            _S3 = y_hat

        return y_hat, y_count, _S1, _S2, _S3, s0_mu

    def print_synapse_stats(self, names):
        io_str = ""
        for i in range(len(names)):
            name = names[i]
            _W = self.circuit.components.get(name).weights
            _str = "{}:  min {} | max {} |  mu {} |  norm {}".format(name, jnp.amin(_W),
                                                                     jnp.amax(_W),jnp.mean(_W),
                                                                     jnp.linalg.norm(_W))
            if i > 0:
                io_str = "{}\n{}".format(io_str, _str)
            else:
                io_str = "{}{}".format(io_str, _str)
        return io_str
