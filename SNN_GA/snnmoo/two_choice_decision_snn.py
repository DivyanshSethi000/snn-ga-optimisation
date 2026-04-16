import numpy as np


class TwoChoiceDecisionSNN:

    def __init__(
        self,
        seed=1,
        wEE_A=0.010,
        wEE_N=0.040,
        wEI_A=0.010,
        wEI_N=0.020,
        wIE_G=0.020,
        wII_G=0.010,
        I_stim=0.0,
        bias=0.0,
    ):
        self.seed = seed

        # Excitatory -> Excitatory synaptic strengths.
        # GA optimizes these once and uses them for both E1 and E2 within-population connections.
        self.wEE_A = wEE_A
        self.wEE_N = wEE_N

        self.wEE_A_cross = 0.0
        self.wEE_N_cross = 0.0

        # E->I and I->E / I->I synaptic strengths.
        self.wEI_A = wEI_A
        self.wEI_N = wEI_N
        self.wIE_G = wIE_G
        self.wII_G = wII_G

        # Stimulus parameters.
        self.I_stim = I_stim
        self.bias = bias

        # Population sizes.
        self.NE1 = 200
        self.NE2 = 200
        self.NI = 100
        self.N = self.NE1 + self.NE2 + self.NI  # total neurons

        # Index ranges for each population (used for slicing).
        self.idxE1 = np.arange(0, self.NE1)
        self.idxE2 = np.arange(self.NE1, self.NE1 + self.NE2)
        self.idxI  = np.arange(self.NE1 + self.NE2, self.N)


    def run(self):
        # Fix random seed for reproducibility of connectivity and noise.
        np.random.seed(self.seed)

        # Simulation parameters.
        T = 2500  # total time in ms
        dt = 1.0  # time step in ms
        steps = int(T / dt)
        time = np.arange(0, T, dt)

        # Izhikevich neuron parameters for each neuron.
        # We use Regular Spiking (RS) for excitatory and Fast Spiking (FS) for inhibitory.
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        d = np.zeros(self.N)

        # Excitatory RS parameters for E1.
        a[self.idxE1] = 0.02
        b[self.idxE1] = 0.2
        c[self.idxE1] = -65
        d[self.idxE1] = 8

        # Excitatory RS parameters for E2.
        a[self.idxE2] = 0.02
        b[self.idxE2] = 0.2
        c[self.idxE2] = -65
        d[self.idxE2] = 8

        # Inhibitory FS parameters for I.
        a[self.idxI] = 0.1
        b[self.idxI] = 0.2
        c[self.idxI] = -65
        d[self.idxI] = 2

        # Initial membrane potentials and recovery variables.
        v = -65 * np.ones(self.N)  # membrane potential (mV)
        u = b * v                  # recovery variable

        # Synaptic conductances for AMPA, NMDA, and GABA(A).
        gA = np.zeros(self.N)  # AMPA conductance
        gN = np.zeros(self.N)  # NMDA conductance
        gG = np.zeros(self.N)  # GABA conductance

        # Synaptic time constants (ms).
        tauA = 5    # AMPA decay
        tauN = 90   # NMDA decay
        tauG = 10   # GABA decay

        # Precompute decay factors for exponential decay per time step.
        decA = 1 - dt / tauA
        decN = 1 - dt / tauN
        decG = 1 - dt / tauG

        # NMDA Mg2+ block parameters.
        # B(v) is a voltage-dependent factor that reduces NMDA conductance at hyperpolarized potentials.
        eta = 0.28
        gamma = 0.062
        B = lambda vv: 1.0 / (1.0 + eta * np.exp(-gamma * vv))

        pEE_within = 0.10
        pEE_cross  = 0.0
        pEI = 0.20
        pIE = 0.40
        pII = 0.20

        # Weight matrices:
        # W_A: AMPA weights
        # W_N: NMDA weights
        # W_G: GABA weights
        # Shape: (post_neuron, pre_neuron)
        W_AMPA = np.zeros((self.N, self.N))
        W_NMDA = np.zeros((self.N, self.N))
        W_GABA = np.zeros((self.N, self.N))
        
        # E1 -> E1
        A = (np.random.rand(self.NE1, self.NE1) < pEE_within).astype(float)
        np.fill_diagonal(A, 0.0)
        W_AMPA[np.ix_(self.idxE1, self.idxE1)] = self.wEE_A * A
        W_NMDA[np.ix_(self.idxE1, self.idxE1)] = self.wEE_N * A

        # E2 -> E2
        A = (np.random.rand(self.NE2, self.NE2) < pEE_within).astype(float)
        np.fill_diagonal(A, 0.0)
        W_AMPA[np.ix_(self.idxE2, self.idxE2)] = self.wEE_A * A
        W_NMDA[np.ix_(self.idxE2, self.idxE2)] = self.wEE_N * A

        # E -> I
        A = (np.random.rand(self.NI, self.NE1 + self.NE2) < pEI).astype(float)
        W_AMPA[np.ix_(self.idxI, np.concatenate((self.idxE1, self.idxE2)))] = self.wEI_A * A
        W_NMDA[np.ix_(self.idxI, np.concatenate((self.idxE1, self.idxE2)))] = self.wEI_N * A

        # I -> E
        A = (np.random.rand(self.NE1 + self.NE2, self.NI) < pIE).astype(float)
        W_GABA[np.ix_(np.concatenate((self.idxE1, self.idxE2)), self.idxI)] = self.wIE_G * A

        # I -> I
        A = (np.random.rand(self.NI, self.NI) < pII).astype(float)
        np.fill_diagonal(A, 0.0)
        W_GABA[np.ix_(self.idxI, self.idxI)] = self.wII_G * A

        # ----------------------------------------------------
        # Recording structures
        # ----------------------------------------------------
        # spikeTimes: list of spike times for each neuron.
        spikeTimes = [[] for _ in range(self.N)]

        # spikeCountE*: number of spikes per ms in each population.
        spikeCountE1 = np.zeros(steps)
        spikeCountE2 = np.zeros(steps)
        spikeCountI  = np.zeros(steps)

        # I_sum: total synaptic current across all neurons at each time step.
        I_sum = np.zeros(steps)

        # Binary spike rasters for plotting
        spikeRasterE1 = np.zeros((self.NE1, steps), dtype=bool)
        spikeRasterE2 = np.zeros((self.NE2, steps), dtype=bool)
        spikeRasterI  = np.zeros((self.NI,  steps), dtype=bool)

        # ----------------------------------------------------
        # External drive parameters (background noise)
        # ----------------------------------------------------
        # I0E, I0I: baseline mean input to E and I.
        # sigmaE, sigmaI: standard deviation of Gaussian noise per time step.
        I0E = 1.0
        I0I = 1.0
        sigmaE = 2.5
        sigmaI = 2.5

        # Evidence (stimulus) window in ms.
        evidenceStart = 500
        evidenceEnd   = 1500

        # ----------------------------------------------------
        # MAIN SIMULATION LOOP
        # ----------------------------------------------------
        for t in range(steps):
            tm = time[t]

            # -----------------------------
            # External current (background)
            # -----------------------------
            # Each population receives Gaussian noise around its baseline.
            Iext = np.zeros(self.N)
            Iext[self.idxE1] = I0E + sigmaE * np.random.randn(self.NE1)
            Iext[self.idxE2] = I0E + sigmaE * np.random.randn(self.NE2)
            Iext[self.idxI]  = I0I + sigmaI * np.random.randn(self.NI)

            # -----------------------------
            # Stimulus drive (evidence)
            # -----------------------------
            # During the evidence window, both E1 and E2 receive additional
            # stimulus current. The bias parameter can tilt the stimulus
            # slightly towards one pool or the other.
            if evidenceStart <= tm <= evidenceEnd:
                Iext[self.idxE1] += self.I_stim * (1 + self.bias)
                Iext[self.idxE2] += self.I_stim * (1 - self.bias)

            # -----------------------------
            # Synaptic currents
            # -----------------------------
            # AMPA and NMDA are excitatory (reversal ~0 mV),
            # GABA is inhibitory (reversal ~ -70 mV).
            IAMP = gA * (0 - v)
            INMD = gN * B(v) * (0 - v)
            IGAB = gG * (-70 - v)

            # Total synaptic current per neuron.
            Isyn = IAMP + INMD + IGAB

            # Record total synaptic current across the network.
            I_sum[t] = np.sum(Isyn)

            # -----------------------------
            # Izhikevich neuron update
            # -----------------------------
            # dv/dt = 0.04 v^2 + 5 v + 140 - u + I
            # du/dt = a (b v - u)
            I = Iext + Isyn
            v = v + dt * (0.04 * v * v + 5 * v + 140 - u + I)
            u = u + dt * (a * (b * v - u))

            # Detect spikes (threshold at +30 mV).
            spiked = (v >= 30)
            if np.any(spiked):
                idx = np.where(spiked)[0]
                for n in idx:
                    spikeTimes[n].append(tm)

                    # Fill raster matrices
                    if n < self.NE1:
                        spikeRasterE1[n, t] = 1
                    elif n < self.NE1 + self.NE2:
                        spikeRasterE2[n - self.NE1, t] = 1
                    else:
                        spikeRasterI[n - (self.NE1 + self.NE2), t] = 1

            # Reset spiking neurons.
            v[spiked] = c[spiked]
            u[spiked] = u[spiked] + d[spiked]

            # -----------------------------
            # Synaptic conductance updates
            # -----------------------------
            # First, exponential decay of existing conductances.
            gA *= decA
            gN *= decN
            gG *= decG

            # Then, add increments from spikes at this time step.
            if np.any(spiked):
                # spk is a binary vector indicating which neurons spiked.
                spk = spiked.astype(float)
                # Post-synaptic conductances are updated by W @ spk.
                gA += W_AMPA @ spk
                gN += W_NMDA @ spk
                gG += W_GABA @ spk

            # -----------------------------
            # Population spike counts
            # -----------------------------
            # Count how many neurons in each population spiked at this time step.
            spikeCountE1[t] = np.sum(spiked[self.idxE1])
            spikeCountE2[t] = np.sum(spiked[self.idxE2])
            spikeCountI[t]  = np.sum(spiked[self.idxI])

        # ----------------------------------------------------
        # PSD COMPUTATION (500–1500 ms window)
        # ----------------------------------------------------
        # We compute the PSD of I_sum in the decision window using:
        # - rFFT (real FFT)
        # - Hanning window
        # - Same normalization as used in the GA.
        dt_sec = dt / 1000.0  # convert ms to seconds
        start = int(500 / dt)
        end   = int(1500 / dt)

        seg = I_sum[start:end]
        seg = seg - np.mean(seg)  # remove DC component
        N = len(seg)

        if N > 1:
            # Apply Hanning window to reduce spectral leakage.
            window = np.hanning(N)
            Xw = seg * window

            # rFFT: frequency-domain representation.
            xf = np.fft.rfft(Xw)

            # PSD normalization (matches GA code):
            # Sxx has units of power per Hz.
            Sxx = np.real((2 * dt_sec**2 / (N * dt_sec)) * (xf * np.conj(xf)))

            # Frequency axis.
            faxis = np.fft.rfftfreq(N, d=dt_sec)

            # Peak frequency in the decision window.
            psd_peak = float(faxis[np.argmax(Sxx)])
        else:
            # Degenerate case (should not happen with normal settings).
            psd_peak = 0.0
            faxis = np.array([0.0])
            Sxx = np.array([0.0])

        # Return all relevant outputs in a dictionary.
        return {
            "time": time,
            "spikeCountE1": spikeCountE1,
            "spikeCountE2": spikeCountE2,
            "spikeCountI": spikeCountI,
            "spikeTimes": spikeTimes,
            "spikeRasterE1": spikeRasterE1,
            "spikeRasterE2": spikeRasterE2,
            "spikeRasterI":  spikeRasterI,
            "I_sum": I_sum,
            "psd_peak": psd_peak,
            "psd_faxis": faxis,
            "psd_power": Sxx,
        }