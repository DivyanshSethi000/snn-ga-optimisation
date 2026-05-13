# snn.py
# Minimal spiking neural network (Izhikevich-style) focused on producing I_sum and spike times.

from dataclasses import dataclass          # Used to define lightweight data containers
from typing import List, Tuple, Dict, Any  # Type hints for clarity and maintainability
import numpy as np                         # Numerical operations, random numbers, vector math
import pandas as pd                        # Used for spike-time DataFrame utilities

@dataclass
class SNNFirings:
    """Container for simulation outputs and metadata."""
    # Simulation parameters used to generate this run
    ge: float
    gi: float
    Ne: int
    Ni: int
    time: int
    sparsity: float
    thalamic_ex: float
    thalamic_in: float

    # firings_time[t] = list of neuron indices that fired at timestep t
    # Length = time
    firings_time: List[List[int]]
    # firings_neuron[n] = list of timesteps at which neuron n fired
    # Length = Ne + Ni
    firings_neuron: List[List[int]]

    # Summed synaptic input current at each timestep (scalar)
    I_sum: np.ndarray
    # Rough floating-point operation count (for diagnostics)
    flops: int

    def _firings_to_spikes_at_time(self) -> pd.DataFrame:
        """Convert firings_time into a DataFrame with columns: time, num_fired, excitatory, inhibitory"""
        rows = []
        for t, fired in enumerate(self.firings_time):
            num = len(fired)                           # total spikes at time t
            ex = sum(1 for n in fired if n < self.Ne)  # excitatory spikes
            inh = num - ex                             # inhibitory spikes
            rows.append({
                "time": int(t),
                "num_fired": int(num),
                "excitatory": int(ex),
                "inhibitory": int(inh)
            })
        return pd.DataFrame(rows)

    def _firings_time_bin(self, interval: int) -> pd.DataFrame:
        """Aggregate spikes into bins of width `interval` (in time steps)."""
        df = self._firings_to_spikes_at_time()
        df["bin"] = df["time"] // int(interval)
        bins = []
        for b, group in df.groupby("bin"):
            bins.append({
                "time": int(group["time"].min()), # start time of bin
                "num_fired": float(group["num_fired"].mean()),
                "excitatory": float(group["excitatory"].mean()),
                "inhibitory": float(group["inhibitory"].mean())
            })
        return pd.DataFrame(bins)

    def _flatten_firings(self) -> List[Tuple[int, int]]:
        """Return list of (time, neuron) pairs for all spikes."""
        return [(int(t), int(n)) for t, fired in enumerate(self.firings_time) for n in fired]


def _snnfirings_score(self) -> Dict[str, Any]:
    """
    Compute firing rates (Hz per neuron) for excitatory and inhibitory populations.
    Uses the last 80% of the simulation (discard first 20% as transient).
    """
    spikes_df = self._firings_to_spikes_at_time()
    cutoff = 0.2
    start_index = int(len(spikes_df) * cutoff)
    trimmed = spikes_df.iloc[start_index:]
    # Total spikes in trimmed window
    total_ex = trimmed["excitatory"].sum() if len(trimmed) > 0 else 0.0
    total_in = trimmed["inhibitory"].sum() if len(trimmed) > 0 else 0.0
    dt = 1.0 / 1000.0  # timestep = 1 ms
    T_sec = len(trimmed) * dt if len(trimmed) > 0 else 1.0
    # Convert spike counts → Hz per neuron
    ex_hz = float(total_ex) / (self.Ne * T_sec) if (self.Ne > 0 and T_sec > 0) else 0.0
    in_hz = float(total_in) / (self.Ni * T_sec) if (self.Ni > 0 and T_sec > 0) else 0.0
    # Average firing rate across all neurons
    average_hz = (float(total_ex) + float(total_in)) / ((self.Ne +self.Ni) * T_sec) if ((self.Ne + self.Ni) > 0 and T_sec > 0) else 0.0
    return {"ex_firing": ex_hz, "in_firing": in_hz, "average_firing": average_hz}
# Attach method to class
setattr(SNNFirings, "score", _snnfirings_score)


@dataclass
class SNN:
    """
    Simple Izhikevich-style network wrapper.

    Parameters:
    ----------
    ge, gi : float
        Scaling factors for excitatory and inhibitory synaptic weights.
        Higher ge → stronger excitatory drive.
        Higher gi → stronger inhibitory suppression.

    Ne, Ni : int
        Number of excitatory and inhibitory neurons.
        Default: 800 EX, 200 IN (classic 80/20 cortical ratio).

    sparsity : float
        Connection probability (1.0 = fully connected).
        Lower sparsity → more random disconnections.

    time : int
        Number of simulation timesteps.
        With dt = 1 ms, time=1000 → 1 second of simulated activity.

    thalamic_ex, thalamic_in : float
        Amplitude of Gaussian noise injected into EX and IN neurons.
        Acts as background drive / thalamic input.
    """
    ge: float = 0.5
    gi: float = 1.0
    Ne: int = 5
    Ni: int = 2
    sparsity: float = 1.0
    time: int = 300000
    thalamic_ex: float = 2.7
    thalamic_in: float = 0.9

    def _run_network(
        self,
        ge: float,
        gi: float,
        Ne: int,
        Ni: int,
        sparsity: float,
        time: int,
        thalamic_ex: float,
        thalamic_in: float,
        seed: int = None
    ) -> Tuple[List[List[int]], List[List[int]], np.ndarray, int]:
        """
        Core simulation loop for the Izhikevich-style spiking network.

        Returns:
            firings_time  : list of neurons that fired at each timestep
            firings_neuron: list of spike times for each neuron
            I_sum         : summed synaptic input current per timestep
            flops         : rough floating-point operation count
        """
        # RANDOM SEEDING
        # Ensures reproducibility when running multiple trials.
        if seed is not None:
            np.random.seed(int(seed))

        # INITIAL HETEROGENEITY
        # re, ri introduce neuron-to-neuron variability
        re = np.random.rand(Ne) # excitatory heterogeneity
        ri = np.random.rand(Ni) # inhibitory heterogeneity

        # IZHIKEVICH PARAMETERS
        # a, b, c, d define the neuron dynamics.
        # EX neurons use fixed parameters.
        # IN neurons vary based on ri.
        a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
        b = np.concatenate((0.2 * np.ones(Ne), 0.25 - 0.05 * ri))
        c = np.concatenate((-65 + 15 * re**2, -65 * np.ones(Ni)))
        d = np.concatenate((8 - 6 * re**2, 2 * np.ones(Ni)))

        # SYNAPTIC WEIGHT MATRIX (columns are presynaptic neurons)
        # W_ex : excitatory → all neurons (positive weights)
        # W_in : inhibitory → all neurons (negative weights)
        # S    : full synaptic matrix (postsynaptic rows × presynaptic columns)
        W_ex = ge * np.random.rand(Ne + Ni, Ne)
        W_in = -gi * np.random.rand(Ne + Ni, Ni)
        S = np.concatenate((W_ex, W_in), axis=1)

        # APPLY SPARSITY
        # Randomly zero out connections to achieve desired sparsity
        if sparsity < 1.0:
            mask = np.random.binomial(1, sparsity, size=S.shape)
            S = S * mask

        # STATE VARIABLES
        # v : membrane potential
        # u : recovery variable
        v = -65.0 * np.ones(Ne + Ni)
        u = b * v
        # firings_time[t] = list of neurons that fired at time t
        firings_time: List[List[int]] = []
        I = np.zeros(Ne + Ni) # I : synaptic + thalamic input vector
        I_sum = np.zeros(time) # I_sum[t] = total synaptic input at time t (scalar)
        flops = 0 # flops : rough operation counter for diagnostics


        # MAIN SIMULATION LOOP
        # This loop runs for `time` timesteps (default = 1000 → 1 second).
        # At each timestep:
        #   1. Add thalamic (noise) input
        #   2. Detect spikes
        #   3. Reset spiking neurons (Izhikevich reset)
        #   4. Propagate synaptic input from fired neurons
        #   5. Record summed input current (I_sum)
        #   6. Integrate membrane potential (two half-steps)
        #   7. Update recovery variable u
        #   8. Count FLOPS for diagnostics
        for t in range(time):
            # THALAMIC INPUT — Gaussian noise per population
            # Excitatory neurons get thalamic_ex * N(0,1)
            # Inhibitory neurons get thalamic_in * N(0,1)
            I_ex = thalamic_ex * np.random.randn(Ne)
            I_in = thalamic_in * np.random.randn(Ni)
            I = np.concatenate((I_ex, I_in)) # Combine into full input vector

            # SPIKE DETECTION
            # Neurons fire when v >= 30 mV (Izhikevich threshold)
            fired_idx = np.where(v >= 30)[0]
            fired_list = fired_idx.tolist()
            firings_time.append(fired_list) # Store list of fired neurons for this timestep
            k = len(fired_list) # Number of spikes this timestep
            flops += (Ne + Ni) * (1 + k) # FLOPS: spike detection + bookkeeping
            # RESET FIRED NEURONS (Izhikevich reset rule)
            if k > 0:
                
                v[fired_idx] = c[fired_idx] # Reset membrane potential to c
                u[fired_idx] = u[fired_idx] + d[fired_idx] # Increase recovery variable u by d
                # SYNAPTIC PROPAGATION
                # Add synaptic input from all fired presynaptic neurons.
                # S[:, fired_idx] selects columns corresponding to fired neurons.
                I += np.sum(S[:, fired_idx], axis=1)

            # RECORD SUMMED INPUT CURRENT
            # I_sum[t] is a scalar: total synaptic + thalamic input.
            I_sum[t] = float(np.sum(I))

            # INTEGRATE MEMBRANE POTENTIAL
            # Izhikevich uses:
            #   dv/dt = 0.04v^2 + 5v + 140 - u + I
            #
            # Two half-steps improve numerical stability.
            dv = 0.04 * v**2 + 5 * v + 140 - u + I
            v = v + 0.5 * dv # first half-step
            v = v + 0.5 * dv # second half-step
            flops += (Ne + Ni) * 22 # FLOPS: membrane integration

            # UPDATE RECOVERY VARIABLE u
            u = u + a * (b * v - u)
            flops += (Ne + Ni) * 4 # FLOPS: recovery update

        # Build per-neuron spike-time lists
        total_neurons = Ne + Ni
        firings_neuron: List[List[int]] = [[] for _ in range(total_neurons)]
        for t, fired in enumerate(firings_time):
            for neuron in fired:
                firings_neuron[int(neuron)].append(int(t))

        return firings_time, firings_neuron, I_sum, int(flops)

    def run_network(self, seed: int = None) -> SNNFirings:
        """Public entry: runs the network with the instance parameters and returns SNNFirings."""
        firings_time, firings_neuron, I_sum, flops = self._run_network(
            self.ge, self.gi, self.Ne, self.Ni,
            self.sparsity, self.time,
            self.thalamic_ex, self.thalamic_in,
            seed=seed
        )
        return SNNFirings(
            ge=self.ge, gi=self.gi, Ne=self.Ne, Ni=self.Ni,
            time=self.time, sparsity=self.sparsity,
            thalamic_ex=self.thalamic_ex, thalamic_in=self.thalamic_in,
            firings_time=firings_time,
            firings_neuron=firings_neuron,
            I_sum=np.asarray(I_sum),
            flops=int(flops)
        )

    def score(self) -> Dict[str, Any]:
        """Not used: scoring is provided by SNNFirings.score()."""
        raise RuntimeError("Call score() on the returned SNNFirings object, not on SNN.")


if __name__ == '__main__':
    model = SNN()
    res = model.run_network(seed=42)
    print("DEBUG: firings_time length:", len(res.firings_time))
    print("DEBUG: firings_neuron length:", len(res.firings_neuron))
    print("DEBUG: I_sum length:", len(res.I_sum))
    print("DEBUG: ex/in firing (Hz per neuron):", res.score())