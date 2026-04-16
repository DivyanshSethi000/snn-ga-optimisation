# snn.py
# Minimal spiking neural network (Izhikevich-style) focused on producing I_sum and spike times.

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class SNNFirings:
    """Container for simulation outputs and metadata."""
    ge: float
    gi: float
    Ne: int
    Ni: int
    time: int
    sparsity: float
    thalamic_ex: float
    thalamic_in: float

    # per-time-step list of fired neuron indices (length == time)
    firings_time: List[List[int]]
    # per-neuron list of spike times (length == Ne + Ni)
    firings_neuron: List[List[int]]

    # summed input current time series (length == time)
    I_sum: np.ndarray
    # rough floating-point operation count estimate
    flops: int

    def _firings_to_spikes_at_time(self) -> pd.DataFrame:
        """Convert firings_time into a DataFrame with columns: time, num_fired, excitatory, inhibitory"""
        rows = []
        for t, fired in enumerate(self.firings_time):
            num = len(fired)
            ex = sum(1 for n in fired if n < self.Ne)
            inh = num - ex
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
                "time": int(group["time"].min()),
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
    total_ex = trimmed["excitatory"].sum() if len(trimmed) > 0 else 0.0
    total_in = trimmed["inhibitory"].sum() if len(trimmed) > 0 else 0.0
    dt = 1.0 / 1000.0  # seconds per time step
    T_sec = len(trimmed) * dt if len(trimmed) > 0 else 1.0
    ex_hz = float(total_ex) / (self.Ne * T_sec) if (self.Ne > 0 and T_sec > 0) else 0.0
    in_hz = float(total_in) / (self.Ni * T_sec) if (self.Ni > 0 and T_sec > 0) else 0.0
    return {"ex_firing": ex_hz, "in_firing": in_hz, "total_firing": ex_hz + in_hz}

setattr(SNNFirings, "score", _snnfirings_score)


@dataclass
class SNN:
    """
    Simple Izhikevich-style network wrapper.
    - ge, gi: excitatory/inhibitory synaptic scaling
    - Ne, Ni: number of excitatory and inhibitory neurons
    - sparsity: connection probability (1.0 = fully connected)
    - time: number of time steps to simulate (default 1000 -> 1 second at dt=1ms)
    - thalamic_ex/in: noise amplitude for each population
    """
    ge: float = 0.5
    gi: float = 1.0
    Ne: int = 800
    Ni: int = 200
    sparsity: float = 1.0
    time: int = 1000
    thalamic_ex: float = 5.0
    thalamic_in: float = 2.0

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
        """Core simulation loop. Returns firings_time, firings_neuron, I_sum, flops."""
        if seed is not None:
            np.random.seed(int(seed))

        # Random initial heterogeneity
        re = np.random.rand(Ne)
        ri = np.random.rand(Ni)

        # Izhikevich parameters per neuron
        a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
        b = np.concatenate((0.2 * np.ones(Ne), 0.25 - 0.05 * ri))
        c = np.concatenate((-65 + 15 * re**2, -65 * np.ones(Ni)))
        d = np.concatenate((8 - 6 * re**2, 2 * np.ones(Ni)))

        # Synaptic weight matrix: columns are presynaptic neurons
        W_ex = ge * np.random.rand(Ne + Ni, Ne)
        W_in = -gi * np.random.rand(Ne + Ni, Ni)
        S = np.concatenate((W_ex, W_in), axis=1)

        # Apply sparsity by zeroing random connections
        if sparsity < 1.0:
            mask = np.random.binomial(1, sparsity, size=S.shape)
            S = S * mask

        # State variables
        v = -65.0 * np.ones(Ne + Ni)
        u = b * v

        firings_time: List[List[int]] = []
        I = np.zeros(Ne + Ni)
        I_sum = np.zeros(time)
        flops = 0

        for t in range(time):
            # Thalamic input (Gaussian noise per population)
            I_ex = thalamic_ex * np.random.randn(Ne)
            I_in = thalamic_in * np.random.randn(Ni)
            I = np.concatenate((I_ex, I_in))

            # Find fired neurons
            fired_idx = np.where(v >= 30)[0]
            fired_list = fired_idx.tolist()
            firings_time.append(fired_list)
            k = len(fired_list)
            flops += (Ne + Ni) * (1 + k)

            if k > 0:
                # reset fired neurons
                v[fired_idx] = c[fired_idx]
                u[fired_idx] = u[fired_idx] + d[fired_idx]
                # add synaptic input from fired neurons (sum columns for fired presyn)
                I += np.sum(S[:, fired_idx], axis=1)

            # record summed input current for diagnostics (scalar per time step)
            I_sum[t] = float(np.sum(I))

            # Integrate membrane potential (two half-steps for stability)
            dv = 0.04 * v**2 + 5 * v + 140 - u + I
            v = v + 0.5 * dv
            v = v + 0.5 * dv
            flops += (Ne + Ni) * 22

            # Update recovery variable
            u = u + a * (b * v - u)
            flops += (Ne + Ni) * 4

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