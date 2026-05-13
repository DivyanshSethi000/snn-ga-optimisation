# Full NSGA‑III optimisation pipeline for a two‑choice decision
# spiking neural network (SNN). This script:
#
#   • Defines a 7‑parameter decision‑making SNN model
#   • Optimises 3 objectives:
#         1. Pre‑stimulus unbiasedness (E1 vs E2)
#         2. Decision‑period separation (winner–loser)
#         3. PSD peak frequency accuracy (target ≈ 30 Hz)
#
#   • Applies hard biological constraints:
#         - No runaway inhibition
#         - No silent networks
#         - No biased pre‑stimulus activity
#
#   • Includes:
#         - NSGA‑III optimisation
#         - Full Pareto frontier extraction
#         - Best‑individual selection
#         - 4‑case classification (E1‑win, E1‑lose, E2‑win, E2‑lose)
#         - Raster plots, firing‑rate plots, PSD plots
#         - Winner distribution histogram
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
# pymoo: multi‑objective evolutionary algorithms
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
# Two‑choice decision SNN model
from snnmoo.two_choice_decision_snn import TwoChoiceDecisionSNN

import argparse
# Command‑line argument: enable optional NMDA ratio constraint
parser = argparse.ArgumentParser()
parser.add_argument("--ratio", action="store_true", help="Enable the NMDA ratio constraint")
args = parser.parse_args()

print(">>> USING NSGA-III DECISION GA (3-objective unbiasedness/separation/PSD, RMSE-normalised) <<<")


def ensure_dir(path: str) -> None:
    # Create directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Number of trials per candidate during GA evaluation
TRIALS_PER_CANDIDATE = 1

# Targets for the 3 RMSE‑style objectives
PRE_DIFF_TARGET = 0.0      # unbiasedness target (E1 ≈ E2 pre‑stim)
SEP_TARGET      = 4.0      # desired winner–loser separation (Hz)
PSD_TARGET      = 30.0     # desired oscillation frequency (Hz)

# Normalisation scales (convert raw errors to dimensionless)
PRE_DIFF_SCALE = 0.2       # acceptable pre-bias scale
SEP_SCALE      = SEP_TARGET
PSD_SCALE      = PSD_TARGET

# GA PROBLEM DEFINITION
# The DecisionProblem class defines:
#   • The dimensionality of the search space (7 parameters)
#   • The 3 objective functions (normalized RMSEs)
#   • Hard biological constraints (reject invalid networks)
#   • The evaluation logic used by NSGA‑III
#
# This class is the core of the decision‑making optimisation.
class DecisionProblem(Problem):

    def __init__(self, apply_ratio_constraint=False):
        """
        Initialise the NSGA‑III optimisation problem.

        Parameters:
            apply_ratio_constraint : bool
                If True, applies an additional constraint on the
                NMDA/AMPA ratio (used in some biological models).
        """
        # Define the search space:
        #   7 parameters controlling recurrent connectivity:
        #       wEE_A  — AMPA E→E
        #       wEE_N  — NMDA E→E
        #       wEI_A  — AMPA E→I
        #       wEI_N  — NMDA E→I
        #       wIE_G  — GABA I→E
        #       wII_G  — GABA I→I
        #       I_stim — stimulus input strength
        #
        # Bounds chosen from biologically plausible ranges.
        super().__init__(
            n_var=7,
            n_obj=3,  # norm RMSE: unbiasedness, separation, PSD
            xl=np.array([0.008, 0.03, 0.006, 0.008, 0.010, 0.005, 0.4]),
            xu=np.array([0.03,  0.10, 0.03,  0.04,  0.035, 0.02,  0.8]),
        )
        self.apply_ratio_constraint = apply_ratio_constraint
        self.PENALTY = 1e6 # large penalty for invalid solutions

    def sanity_check_params(self, wEE_A, wEE_N, wEI_A, wEI_N, wIE_G, wII_G, I_stim):
        """
        Quick validation of parameter bounds and non-negativity.
        Rejects obviously invalid parameter sets before simulation.
        """

        # All synaptic weights must be positive
        if wEE_A <= 0 or wEE_N <= 0:
            return False
        if wEI_A <= 0 or wEI_N <= 0:
            return False
        if wIE_G <= 0 or wII_G <= 0:
            return False
        # Stimulus input must be within a reasonable range
        if not (0.2 <= I_stim <= 0.8):
            return False
        return True

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate a batch of candidate solutions X.

        For each candidate:
            1. Validate parameters
            2. Run TRIALS_PER_CANDIDATE simulations
            3. Compute:
                - Pre‑stimulus unbiasedness (E1 vs E2)
                - Decision‑period separation (winner–loser)
                - PSD peak frequency
            4. Apply hard biological constraints
            5. Compute normalized RMSE objectives
            6. Store metadata for analysis
        """
        F = []              # objective vectors
        meta_list_all = []  # metadata for each candidate

        dt_sec = 1e-3
        NE1, NE2, NI = 200, 200, 100   # population sizes

        for row in X:
            # Unpack parameters
            wEE_A, wEE_N, wEI_A, wEI_N, wIE_G, wII_G, I_stim = row

            # Sanity check: reject invalid parameter sets
            if not self.sanity_check_params(wEE_A, wEE_N, wEI_A, wEI_N, wIE_G, wII_G, I_stim):
                # Penalize invalid parameter vectors with very large objective values
                F.append([self.PENALTY, self.PENALTY, self.PENALTY])
                meta_list_all.append({
                    "pre_diff_raw": np.nan,
                    "sep_raw": np.nan,
                    "psd_raw": np.nan,
                    "pre_diff_norm": np.nan,
                    "sep_norm": np.nan,
                    "psd_norm": np.nan,
                    "wEE_A": wEE_A,
                    "wEE_N": wEE_N,
                    "wEI_A": wEI_A,
                    "wEI_N": wEI_N,
                    "wIE_G": wIE_G,
                    "wII_G": wII_G,
                    "I_stim": I_stim,
                })
                continue
            # Storage for trial‑averaged metrics        
            frE1_pre_trials = []
            frE2_pre_trials = []
            sep_trials = []
            psd_trials = []
            frI_dec_trials = []
            frI_pre_trials = []
            frE1_dec_trials = []
            frE2_dec_trials = []

            # Ratio check quantity used for optional constraint/Optional NMDA/AMPA ratio constraint
            ratio = (wEI_N * wIE_G) / (wEE_N + 1e-9)
            ratio_error = abs(ratio - 0.01)

            for trial in range(TRIALS_PER_CANDIDATE):

                model = TwoChoiceDecisionSNN(
                    seed=trial,
                    wEE_A=wEE_A,
                    wEE_N=wEE_N,
                    wEI_A=wEI_A,
                    wEI_N=wEI_N,
                    wIE_G=wIE_G,
                    wII_G=wII_G,
                    I_stim=I_stim,
                    bias=0.0,
                )
                out_dict = model.run()

                time    = out_dict["time"]
                spikeE1 = out_dict["spikeCountE1"]
                spikeE2 = out_dict["spikeCountE2"]
                spikeI  = out_dict["spikeCountI"]
                # Pre‑stimulus window
                # Epoch 1: 200-500 ms (pre-stimulus period)
                mask_pre = (time >= 200.0) & (time <= 500.0)
                T_pre = np.sum(mask_pre) * dt_sec

                frE1_pre = np.sum(spikeE1[mask_pre]) / (NE1 * T_pre)
                frE2_pre = np.sum(spikeE2[mask_pre]) / (NE2 * T_pre)
                frI_pre  = np.sum(spikeI[mask_pre])  / (NI  * T_pre) 

                frE1_pre_trials.append(frE1_pre)
                frE2_pre_trials.append(frE2_pre)
                frI_pre_trials.append(frI_pre) 

                # Epoch 2: 500-1500 ms (decision period)
                mask_dec = (time >= 500.0) & (time <= 1500.0)
                T_dec = np.sum(mask_dec) * dt_sec

                frE1_dec = np.sum(spikeE1[mask_dec]) / (NE1 * T_dec)
                frE2_dec = np.sum(spikeE2[mask_dec]) / (NE2 * T_dec)
                frI_dec  = np.sum(spikeI[mask_dec])  / (NI  * T_dec)
                
                frE1_dec_trials.append(frE1_dec)
                frE2_dec_trials.append(frE2_dec)
                
                sep_trials.append(abs(frE1_dec - frE2_dec)) # Winner–loser separation
                psd_trials.append(out_dict["psd_peak"]) # PSD peak frequency (computed inside model)
                frI_dec_trials.append(frI_dec)

            # Averages across trials
            pre_diff = abs(np.mean(frE1_pre_trials) - np.mean(frE2_pre_trials))
            sep_mean = np.mean(sep_trials)
            psd_mean = np.mean(psd_trials)
            
            frI_dec_mean = float(np.mean(frI_dec_trials))
            frI_pre_mean = float(np.mean(frI_pre_trials))
            
            frE1_dec_mean = float(np.mean(frE1_dec_trials))
            frE2_dec_mean = float(np.mean(frE2_dec_trials))
            

            # HARD BIOLOGICAL CONSTRAINTS
            # These constraints ensure that the GA does not waste time exploring parameter regimes that cannot support
            # decision‑making behaviour. If any condition fails, the candidate receives a huge penalty (1e6).

            # 1. Joint unbiasedness + minimum separation gate
            # Compute per‑trial winner–loser separations
            separations = []
            for e1, e2 in zip(frE1_dec_trials, frE2_dec_trials):
                if e1 > e2:
                    separations.append(e1 - e2)
                elif e2 > e1:
                    separations.append(e2 - e1)
            
            middle_sep = np.mean(separations) if len(separations) > 0 else 0.0
            
            # Condition:
            #   - Pre‑stimulus unbiasedness: pre_diff < 0.3 Hz
            #   - Minimum decision separation: middle_sep > 1 Hz
            if not (pre_diff < 0.3 and middle_sep > 1.0):
                F.append([self.PENALTY, self.PENALTY, self.PENALTY])
                meta_list_all.append({
                    "pre_diff_raw": pre_diff,
                    "sep_raw": sep_mean,
                    "psd_raw": psd_mean,
                    "reason": "joint pre_diff/middle_sep failed"
                })
                continue
            
            # 2. Reject networks with no decision activity
            # If both excitatory pools fire < 0.5 Hz during the decision window, the network is effectively silent.
            if frE1_dec_mean < 0.5 and frE2_dec_mean < 0.5:
                F.append([self.PENALTY, self.PENALTY, self.PENALTY])
                meta_list_all.append({
                    "pre_diff_raw": pre_diff,
                    "sep_raw": sep_mean,
                    "psd_raw": psd_mean,
                    "reason": "both E pools nearly silent"
                })
                continue

            
            # 3. Reject runaway inhibition
            # If inhibitory firing exceeds 25 Hz during the decision window, the network collapses into an inhibition‑dominated regime.
            if frI_dec_mean > 25.0:
                F.append([self.PENALTY, self.PENALTY, self.PENALTY])
                meta_list_all.append({
                    "pre_diff_raw": pre_diff,
                    "sep_raw": sep_mean,
                    "psd_raw": psd_mean,
                    "frI_dec_mean": frI_dec_mean,
                    "reason": "runaway inhibition"
                })
                continue

            # RAW OBJECTIVE ERRORS
            # Targets:
            #   pre_diff_raw → 0 (unbiased)
            #   sep_raw      → 0 (achieved if sep_mean ≥ SEP_TARGET)
            #   psd_raw      → 0 (achieved if psd_mean ≈ PSD_TARGET)
            pre_diff_raw = pre_diff                      
            sep_raw      = max(0.0, SEP_TARGET - sep_mean)
            psd_raw      = abs(psd_mean - PSD_TARGET)  

            # NORMALISED RMSE OBJECTIVES (dimensionless)
            pre_diff_norm = pre_diff_raw / PRE_DIFF_SCALE 
            sep_norm      = sep_raw      / SEP_SCALE 
            psd_norm      = psd_raw      / PSD_SCALE 
            # Add objective vector
            F.append([pre_diff_norm, sep_norm, psd_norm])
            # STORE METADATA FOR ANALYSIS
            meta_list_all.append({
                "pre_diff_raw": pre_diff_raw,
                "sep_raw": sep_raw,
                "psd_raw": psd_raw,
                "pre_diff_norm": pre_diff_norm,
                "sep_norm": sep_norm,
                "psd_norm": psd_norm,
                "pre_diff_mean": float(pre_diff),
                "sep_mean": float(sep_mean),
                "psd_mean": float(psd_mean),
                "frI_pre_mean": frI_pre_mean,
                "frI_dec_mean": frI_dec_mean,
                "wEE_A": float(wEE_A),
                "wEE_N": float(wEE_N),
                "wEI_A": float(wEI_A),
                "wEI_N": float(wEI_N),
                "wIE_G": float(wIE_G),
                "wII_G": float(wII_G),
                "I_stim": float(I_stim),
            })
        # Final outputs for pymoo
        out["F"] = np.array(F)
        out["meta"] = np.array(meta_list_all, dtype=object)


def pick_best_trial_for_case(trials, winner="E1", lambda_bias=2.0):
    """
    Select the single best trial from a set of trials for a given case.

    The score balances:
        • Decision separation (winner – loser)
        • Pre‑stimulus unbiasedness (penalised via lambda_bias)

    Parameters:
        trials       : list of trial output dicts
        winner       : "E1" or "E2" — which pool should win
        lambda_bias  : penalty weight for pre‑stimulus bias

    Returns:
        best_trial : the trial with the highest score
    """
    dt_sec = 1e-3
    NE1, NE2 = 200, 200

    best_score = -np.inf
    best_trial = None

    for out in trials:
        time = out["time"]
        sE1 = out["spikeCountE1"]
        sE2 = out["spikeCountE2"]

        # Pre‑stimulus window (0–500 ms)
        mask_pre = (time >= 0) & (time < 500)
        T_pre = np.sum(mask_pre) * dt_sec
        rE1_pre = np.sum(sE1[mask_pre]) / (NE1 * T_pre)
        rE2_pre = np.sum(sE2[mask_pre]) / (NE2 * T_pre)
        pre_bias = abs(rE1_pre - rE2_pre)

        # Decision window (500–1500 ms)
        mask_mid = (time >= 500) & (time < 1500)
        T_mid = np.sum(mask_mid) * dt_sec
        rE1_mid = np.sum(sE1[mask_mid]) / (NE1 * T_mid)
        rE2_mid = np.sum(sE2[mask_mid]) / (NE2 * T_mid)
        # Winner–loser separation
        if winner == "E1":
            sep = rE1_mid - rE2_mid
        else:
            sep = rE2_mid - rE1_mid
        # Composite score
        score = sep - lambda_bias * pre_bias

        if score > best_score:
            best_score = score
            best_trial = out

    return best_trial
    
def plot_case_composite_from_arrays(E1_list, E2_list, I_list,
                                    raw_trials_subset,
                                    label, base_out_dir, tag):
    """
    Generate composite plots for a single behavioural case:
        • PSD in three epochs (0–500, 500–1500, 1500–2500 ms)
        • Firing‑rate traces for E1, E2, I
        • Raster plot for the same trial

    Parameters:
        E1_list, E2_list, I_list : lists of firing‑rate arrays
        raw_trials_subset        : list containing the chosen trial
        label                    : case label (e.g., "Case1_E1win")
        base_out_dir             : output directory
        tag                      : filename tag
    """

    if len(E1_list) == 0 or len(E2_list) == 0 or len(I_list) == 0:
        return
    # Create case‑specific folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = os.path.join(base_out_dir, f"{label}_{ts}")
    ensure_dir(case_dir)

    # Mean firing rates from 4-case arrays
    rateE1_mean = np.mean(np.stack(E1_list), axis=0)
    rateE2_mean = np.mean(np.stack(E2_list), axis=0)
    rateI_mean  = np.mean(np.stack(I_list),  axis=0)

    time = raw_trials_subset[0]["time"]

    #PSD computation uses time series segments and FFT
    dt = 1.0
    dt_sec = dt / 1000.0
    idx0 = int(0 / dt)
    idx1 = int(500 / dt)
    idx2 = int(1500 / dt)
    idx3 = int(2500 / dt)

    def compute_psd_segment(seg):
        seg = seg - np.mean(seg)
        N = len(seg)
        if N <= 1:
            return np.array([0.0]), np.array([0.0]), 0.0
        window = np.hanning(N)
        Xw = seg * window
        xf = np.fft.rfft(Xw)
        Sxx = np.real((2 * dt_sec**2 / (N * dt_sec)) * (xf * np.conj(xf)))
        faxis = np.fft.rfftfreq(N, d=dt_sec)
        peak = float(faxis[np.argmax(Sxx)])
        return faxis, Sxx, peak

    psd1_list, psd2_list, psd3_list = [], [], []

    for out in raw_trials_subset:
        I_sum = out["I_sum"]
        f1, S1, _ = compute_psd_segment(I_sum[idx0:idx1])
        f2, S2, _ = compute_psd_segment(I_sum[idx1:idx2])
        f3, S3, _ = compute_psd_segment(I_sum[idx2:idx3])
        psd1_list.append(S1)
        psd2_list.append(S2)
        psd3_list.append(S3)

    psd1_mean = np.mean(np.stack(psd1_list), axis=0)
    psd2_mean = np.mean(np.stack(psd2_list), axis=0)
    psd3_mean = np.mean(np.stack(psd3_list), axis=0)

    peak1 = float(f1[np.argmax(psd1_mean)])
    peak2 = float(f2[np.argmax(psd2_mean)])
    peak3 = float(f3[np.argmax(psd3_mean)])

    # MINI-PSD PLOTS (saved separately, 0-60 Hz, Roman I-III)
    def style_axes_psd(ax, title_roman):
        # Standard styling for PSD subplots
        ax.set_xlim(0, 60)
        plt.xticks([0, 20, 40, 60])
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel(r"Power ($\mu$V$^2$/Hz)", fontsize=20)
        ax.set_title(title_roman, fontsize=22)
        ax.tick_params(axis="both", labelsize=18)
        # remove top/right spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # PSD I (0-500 ms)
    fig_psd1, ax_psd1 = plt.subplots(figsize=(4, 3))
    ax_psd1.plot(f1, psd1_mean, color="black")
    ax_psd1.axvline(peak1, color="red", linestyle="--")
    style_axes_psd(ax_psd1, "I")
    fig_psd1.tight_layout()
    fig_psd1.savefig(os.path.join(case_dir, f"PSD_I_{label}_{tag}.png"), dpi=300)
    plt.close(fig_psd1)

    # PSD II (500-1500 ms)
    fig_psd2, ax_psd2 = plt.subplots(figsize=(4, 3))
    ax_psd2.plot(f2, psd2_mean, color="black")
    ax_psd2.axvline(peak2, color="red", linestyle="--")
    ax_psd2.axvline(30, color="green", linestyle="--")
    style_axes_psd(ax_psd2, "II")
    fig_psd2.tight_layout()
    fig_psd2.savefig(os.path.join(case_dir, f"PSD_II_{label}_{tag}.png"), dpi=300)
    plt.close(fig_psd2)

    # PSD III (1500-2500 ms)
    fig_psd3, ax_psd3 = plt.subplots(figsize=(4, 3))
    ax_psd3.plot(f3, psd3_mean, color="black")
    ax_psd3.axvline(peak3, color="red", linestyle="--")
    style_axes_psd(ax_psd3, "III")
    fig_psd3.tight_layout()
    fig_psd3.savefig(os.path.join(case_dir, f"PSD_III_{label}_{tag}.png"), dpi=300)
    plt.close(fig_psd3)

    # FIRING RATES (Case1/Case2) with Roman numerals, B., 1/7/15 Hz
    fig = plt.figure(figsize=(10, 6))
    ax_rate = fig.add_subplot(1, 1, 1)

    ax_rate.plot(time, rateE1_mean, label="E1", color="blue", linewidth=2.5)
    ax_rate.plot(time, rateE2_mean, label="E2", color="red", linewidth=2.5)
    ax_rate.plot(time, rateI_mean,  label="I",  color="magenta", linewidth=2.0)

    ax_rate.axvline(500, linestyle="--", color="k")
    ax_rate.axvline(1500, linestyle="--", color="k")

    # horizontal reference lines
    for yref in [1.0, 7.0, 15.0]:
        ax_rate.axhline(yref, linestyle="--", color="gray", linewidth=1.0)

    # y-limit to nearest 5 above max
    max_rate = np.nanmax([rateE1_mean.max(), rateE2_mean.max(), rateI_mean.max()])
    ymax = 5 * np.ceil(max_rate / 5.0)
    if ymax < 20:
        ymax = 20  # at least 20 Hz
    ax_rate.set_ylim(0, ymax)

    ax_rate.set_xlim(0, 2500)
    ax_rate.set_xlabel("Time (ms)", fontsize=20)
    ax_rate.set_ylabel("Rate (Hz)", fontsize=20)

    # Roman numerals I, II, III above epochs
    ax_rate.text(250,  ymax * 0.95, "I",  ha="center", va="top", fontsize=22)
    ax_rate.text(1000, ymax * 0.95, "II", ha="center", va="top", fontsize=22)
    ax_rate.text(2000, ymax * 0.95, "III",ha="center", va="top", fontsize=22)

    # "B." in upper-left corner
    ax_rate.text(0.01, 0.98, "B.", transform=ax_rate.transAxes,
                 ha="left", va="top", fontsize=22, fontweight="bold")

    # legend outside to the right
    ax_rate.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=18)

    # remove top/right spines
    for spine in ["top", "right"]:
        ax_rate.spines[spine].set_visible(False)

    ax_rate.tick_params(axis="both", labelsize=18)

    fig.tight_layout()
    fig.savefig(os.path.join(case_dir, f"rates_{label}_{tag}.png"), dpi=300)
    plt.close(fig)

    # MATCHING RASTER (A., same x-axis)
    fig_r = plt.figure(figsize=(10, 6))
    ax_r = fig_r.add_subplot(1, 1, 1)

    neuron_offset_E1 = 0
    neuron_offset_E2 = 200
    neuron_offset_I  = 400

    for out in raw_trials_subset:
        spikesE1 = out["spikeRasterE1"]
        spikesE2 = out["spikeRasterE2"]
        spikesI  = out["spikeRasterI"]
        t = out["time"]

        nz = np.nonzero(spikesE1)
        ax_r.scatter(t[nz[1]], neuron_offset_E1 + nz[0],
                     s=1, color="blue", alpha=0.6)
        nz = np.nonzero(spikesE2)
        ax_r.scatter(t[nz[1]], neuron_offset_E2 + nz[0],
                     s=1, color="red", alpha=0.6)
        nz = np.nonzero(spikesI)
        ax_r.scatter(t[nz[1]], neuron_offset_I + nz[0],
                     s=1, color="magenta", alpha=0.6)

    ax_r.axvline(500, linestyle="--", color="k")
    ax_r.axvline(1500, linestyle="--", color="k")

    ax_r.set_xlim(0, 2500)
    ax_r.set_xlabel("Time (ms)", fontsize=20)
    ax_r.set_ylabel("Neuron index", fontsize=20)
    ax_r.tick_params(axis="both", labelsize=18)

    # "A." in upper-left
    ax_r.text(0.01, 0.98, "A.", transform=ax_r.transAxes,
              ha="left", va="top", fontsize=22, fontweight="bold")

    for spine in ["top", "right"]:
        ax_r.spines[spine].set_visible(False)

    fig_r.tight_layout()
    fig_r.savefig(os.path.join(case_dir, f"raster_{label}_{tag}.png"), dpi=300)
    plt.close(fig_r)

def plot_decision_results(model_params: dict, out_dir: str, tag: str):
    """
    Run full analysis pipeline for the best GA solution:
        1. Run 30 trials
        2. Compute decision‑window statistics
        3. Classify trials into 4 behavioural cases
        4. Compute PSD means
        5. Generate composite plots for Case1 and Case2
        6. Save raw data for reproducibility
    """
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 22
    })

    ensure_dir(out_dir)

    #1. Run trials
    trial_outputs = _run_trials(model_params, n_trials=30)

    #2. Compute decision-window stats
    stats = _compute_decision_window_stats(trial_outputs)
    _save_stats(stats, out_dir)

    #3. Compute 4-case classification
    four_case = _classify_four_cases(trial_outputs)

    #4. Compute PSD means
    psd_data = _compute_psd_means(trial_outputs)

    #5. Generate Case1/Case2 composites
    case1, case2 = four_case["case1"], four_case["case2"]
    best_case1_trial = pick_best_trial_for_case(case1["trials"], winner="E1")
    plot_case_composite_from_arrays(
        case1["E1"], case1["E2"], case1["I"],
        [best_case1_trial], "Case1_E1win", out_dir, tag
    )
    best_case2_trial = pick_best_trial_for_case(case2["trials"], winner="E2")
    plot_case_composite_from_arrays(
        case2["E1"], case2["E2"], case2["I"],
        [best_case2_trial], "Case2_E2win", out_dir, tag
    )

    #6. Save raw data
    _save_raw_data(trial_outputs, four_case, psd_data, out_dir)
    
    return {
        "trial_outputs": trial_outputs,
        "four_case": four_case,
        "psd_data": psd_data
    }

def _run_trials(model_params, n_trials):
    outputs = []
    for t in range(n_trials):
        model = TwoChoiceDecisionSNN(seed=1000 + t, **model_params)
        outputs.append(model.run())
    return outputs

def _compute_decision_window_stats(trial_outputs):
    """
    Compute firing‑rate statistics during the decision window
    (500–1500 ms) across all trials.
    """
    dt_sec = 1e-3
    NE1, NE2, NI = 200, 200, 100

    frE1, frE2, frI, psd, winners = [], [], [], [], []

    for out in trial_outputs:
        time = out["time"]
        mask = (time >= 500) & (time <= 1500)
        T = np.sum(mask) * dt_sec

        frE1.append(np.sum(out["spikeCountE1"][mask]) / (NE1 * T))
        frE2.append(np.sum(out["spikeCountE2"][mask]) / (NE2 * T))
        frI.append(np.sum(out["spikeCountI"][mask]) / (NI * T))

        psd.append(out["psd_peak"])
        winners.append(1 if frE1[-1] > frE2[-1] else 2)

    frE1, frE2, frI, psd, winners = map(np.array, (frE1, frE2, frI, psd, winners))

    return {
        "frE1_mean": float(frE1.mean()),
        "frE1_std": float(frE1.std()),
        "frE2_mean": float(frE2.mean()),
        "frE2_std": float(frE2.std()),
        "frI_mean": float(frI.mean()),
        "frI_std": float(frI.std()),
        "psd_mean": float(psd.mean()),
        "psd_std": float(psd.std()),
        "winner_E1_count": int(np.sum(winners == 1)),
        "winner_E2_count": int(np.sum(winners == 2)),
    }

def _save_stats(stats, out_dir):
    # Write decision statistics to a text file
    with open(os.path.join(out_dir, "decision_stats_30trials.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

def _classify_four_cases(trial_outputs):
    """
    Classify each trial into one of four behavioural cases:
        Case 1: E1 wins (E1 > E2 during decision window)
        Case 2: E2 wins (E2 > E1 during decision window)

    For each case, we store:
        • Smoothed firing‑rate traces for E1, E2, I
        • The raw trial dictionaries
        • Mean traces for plotting

    Returns:
        {
            "case1": { ... },
            "case2": { ... }
        }
    """
    dt_sec = 1e-3
    NE1, NE2, NI = 200, 200, 100
    time = trial_outputs[0]["time"]
    # Smoothing kernel (50 ms moving average)
    smooth = np.ones(50) / 50
    # Storage for the four behavioural categories
    e1_win, e1_lose, e2_win, e2_lose = [], [], [], []
    i_E1win, i_E2win = [], []
    trials_E1win, trials_E2win = [], []

    for out in trial_outputs:
        sE1, sE2, sI = out["spikeCountE1"], out["spikeCountE2"], out["spikeCountI"]
        # Smooth firing‑rate traces (Hz)
        rE1 = np.convolve(sE1, smooth, mode="same") * (1000 / NE1)
        rE2 = np.convolve(sE2, smooth, mode="same") * (1000 / NE2)
        rI  = np.convolve(sI,  smooth, mode="same") * (1000 / NI)
        # Decision window (500–1500 ms)
        mask = (time >= 500) & (time <= 1500)
        T = np.sum(mask) * dt_sec

        mE1 = np.sum(sE1[mask]) / (NE1 * T)
        mE2 = np.sum(sE2[mask]) / (NE2 * T)
        # Classification rule:
        #   If E1 > E2 → E1 wins
        #   Else        → E2 wins
        if mE1 > mE2:
            e1_win.append(rE1)
            e2_lose.append(rE2)
            i_E1win.append(rI)
            trials_E1win.append(out)
        else:
            e2_win.append(rE2)
            e1_lose.append(rE1)
            i_E2win.append(rI)
            trials_E2win.append(out)

    def avg(lst):
        """Return mean trace if list is non‑empty, else NaN array."""
        return np.mean(np.stack(lst), axis=0) if lst else np.full_like(time, np.nan)

    return {
        "case1": {
            "E1": e1_win,
            "E2": e2_lose,
            "I": i_E1win,
            "trials": trials_E1win,
            "mean_E1": avg(e1_win),
            "mean_E2": avg(e2_lose),
        },
        "case2": {
            "E1": e1_lose,
            "E2": e2_win,
            "I": i_E2win,
            "trials": trials_E2win,
            "mean_E1": avg(e1_lose),
            "mean_E2": avg(e2_win),
        }
    }

def _compute_psd_means(trial_outputs):
    """
    Compute mean PSD for three epochs:
        I   : 0–500 ms
        II  : 500–1500 ms
        III : 1500–2500 ms
    """
    dt = 1.0
    dt_sec = dt / 1000
    idx0, idx1, idx2, idx3 = 0, 500, 1500, 2500

    def compute(seg):
        seg = seg - np.mean(seg)
        N = len(seg)
        if N <= 1:
            return np.array([0.0]), np.array([0.0])
        win = np.hanning(N)
        xf = np.fft.rfft(seg * win)
        Sxx = np.real((2 * dt_sec**2 / (N * dt_sec)) * (xf * np.conj(xf)))
        f = np.fft.rfftfreq(N, d=dt_sec)
        return f, Sxx

    psd1, psd2, psd3 = [], [], []

    for out in trial_outputs:
        I = out["I_sum"]
        psd1.append(compute(I[idx0:idx1])[1])
        psd2.append(compute(I[idx1:idx2])[1])
        psd3.append(compute(I[idx2:idx3])[1])

    f1, _ = compute(trial_outputs[0]["I_sum"][idx0:idx1])
    f2, _ = compute(trial_outputs[0]["I_sum"][idx1:idx2])
    f3, _ = compute(trial_outputs[0]["I_sum"][idx2:idx3])

    return {
        "f1": f1, "psd1": np.mean(psd1, axis=0),
        "f2": f2, "psd2": np.mean(psd2, axis=0),
        "f3": f3, "psd3": np.mean(psd3, axis=0),
    }

def _save_raw_data(trial_outputs, four_case, psd_data, out_dir):
    """
    Save all raw data for reproducibility:
        • All trial outputs
        • Four‑case classification
        • PSD means
    """
    np.savez_compressed(
        os.path.join(out_dir, "raw_data_best.npz"),
        trial_outputs=trial_outputs,
        four_case=four_case,
        psd_data=psd_data,
    )
def plot_four_case_rates(four_case, out_dir, tag):

    ensure_dir(out_dir)

    # Extract averaged traces
    mean_e1_win  = four_case["case1"]["mean_E1"]
    mean_e1_lose = four_case["case2"]["mean_E1"]
    mean_e2_win  = four_case["case2"]["mean_E2"]
    mean_e2_lose = four_case["case1"]["mean_E2"]

    time = np.linspace(0, 2500, len(mean_e1_win))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot only if not all-NaN
    def safe_plot(arr, label, color, ls):
        if not np.all(np.isnan(arr)):
            ax.plot(time, arr, label=label, color=color, linestyle=ls, linewidth=2.5)

    safe_plot(mean_e1_win,  "E1 (winner)", "blue", "-")
    safe_plot(mean_e1_lose, "E1 (loser)",  "blue", "--")
    safe_plot(mean_e2_win,  "E2 (winner)", "red",  "-")
    safe_plot(mean_e2_lose, "E2 (loser)",  "red", "--")

    # Epoch boundaries
    ax.axvline(500, linestyle="--", color="k")
    ax.axvline(1500, linestyle="--", color="k")

    # Horizontal reference lines
    for yref in [1.0, 7.0, 15.0]:
        ax.axhline(yref, linestyle="--", color="gray", linewidth=1.2)

    # Y-axis limit
    max_rate = np.nanmax([
        np.nanmax(mean_e1_win),
        np.nanmax(mean_e1_lose),
        np.nanmax(mean_e2_win),
        np.nanmax(mean_e2_lose)
    ])
    ymax = 5 * np.ceil(max_rate / 5.0)
    if ymax < 20:
        ymax = 20
    ax.set_ylim(0, ymax)

    ax.set_xlim(0, 2500)
    ax.set_xlabel("Time (ms)", fontsize=20)
    ax.set_ylabel("Rate (Hz)", fontsize=20)

    # Roman numerals above epochs
    ax.text(250,  ymax * 0.95, "I",  ha="center", va="top", fontsize=22)
    ax.text(1000, ymax * 0.95, "II", ha="center", va="top", fontsize=22)
    ax.text(2000, ymax * 0.95, "III",ha="center", va="top", fontsize=22)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=18)

    # Remove box
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="both", labelsize=18)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"four_case_rates_{tag}.png"), dpi=300)
    plt.close(fig)


def plot_raster_single_trial(model_params: dict, out_dir: str, tag: str):
    """
    Generate a raster plot for a single trial of the best model.

    Shows:
        • E1 spikes (blue)
        • E2 spikes (red)
        • I spikes  (magenta)
        • Epoch boundaries (500 ms, 1500 ms)
        • Population boundaries (E1/E2/I)
    """
    ensure_dir(out_dir)
    # Run one deterministic trial
    model = TwoChoiceDecisionSNN(seed=1, **model_params)
    out = model.run()

    time = out["time"]
    spikeTimes = out["spikeTimes"]
    # Neuron index ranges
    idxE1 = np.arange(0, 200)
    idxE2 = np.arange(200, 400)
    idxI  = np.arange(400, 500)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot E1 spikes
    for i, n in enumerate(idxE1):
        st = np.array(spikeTimes[n])
        if st.size > 0:
            ax.plot(st, np.full_like(st, i), '.', color='blue', markersize=2)
    # Plot E2 spikes 
    offset_E2 = len(idxE1) + 10
    for i, n in enumerate(idxE2):
        st = np.array(spikeTimes[n])
        if st.size > 0:
            ax.plot(st, np.full_like(st, offset_E2 + i), '.', color='red', markersize=2)
    # Plot I spikes
    offset_I = offset_E2 + len(idxE2) + 10
    for i, n in enumerate(idxI):
        st = np.array(spikeTimes[n])
        if st.size > 0:
            ax.plot(st, np.full_like(st, offset_I + i), '.', color='magenta', markersize=2)
    # Population boundaries
    ax.axhline(len(idxE1) + 5, linestyle='--', color='k')
    ax.axhline(offset_E2 + len(idxE2) + 5, linestyle='--', color='k')
    # Epoch boundaries
    ax.axvline(500, linestyle='--', color='k')
    ax.axvline(1500, linestyle='--', color='k')

    ax.set_xlim(0, 2500)
    ax.set_xlabel("Time (ms)", fontsize=20)
    ax.set_ylabel("Neuron index", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    # Panel label
    # "A." in upper-left
    ax.text(0.01, 0.98, "A.", transform=ax.transAxes,ha="left", va="top", fontsize=22, fontweight="bold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Clean aesthetics
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"raster_{tag}.png"), dpi=300)
    plt.close(fig)


def plot_winner_distribution(best_params, out_dir):
    ensure_dir(out_dir)
    """
    Plot histogram of winners across 30 trials.
    Shows whether the network consistently chooses E1 or E2.
    """
    NE1, NE2, NI = 200, 200, 100
    dt_sec = 1e-3

    winners = []
    for t in range(30):
        model = TwoChoiceDecisionSNN(seed=2000 + t, **best_params)
        out = model.run()
        time = out["time"]
        mask = (time >= 500) & (time <= 1500)
        T = np.sum(mask) * dt_sec

        frE1 = np.sum(out["spikeCountE1"][mask]) / (NE1 * T)
        frE2 = np.sum(out["spikeCountE2"][mask]) / (NE2 * T)

        winners.append(1 if frE1 > frE2 else 2)

    plt.figure(figsize=(6, 4))
    plt.hist(winners, bins=[1, 2, 3], rwidth=0.8, color="gray")
    plt.xticks([1.5], ["E1 vs E2"])
    plt.title("Winner Distribution (30 trials)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "winner_distribution.png"))
    plt.close()

def plot_pareto_frontier_decision(res, out_dir):
    """
    Extract and visualise the Pareto frontier for the decision model.

    Steps:
        1. Extract all non‑dominated individuals from all generations
        2. Save them to frontier_full.csv
        3. Filter out penalised individuals (1e6)
        4. Plot 3D Pareto scatter (pre_norm, sep_norm, psd_norm)
        5. Save summary statistics
    """
    ensure_dir(out_dir)

    # 1. Extract full frontier (all individuals from all generations)
    frontier_rows = []
    for gen_idx, gen in enumerate(res.history, start=1):
        for ind in gen.opt:   # ONLY NON-DOMINATED INDIVIDUALS
            f = ind.F
            frontier_rows.append({
                "generation": gen_idx,
                "pre_norm": float(f[0]),
                "sep_norm": float(f[1]),
                "psd_norm": float(f[2]),
                "composite": float(np.sqrt(f[0]**2 + f[1]**2 + f[2]**2))
            })

    import pandas as pd
    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(os.path.join(out_dir, "frontier_full.csv"), index=False)

    # 2. Filter out penalized individuals (1e6)
    pf = frontier_df[
        (frontier_df["pre_norm"] < 1e5) &
        (frontier_df["sep_norm"] < 1e5) &
        (frontier_df["psd_norm"] < 1e5)
    ]

    if len(pf) == 0:
        print("No feasible individuals in Pareto front.")
        return

    # 3. 3D PARETO SCATTER
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        pf["pre_norm"],
        pf["sep_norm"],
        pf["psd_norm"],
        c=pf["composite"],
        cmap="viridis",
        s=40,
        alpha=0.8
    )

    ax.set_xlabel("Normalized Y_I RMSE")
    ax.set_ylabel("Normalized Y_II RMSE")
    ax.set_zlabel("Normalized PSD RMSE")
    fig.colorbar(sc, ax=ax, label="Composite L2")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_3d_decision.png"), dpi=300)
    plt.close(fig)

    # 4. Save summary statistics for the clean Pareto front
    summary = [
        "=== Clean Pareto summary (non-dominated only) ===",
        f"N individuals: {len(pf)}",
        f"pre_norm: min={pf['pre_norm'].min():.3f}, median={pf['pre_norm'].median():.3f}, max={pf['pre_norm'].max():.3f}",
        f"sep_norm: min={pf['sep_norm'].min():.3f}, median={pf['sep_norm'].median():.3f}, max={pf['sep_norm'].max():.3f}",
        f"psd_norm: min={pf['psd_norm'].min():.3f}, median={pf['psd_norm'].median():.3f}, max={pf['psd_norm'].max():.3f}",
    ]

    with open(os.path.join(out_dir, "pareto_summary_clean.txt"), "w") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))


    
def baseline_stability_test():
    """
    Quick diagnostic to ensure the decision SNN baseline dynamics
    are stable before running the GA.

    Checks:
        • E1/E2 baseline firing rates (0.5–2 Hz)
        • Unbiasedness (|E1–E2| < 1.5 Hz)
        • Inhibitory baseline (2–6 Hz)
        • PSD peak frequency (< 20 Hz)

    If any condition fails, the GA is unlikely to converge.
    """
    print("\n=== BASELINE STABILITY TEST ===")

    test_params = dict(
        wEE_A=0.010,
        wEE_N=0.040,
        wEI_A=0.010,
        wEI_N=0.020,
        wIE_G=0.020,
        wII_G=0.010,
        I_stim=0.0,
        bias=0.0,
    )
    # Run one baseline trial
    model = TwoChoiceDecisionSNN(seed=1, **test_params)
    out = model.run()

    time = out["time"]
    mask = (time >= 200) & (time <= 500)
    dt_sec = 1e-3
    T_sec = np.sum(mask) * dt_sec

    NE1, NE2, NI = 200, 200, 100
    # Compute baseline firing rates
    frE1 = np.sum(out["spikeCountE1"][mask]) / (NE1 * T_sec)
    frE2 = np.sum(out["spikeCountE2"][mask]) / (NE2 * T_sec)
    frI  = np.sum(out["spikeCountI"][mask])  / (NI * T_sec)
    psd  = out["psd_peak"]

    print(f"E1 rate: {frE1:.2f} Hz")
    print(f"E2 rate: {frE2:.2f} Hz")
    print(f"I  rate: {frI:.2f} Hz")
    print(f"PSD peak: {psd:.2f} Hz")

    stable = True

    # Excitatory baseline range
    if frE1 < 0.5 or frE1 > 2:
        print("E1 baseline out of range.")
        stable = False
    if frE2 < 0.5 or frE2 > 2:
        print("E2 baseline out of range.")
        stable = False

    # Unbiasedness
    if abs(frE1 - frE2) > 1.5:
        print("E1/E2 pre-stim bias large.")
        stable = False

    # Inhibitory baseline
    if frI < 2 or frI > 6:
        print("Inhibitory baseline out of range.")
        stable = False

    # Oscillation frequency
    if psd > 20:
        print("Baseline oscillation frequency too high.")
        stable = False

    if stable:
        print("Baseline looks stable and MATLAB-like.")
    else:
        print("Baseline unstable - GA may struggle.")

    return stable

# MAIN RUNNER BLOCK
# This block orchestrates the entire optimisation pipeline:
#
#   1. Baseline stability test
#   2. Output directory setup
#   3. NSGA‑III optimisation
#   4. Crash‑proof GA history saving
#   5. Best‑individual selection
#   6. Full behavioural analysis pipeline
#   7. Plot generation (raster, PSD, rates, Pareto)
#
# This is the final section of the script.
if __name__ == "__main__":
    # Baseline check before running GA
    if not baseline_stability_test():
        print("Baseline looks rough, stopping before GA run.")
        sys.exit()

    out_dir = "SNN/MyCode/results/decision_nsga3/run_1" #change path as per see fit or append the run_x x being the number manually
    # Optional NMDA ratio constraint, The user can toggle an NMDA ratio constraint via --ratio.
    if args.ratio:
        subfolder = "with_ratio"
    else:
        subfolder = "without_ratio"
    # Final output directory for this run
    out_dir = os.path.join(subfolder) #change path as per see fit
    ensure_dir(out_dir)
    
    print("Saving results to:", out_dir)

    # 1. RUN NSGA-III OPTIMISATION
    problem = DecisionProblem(apply_ratio_constraint=args.ratio)
    ref_dirs = get_reference_directions(
        "das-dennis",
        n_dim=3,
        n_points= 1
    )
    
    algorithm = NSGA3(
        pop_size=1,
        ref_dirs=ref_dirs,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    # Run optimisation
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 1),
        seed=1,
        verbose=True,
        save_history=True
    )

    # 2. SAVE GA HISTORY IMMEDIATELY (CRASH-PROOF)
    history_path = os.path.join(out_dir, "ga_history_decision.npz")
    
    all_pop = []
    all_F_norm = []
    all_meta_raw = []
    all_composite = []
    
    for gen in res.history:
        pop = np.array([ind.X for ind in gen.pop], dtype=float)
        F   = np.array([ind.F for ind in gen.pop], dtype=float)
    
        meta_gen = []
        for ind in gen.pop:
            m = getattr(ind, "meta", None)
            if m is None:
                meta_gen.append({})
            else:
                meta_gen.append(m)
    
        comp = np.sqrt(np.sum(F**2, axis=1))
    
        all_pop.append(pop)
        all_F_norm.append(F)
        all_meta_raw.append(meta_gen)
        all_composite.append(comp)
    
    np.savez(
        history_path,
        pop=all_pop,
        F_norm=all_F_norm,
        meta=all_meta_raw,
        composite=all_composite,
    )

    print(f"Saved GA history to: {history_path}")
    
    
    # 3. PICK A SINGLE BEST INDIVIDUAL FOR PLOTTING
    # Use final generation, minimise composite L2 of normalised RMSE
    last_pop = res.history[-1].pop
    F_last = np.array([ind.F for ind in last_pop], dtype=float)
    comp_last = np.sqrt(np.sum(F_last**2, axis=1))
    best_idx = int(np.argmin(comp_last))
    best = last_pop[best_idx].X
    best_err = float(comp_last[best_idx])

    best_params = {
        "wEE_A": float(best[0]),
        "wEE_N": float(best[1]),
        "wEI_A": float(best[2]),
        "wEI_N": float(best[3]),
        "wIE_G": float(best[4]),
        "wII_G": float(best[5]),
        "I_stim": float(best[6]),
    }

    print("\nBest decision-network parameters (by L2 of normalised RMSE):")
    for k, v in best_params.items():
        print(f"  {k} = {v:.5f}")
    print(f"Composite error (norm L2) = {best_err:.5f}")

    with open(os.path.join(out_dir, "best_solution_decision_v4.json"), "w") as f:
        json.dump({"params": best_params, "composite_norm_L2": best_err}, f, indent=2)

    # 4. RUN BEST SOLUTION AND GENERATE ALL PLOTS
    results = plot_decision_results(best_params, out_dir, tag="best")
    trial_outputs = results["trial_outputs"]
    four_case = results["four_case"]  
    psd_data = results["psd_data"]

    plot_raster_single_trial(best_params, out_dir, tag="best")
    plot_winner_distribution(best_params, out_dir)
    plot_four_case_rates(four_case, out_dir, tag="best")
    plot_pareto_frontier_decision(res, out_dir)