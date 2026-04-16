import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from snnmoo.two_choice_decision_snn import TwoChoiceDecisionSNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ratio", action="store_true", help="Enable the NMDA ratio constraint")
args = parser.parse_args()

print(">>> USING NSGA-III DECISION GA (3-objective unbiasedness/separation/PSD, RMSE-normalised) <<<")


def ensure_dir(path: str) -> None:
    # Create directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


TRIALS_PER_CANDIDATE = 20

# Targets for RMSE-style objectives
PRE_DIFF_TARGET = 0.0      # target for unbiased pre-stimulus rates
SEP_TARGET      = 4.0      # target separation in Hz
PSD_TARGET      = 30.0     # target PSD peak frequency in Hz

# Normalisation scales to make objectives dimensionless
PRE_DIFF_SCALE = 0.2       # acceptable pre-bias scale
SEP_SCALE      = SEP_TARGET
PSD_SCALE      = PSD_TARGET


class DecisionProblem(Problem):

    def __init__(self, apply_ratio_constraint=False):
        # Define GA problem with 7 variables and 3 objectives
        super().__init__(
            n_var=7,
            n_obj=3,  # norm RMSE: unbiasedness, separation, PSD
            xl=np.array([0.008, 0.03, 0.006, 0.008, 0.010, 0.005, 0.4]),
            xu=np.array([0.03,  0.10, 0.03,  0.04,  0.035, 0.02,  0.8]),
        )
        self.apply_ratio_constraint = apply_ratio_constraint
        self.PENALTY = 1e6

    def sanity_check_params(self, wEE_A, wEE_N, wEI_A, wEI_N, wIE_G, wII_G, I_stim):
        # Quick validation of parameter bounds and non-negativity
        if wEE_A <= 0 or wEE_N <= 0:
            return False
        if wEI_A <= 0 or wEI_N <= 0:
            return False
        if wIE_G <= 0 or wII_G <= 0:
            return False
        if not (0.2 <= I_stim <= 0.8):
            return False
        return True

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate a batch of candidate solutions X
        F = []
        meta_list_all = []

        dt_sec = 1e-3
        NE1, NE2, NI = 200, 200, 100

        for row in X:
            wEE_A, wEE_N, wEI_A, wEI_N, wIE_G, wII_G, I_stim = row

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

            frE1_pre_trials = []
            frE2_pre_trials = []
            sep_trials = []
            psd_trials = []
            frI_dec_trials = []
            frI_pre_trials = []
            frE1_dec_trials = []
            frE2_dec_trials = []

            # Ratio check quantity used for optional constraint
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

                sep_trials.append(abs(frE1_dec - frE2_dec))
                psd_trials.append(out_dict["psd_peak"])
                frI_dec_trials.append(frI_dec)

            # Averages over trials
            pre_diff = abs(np.mean(frE1_pre_trials) - np.mean(frE2_pre_trials))
            sep_mean = np.mean(sep_trials)
            psd_mean = np.mean(psd_trials)
            
            frI_dec_mean = float(np.mean(frI_dec_trials))
            frI_pre_mean = float(np.mean(frI_pre_trials))
            
            frE1_dec_mean = float(np.mean(frE1_dec_trials))
            frE2_dec_mean = float(np.mean(frE2_dec_trials))
            
            # joint pre-diff + separation gate
            # Compute per-trial winner-loser separations
            separations = []
            for e1, e2 in zip(frE1_dec_trials, frE2_dec_trials):
                if e1 > e2:
                    separations.append(e1 - e2)
                elif e2 > e1:
                    separations.append(e2 - e1)
            
            middle_sep = np.mean(separations) if len(separations) > 0 else 0.0
            
            # Hard joint condition: unbiased pre + decent separation
            if not (pre_diff < 0.3 and middle_sep > 1.0):
                F.append([self.PENALTY, self.PENALTY, self.PENALTY])
                meta_list_all.append({
                    "pre_diff_raw": pre_diff,
                    "sep_raw": sep_mean,
                    "psd_raw": psd_mean,
                    "reason": "joint pre_diff/middle_sep failed"
                })
                continue
            
            # Hard reject totally dead networks in decision window:
            if frE1_dec_mean < 0.5 and frE2_dec_mean < 0.5:
                F.append([self.PENALTY, self.PENALTY, self.PENALTY])
                meta_list_all.append({
                    "pre_diff_raw": pre_diff,
                    "sep_raw": sep_mean,
                    "psd_raw": psd_mean,
                    "reason": "both E pools nearly silent"
                })
                continue

            
            # Hard reject runaway inhibition in the decision window:
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

            # Raw error objectives (targets: 0, 0, 0)
            pre_diff_raw = pre_diff                      # want 0
            sep_raw      = max(0.0, SEP_TARGET - sep_mean)   # want 0 if sep_mean >= SEP_TARGET
            psd_raw      = abs(psd_mean - PSD_TARGET)    # want 0 if 30 Hz

            # Normalised RMSE-like objectives
            pre_diff_norm = pre_diff_raw / PRE_DIFF_SCALE 
            sep_norm      = sep_raw      / SEP_SCALE 
            psd_norm      = psd_raw      / PSD_SCALE 

            F.append([pre_diff_norm, sep_norm, psd_norm])

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

        out["F"] = np.array(F)
        out["meta"] = np.array(meta_list_all, dtype=object)


def pick_best_trial_for_case(trials, winner="E1", lambda_bias=2.0):
    # Pick the single best trial from a case by maximizing separation
    # while penalizing pre-stimulus bias
    dt_sec = 1e-3
    NE1, NE2 = 200, 200

    best_score = -np.inf
    best_trial = None

    for out in trials:
        time = out["time"]
        sE1 = out["spikeCountE1"]
        sE2 = out["spikeCountE2"]

        # Pre-stim window
        mask_pre = (time >= 0) & (time < 500)
        T_pre = np.sum(mask_pre) * dt_sec
        rE1_pre = np.sum(sE1[mask_pre]) / (NE1 * T_pre)
        rE2_pre = np.sum(sE2[mask_pre]) / (NE2 * T_pre)
        pre_bias = abs(rE1_pre - rE2_pre)

        # Decision window
        mask_mid = (time >= 500) & (time < 1500)
        T_mid = np.sum(mask_mid) * dt_sec
        rE1_mid = np.sum(sE1[mask_mid]) / (NE1 * T_mid)
        rE2_mid = np.sum(sE2[mask_mid]) / (NE2 * T_mid)

        if winner == "E1":
            sep = rE1_mid - rE2_mid
        else:
            sep = rE2_mid - rE1_mid

        score = sep - lambda_bias * pre_bias

        if score > best_score:
            best_score = score
            best_trial = out

    return best_trial
    
def plot_case_composite_from_arrays(E1_list, E2_list, I_list,
                                    raw_trials_subset,
                                    label, base_out_dir, tag):

    if len(E1_list) == 0 or len(E2_list) == 0 or len(I_list) == 0:
        return

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
    dt_sec = 1e-3
    NE1, NE2, NI = 200, 200, 100
    time = trial_outputs[0]["time"]

    smooth = np.ones(50) / 50

    e1_win, e1_lose, e2_win, e2_lose = [], [], [], []
    i_E1win, i_E2win = [], []
    trials_E1win, trials_E2win = [], []

    for out in trial_outputs:
        sE1, sE2, sI = out["spikeCountE1"], out["spikeCountE2"], out["spikeCountI"]

        rE1 = np.convolve(sE1, smooth, mode="same") * (1000 / NE1)
        rE2 = np.convolve(sE2, smooth, mode="same") * (1000 / NE2)
        rI  = np.convolve(sI,  smooth, mode="same") * (1000 / NI)

        mask = (time >= 500) & (time <= 1500)
        T = np.sum(mask) * dt_sec

        mE1 = np.sum(sE1[mask]) / (NE1 * T)
        mE2 = np.sum(sE2[mask]) / (NE2 * T)

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
    ensure_dir(out_dir)

    model = TwoChoiceDecisionSNN(seed=1, **model_params)
    out = model.run()

    time = out["time"]
    spikeTimes = out["spikeTimes"]

    idxE1 = np.arange(0, 200)
    idxE2 = np.arange(200, 400)
    idxI  = np.arange(400, 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, n in enumerate(idxE1):
        st = np.array(spikeTimes[n])
        if st.size > 0:
            ax.plot(st, np.full_like(st, i), '.', color='blue', markersize=2)

    offset_E2 = len(idxE1) + 10
    for i, n in enumerate(idxE2):
        st = np.array(spikeTimes[n])
        if st.size > 0:
            ax.plot(st, np.full_like(st, offset_E2 + i), '.', color='red', markersize=2)

    offset_I = offset_E2 + len(idxE2) + 10
    for i, n in enumerate(idxI):
        st = np.array(spikeTimes[n])
        if st.size > 0:
            ax.plot(st, np.full_like(st, offset_I + i), '.', color='magenta', markersize=2)

    ax.axhline(len(idxE1) + 5, linestyle='--', color='k')
    ax.axhline(offset_E2 + len(idxE2) + 5, linestyle='--', color='k')

    ax.axvline(500, linestyle='--', color='k')
    ax.axvline(1500, linestyle='--', color='k')

    ax.set_xlim(0, 2500)
    ax.set_xlabel("Time (ms)", fontsize=20)
    ax.set_ylabel("Neuron index", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)

    # "A." in upper-left
    ax.text(0.01, 0.98, "A.", transform=ax.transAxes,ha="left", va="top", fontsize=22, fontweight="bold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"raster_{tag}.png"), dpi=300)
    plt.close(fig)


def plot_winner_distribution(best_params, out_dir):
    ensure_dir(out_dir)

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

    # 4. Save summary
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

    model = TwoChoiceDecisionSNN(seed=1, **test_params)
    out = model.run()

    time = out["time"]
    mask = (time >= 200) & (time <= 500)
    dt_sec = 1e-3
    T_sec = np.sum(mask) * dt_sec

    NE1, NE2, NI = 200, 200, 100
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


if __name__ == "__main__":

    if not baseline_stability_test():
        print("Baseline looks rough, stopping before GA run.")
        sys.exit()

    out_dir = "SNN/MyCode/results/decision_nsga3/run_1"
    if args.ratio:
        subfolder = "with_ratio"
    else:
        subfolder = "without_ratio"
    
    out_dir = os.path.join("results_decision_v4", "run_3", subfolder)
    ensure_dir(out_dir)
    
    print("Saving results to:", out_dir)

    #1. RUN GA OPTIMISATION
    problem = DecisionProblem(apply_ratio_constraint=args.ratio)
    ref_dirs = get_reference_directions(
        "das-dennis",
        n_dim=3,
        n_points=36
    )
    
    algorithm = NSGA3(
        pop_size=36,
        ref_dirs=ref_dirs,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 20),
        seed=1,
        verbose=True,
        save_history=True
    )

    # SAVE GA HISTORY IMMEDIATELY (CRASH-PROOF)
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

    # ===== 4. RUN BEST SOLUTION AND GENERATE ALL PLOTS =====
    results = plot_decision_results(best_params, out_dir, tag="best")
    trial_outputs = results["trial_outputs"]
    four_case = results["four_case"]  
    psd_data = results["psd_data"]

    plot_raster_single_trial(best_params, out_dir, tag="best")
    plot_winner_distribution(best_params, out_dir)
    plot_four_case_rates(four_case, out_dir, tag="best")
    plot_pareto_frontier_decision(res, out_dir)