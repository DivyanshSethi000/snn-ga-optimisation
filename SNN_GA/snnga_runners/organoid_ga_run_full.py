# GA trial script (I_sum PSD only) with grid over three firing-rate targets and sparsities.
# Synaptic-sum code removed; PSD computed from result.I_sum only.
# Retains plotting, post-hoc re-eval, bootstrap diagnostics, validated best-gen selection,
# and convergence detection based on medians and SEMs (no statistical tests required).

from typing import Dict, List, Tuple, Any

import os
import json
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from snnmoo.organoid_snn import SNN, SNNFirings


# Utility helpers

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def rfft_psd_peak(I_sum: np.ndarray, start_index: int, end_index: int, dt: float = 1.0/1000.0):
    """Return (peak_freq, faxis, Sxx) but ONLY within 0-0.5 Hz band."""
    X = I_sum[start_index:end_index] - np.mean(I_sum[start_index:end_index])
    N = len(X)
    if N <= 1:
        return 0.0, np.array([0.0]), np.array([0.0])

    xf = np.fft.rfft(X * np.hanning(N))
    Sxx = np.real((2 * dt**2 / (N * dt)) * (xf * np.conj(xf)))
    faxis = np.fft.rfftfreq(N, d=dt)

    # Restrict to 0-0.5 Hz
    mask = (faxis >= 0) & (faxis <= 0.5)
    f_band = faxis[mask]
    S_band = Sxx[mask]

    if len(S_band) == 0:
        return 0.0, faxis, Sxx

    peak_freq = float(f_band[np.argmax(S_band)])
    return peak_freq, faxis, Sxx


# Constants for reproducibility and meta history

SEED_BASE_EVAL = 1000
SEED_LOG = 12345
SEED_BOOTSTRAP = 20000
SEED_PLOT = 30000

# Module-level store to reliably capture per-evaluate meta information
META_HISTORY: List[List[Dict[str, Any]]] = []


# GA problem definition
def snn_firing_test(
    firing_rate_target: float,
    psd_peak_target: float,
    max_error: float,
    generations: int = 25,
    pop_size: int = 50,
    trials_per_candidate: int = 1,
) -> Tuple[Any, Dict]:
    """Create and run the SNN multi-objective optimization problem using I_sum PSD only."""

    META_HISTORY.clear()

    class SNNProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=3,          # ge, gi, sparsity
                n_obj=2,          # firing-rate RMSE (norm), PSD peak RMSE (norm)
                n_constr=2,       # each normalized RMSE must be < max_error
                xl=np.array([0.7, 0.8, 0.8]),
                xu=np.array([1.1, 1.2, 1.0]),
            )

        def _evaluate(self, X: np.ndarray, out: Dict, *args: List, **kwargs: Dict) -> None:
            F = []
            G = []
            meta_list: List[Dict[str, Any]] = []

            trial_seeds = np.arange(trials_per_candidate) + SEED_BASE_EVAL
            dt = 1.0 / 1000.0
            start_index = int(0.2 / dt)     # 200 ms
            end_index   = int(300.0 / dt)   # 300 s = 300,000 ms

            for chromosome in X:
                ge_val       = float(chromosome[0])
                gi_val       = float(chromosome[1])
                sparsity_val = float(chromosome[2])

                firing_vals: List[float] = []
                psd_vals:    List[float] = []

                for trial_idx in range(trials_per_candidate):
                    seed = int(trial_seeds[trial_idx])
                    model = SNN(ge=ge_val, gi=gi_val, sparsity=sparsity_val)
                    result = model.run_network(seed=seed)

                    score = result.score()
                    firing_vals.append(score["average_firing"])

                    peak_freq_isum, _, _ = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
                    psd_vals.append(peak_freq_isum)

                firing_vals_arr = np.array(firing_vals)
                psd_vals_arr    = np.array(psd_vals)

                if firing_vals_arr.size == 0 or psd_vals_arr.size == 0:
                    rmse_firing_rate_raw = float("inf")
                    rmse_psd_isum_raw    = float("inf")
                else:
                    rmse_firing_rate_raw = float(
                        np.sqrt(np.mean((firing_vals_arr - firing_rate_target) ** 2))
                    )
                    rmse_psd_isum_raw = float(
                        np.sqrt(np.mean((psd_vals_arr - psd_peak_target) ** 2))
                    )

                eps = 1e-9
                rmse_firing_rate_norm = rmse_firing_rate_raw / (firing_rate_target + eps)
                rmse_psd_norm         = rmse_psd_isum_raw    / (psd_peak_target    + eps)

                F.append([rmse_firing_rate_norm, rmse_psd_norm])
                G.append([
                    rmse_firing_rate_norm - max_error,
                    rmse_psd_norm         - max_error
                ])

                meta_list.append({
                    "rmse_firing_rate_raw": rmse_firing_rate_raw,
                    "rmse_psd_isum_raw":    rmse_psd_isum_raw,
                    "ge": ge_val,
                    "gi": gi_val,
                    "sparsity": sparsity_val
                })

            out["F"]    = np.array(F)
            out["G"]    = np.array(G)
            out["meta"] = meta_list

            try:
                META_HISTORY.append(meta_list)
            except Exception:
                if 'META_HISTORY' not in globals():
                    globals()['META_HISTORY'] = []
                globals()['META_HISTORY'].append(meta_list)

    problem = SNNProblem()

    ref_dirs = get_reference_directions(
        "das-dennis",
        n_dim=2,          # because we have 2 objectives in the script
        n_points=pop_size
    )

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", generations),
        verbose=True,
        save_history=True,
        seed=1
    )

    params = {
        "firing_rate_target": firing_rate_target,
        "psd_peak_target":    psd_peak_target,
        "max_error":          max_error,
        "generations":        generations,
        "pop_size":           pop_size,
        "trials_per_candidate": trials_per_candidate,
        "psd_mode":           "isum"
    }

    return res, params

def serialse_results_isum(res: Any, params: Dict, output: str) -> None:
    """Save optimization results, plots, and parameters to disk and pick best generation."""

    ensure_dir(output)

    n_evals = []
    firing_err_vals = []   # normalized firing error from F
    psd_err_vals    = []   # normalized PSD error from F
    ge_vals = []
    gi_vals = []
    sparsity_vals = []
    flops_vals = []
    psd_peak_freqs = []

    dt = 1.0 / 1000.0
    start_index = int(0.2 / dt)
    end_index   = int(300.0 / dt)

    # Iterate over history and collect best individual per generation
    for e in res.history:
        n_evals.append(e.evaluator.n_eval)
        best_ind = e.opt[0]

        firing_err_vals.append(float(best_ind.F[0]))
        psd_err_vals.append(float(best_ind.F[1]))

        ge_vals.append(float(best_ind.X[0]))
        gi_vals.append(float(best_ind.X[1]))
        sparsity_vals.append(float(best_ind.X[2]))

        model = SNN(
            ge=float(best_ind.X[0]),
            gi=float(best_ind.X[1]),
            sparsity=float(best_ind.X[2])
        )
        result = model.run_network(seed=SEED_LOG)
        flops_vals.append(result.flops)

        peak_freq, _, _ = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
        psd_peak_freqs.append(peak_freq)

    d = pd.DataFrame({
        "generation": list(range(1, len(n_evals) + 1)),
        "n_evals": n_evals,
        "firing_error": firing_err_vals,
        "psd_peak_error": psd_err_vals,
        "ge": ge_vals,
        "gi": gi_vals,
        "sparsity": sparsity_vals,
        "flops": flops_vals,
        "psd_peak_freq": psd_peak_freqs,
    })

    # Extract raw RMSEs from history meta if available, fallback to META_HISTORY
    raw_firing_rate = []
    raw_psd_isum    = []
    for gen_idx, gen in enumerate(res.history, start=1):
        ind_meta = None
        try:
            ind_meta = gen.opt[0].get("meta", None)
        except Exception:
            ind_meta = None

        if ind_meta is not None:
            m = ind_meta[0] if isinstance(ind_meta, list) else ind_meta
            raw_firing_rate.append(m.get("rmse_firing_rate_raw", np.nan))
            raw_psd_isum.append(m.get("rmse_psd_isum_raw", np.nan))
        else:
            try:
                batch_meta = META_HISTORY[gen_idx - 1]
                best_meta = batch_meta[0] if isinstance(batch_meta, list) else batch_meta
                raw_firing_rate.append(best_meta.get("rmse_firing_rate_raw", np.nan))
                raw_psd_isum.append(best_meta.get("rmse_psd_isum_raw", np.nan))
            except Exception:
                raw_firing_rate.append(np.nan)
                raw_psd_isum.append(np.nan)

    d["firing_rate_raw"]        = raw_firing_rate
    d["psd_peak_error_raw_isum"] = raw_psd_isum

    eps = 1e-9
    d["rmse_firing_rates_n"] = d["firing_rate_raw"]        / (params["firing_rate_target"] + eps)
    d["rmse_psd_n"]          = d["psd_peak_error_raw_isum"] / (params["psd_peak_target"]    + eps)

    d["rmse_l2"]        = np.sqrt(d["rmse_firing_rates_n"].fillna(0)**2 + d["rmse_psd_n"].fillna(0)**2)
    d["rmse_mean"]      = (d["rmse_firing_rates_n"].fillna(0) + d["rmse_psd_n"].fillna(0)) / 2.0
    d["rmse_max"]       = d[["rmse_firing_rates_n", "rmse_psd_n"]].max(axis=1)
    d["rmse_l2_smooth"] = d["rmse_l2"].rolling(window=3, min_periods=1).mean()

    d.to_csv(f"{output}/opt.csv", index=False)

    # Save full frontier (all individuals from history)
    frontier_rows = []
    for gen_idx, pop in enumerate(res.history, start=1):
        for ind in pop.opt:
            f = ind.F
            frontier_rows.append({
                "generation": gen_idx,
                "ge": float(ind.X[0]),
                "gi": float(ind.X[1]),
                "sparsity": float(ind.X[2]),
                "firing_rate_error_norm": float(f[0]),
                "psd_peak_error_norm":    float(f[1])
            })
    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(f"{output}/frontier_full.csv", index=False)
    
    TOP_K = 3
    N_REEVAL = 5          
    SMOOTH_WINDOW = 5
    SUSTAIN_WINDOW = 3
    STABLE_RUN = 5
    REL_CHANGE_THRESH = 0.01
    N_BOOTSTRAP_FINAL = 10 

    def reeval_trials(ge, gi, sparsity, n_trials, seed_base):
        firings = []
        psd     = []
        for t in range(n_trials):
            seed = int(seed_base + t)
            model = SNN(ge=ge, gi=gi, sparsity=sparsity)
            r = model.run_network(seed=seed)
            s = r.score()
            # BUG (old): firings.append(s["firing_rate"])
            #   - score() returns "average_firing"
            firings.append(s["average_firing"])
            peak, _, _ = rfft_psd_peak(r.I_sum, start_index, end_index, dt=dt)
            psd.append(peak)
        return {"firings": np.array(firings), "psd": np.array(psd)}
    
    
    gen_validated = []
    for gen_idx, gen in enumerate(res.history, start=1):
        pop_sorted = sorted(gen.opt, key=lambda ind: float(np.sum(ind.F)))
        topk = pop_sorted[:TOP_K]
        candidates = []
        for ind in topk:
            ge        = float(ind.X[0])
            gi        = float(ind.X[1])
            sparsity  = float(ind.X[2])
            trials = reeval_trials(ge, gi, sparsity, n_trials=N_REEVAL,
                                   seed_base=SEED_BOOTSTRAP + gen_idx*1000)
            if trials["firings"].size == 0 or trials["psd"].size == 0:
                continue

            med_firing = float(np.median(trials["firings"]))
            sem_firing = float(np.std(trials["firings"], ddof=1)/np.sqrt(len(trials["firings"]))) if len(trials["firings"]) > 1 else float("nan")
            med_psd    = float(np.median(trials["psd"]))
            sem_psd    = float(np.std(trials["psd"], ddof=1)/np.sqrt(len(trials["psd"]))) if len(trials["psd"]) > 1 else float("nan")

            rmse_firing = abs(med_firing - params["firing_rate_target"])
            rmse_psd    = abs(med_psd    - params["psd_peak_target"])

            eps = 1e-9
            rn_firing = rmse_firing / (params["firing_rate_target"] + eps)
            rn_psd    = rmse_psd    / (params["psd_peak_target"]    + eps)
            composite = float(np.sqrt(rn_firing**2 + rn_psd**2))
            comp_sem  = float(np.sqrt(
                ((sem_firing/(params["firing_rate_target"]+eps))**2 if not np.isnan(sem_firing) else 0.0) +
                ((sem_psd   /(params["psd_peak_target"]   +eps))**2 if not np.isnan(sem_psd)    else 0.0)
            ))

            candidates.append({
                "ind": ind,
                "ge": ge,
                "gi": gi,
                "sparsity": sparsity,
                "composite": composite,
                "comp_sem": comp_sem,
                "med_firing": med_firing,
                "sem_firing": sem_firing,
                "med_psd": med_psd,
                "sem_psd": sem_psd,
                "trials": trials
            })

            np.savez_compressed(
                f"{output}/gen{gen_idx}_ge{ge:.3f}_gi{gi:.3f}_trials.npz",
                firings=trials["firings"],
                psd=trials["psd"]
            )

        best_cand = min(candidates, key=lambda c: c["composite"]) if candidates else None
        gen_validated.append({"generation": gen_idx, "best": best_cand})
        
    comps = np.array([g["best"]["composite"] if g["best"] is not None else np.nan for g in gen_validated])
    smoothed = pd.Series(comps).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values

    chosen_gen = None
    proven_flag = False

    for idx, entry in enumerate(gen_validated):
        gen_idx = entry["generation"]
        best = entry["best"]
        if best is None:
            continue
        if gen_idx <= SUSTAIN_WINDOW:
            continue
        if not np.isclose(smoothed[idx], np.nanmin(smoothed), atol=1e-12):
            continue
        prev_idxs = list(range(max(0, idx - SUSTAIN_WINDOW), idx))
        if len(prev_idxs) == 0:
            continue
        prev_vals = smoothed[prev_idxs]
        if not all(smoothed[idx] + 1e-12 < pv - best["comp_sem"] for pv in prev_vals):
            continue
        prev_best = gen_validated[prev_idxs[-1]]["best"]
        if prev_best is None:
            continue
        curr_best = best
        if (curr_best["med_firing"] > prev_best["med_firing"] + (curr_best["sem_firing"] if not np.isnan(curr_best["sem_firing"]) else 0)) or \
           (abs(curr_best["med_psd"] - params["psd_peak_target"]) >
            abs(prev_best["med_psd"] - params["psd_peak_target"]) + (curr_best["sem_psd"] if not np.isnan(curr_best["sem_psd"]) else 0)):
            continue
        chosen_gen = gen_idx
        proven_flag = True
        break

    if not proven_flag:
        stable_start = None
        for i in range(len(smoothed) - STABLE_RUN + 1):
            window = smoothed[i:i+STABLE_RUN]
            if np.any(np.isnan(window)):
                continue
            rel_changes = np.abs(np.diff(window)) / (np.maximum(np.abs(window[:-1]), 1e-12))
            if np.all(rel_changes < REL_CHANGE_THRESH):
                stable_start = i
                break
        if stable_start is not None:
            chosen_gen = int(gen_validated[stable_start + STABLE_RUN - 1]["generation"])
            params["best_gen_validated"] = False
        else:
            if np.all(np.isnan(smoothed)):
                chosen_gen = 1
            else:
                chosen_idx = int(np.nanargmin(smoothed))
                chosen_gen = int(gen_validated[chosen_idx]["generation"])
            params["best_gen_validated"] = False
    else:
        params["best_gen_validated"] = True

    params["best_generation_by_l2"] = int(chosen_gen)

    chosen_entry = next((g for g in gen_validated if g["generation"] == chosen_gen), None)
    if chosen_entry is None or chosen_entry["best"] is None:
        best_idx = int(d["rmse_l2"].idxmin())
        chosen_gen = int(d.loc[best_idx, "generation"])
        best_history = res.history[chosen_gen - 1]
        best_ind = best_history.opt[0]
        best_ge       = float(best_ind.X[0])
        best_gi       = float(best_ind.X[1])
        best_sparsity = float(best_ind.X[2])
        params["best_candidate"] = {
            "ge": best_ge,
            "gi": best_gi,
            "sparsity": best_sparsity,
            "composite": None
        }
    else:
        chosen_best = chosen_entry["best"]
        params["best_candidate"] = {
            "ge": chosen_best["ge"],
            "gi": chosen_best["gi"],
            "sparsity": chosen_best["sparsity"],
            "composite": chosen_best["composite"]
        }

    final_ge       = params["best_candidate"]["ge"]
    final_gi       = params["best_candidate"]["gi"]
    final_sparsity = params["best_candidate"]["sparsity"]

    final_trials = reeval_trials(
        final_ge, final_gi, final_sparsity,
        n_trials=N_BOOTSTRAP_FINAL,
        seed_base=SEED_BOOTSTRAP + 999999
    )

    np.savez_compressed(
        f"{output}/bestgen_{chosen_gen}_final_trials.npz",
        firings=final_trials["firings"],
        psd=final_trials["psd"]
    )

    params["final_bootstrap_n"] = N_BOOTSTRAP_FINAL
    params["chosen_generation"] = int(chosen_gen)
    params["validation_params"] = {
        "TOP_K": TOP_K,
        "N_REEVAL": N_REEVAL,
        "SMOOTH_WINDOW": SMOOTH_WINDOW,
        "SUSTAIN_WINDOW": SUSTAIN_WINDOW,
        "STABLE_RUN": STABLE_RUN,
        "REL_CHANGE_THRESH": REL_CHANGE_THRESH
    }

    np.savez_compressed(
        f"{output}/rmse_convergence_raw.npz",
        generation=d["generation"].values,
        rmse_l2=d["rmse_l2"].values,
        rmse_l2_smooth=d["rmse_l2_smooth"].values,
        firing_error=d["firing_error"].values,
        psd_error=d["psd_peak_error"].values,
        chosen_gen=chosen_gen
    )

    with open(f"{output}/params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    try:
        pf = frontier_df
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(
            pf["firing_rate_error_norm"],
            pf["psd_peak_error_norm"],
            c='blue', s=20, alpha=0.6
        )
        ax.set_xlabel("Normalised Firing Rate RMSE")
        ax.set_ylabel("Normalised Frequency RMSE")
        plt.tight_layout()
        plt.savefig(f"{output}/pareto_2d_full_history.png")
        plt.close()
    except Exception:
        pass

    # Use the final chosen candidate for plotting
    best_ge = params["best_candidate"]["ge"]
    best_gi = params["best_candidate"]["gi"]
    best_sparsity = params["best_candidate"]["sparsity"]


    firing_series = []
    psd_trials = []
    N_PLOT_TRIALS = 5

    for trial_idx in range(N_PLOT_TRIALS):
        seed = SEED_PLOT + trial_idx
        model = SNN(ge=best_ge, gi=best_gi, sparsity=best_sparsity)
        result = model.run_network(seed=seed)
        s = result.score()
        spikes = result._firings_to_spikes_at_time()
        # FIX: append first
        firing_series.append(spikes["num_fired"].values)
        firing_array = np.stack(firing_series)
        firing_mean  = np.mean(firing_array, axis=0)
        firing_sem   = np.std(firing_array, axis=0, ddof=1) / np.sqrt(firing_array.shape[0])
        
        time_axis = np.arange(start_index, end_index) * dt * 1000.0
        firing_mean_win = firing_mean[start_index:end_index]
        firing_sem_win  = firing_sem[start_index:end_index]


        peak, faxis, Sxx = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
        np.savez_compressed(
            f"{output}/gen{chosen_gen}_trial{trial_idx}_raw.npz",
            I_sum=result.I_sum,
            spikes_num_fired=spikes["num_fired"].values,
            faxis=faxis,
            psd=Sxx,
            average_firing=s["average_firing"],
        )

        psd_trials.append(Sxx)
        #results_for_raster.append(result)

    final_faxis = faxis.copy()
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    # --- Averaged firing rate in Hz per neuron ---
    if len(firing_series) > 0:
        firing_array = np.stack(firing_series)          # shape: (n_trials, T)
        firing_mean  = np.mean(firing_array, axis=0)    # spikes per timestep
        firing_sem   = np.std(firing_array, axis=0, ddof=1) / np.sqrt(firing_array.shape[0])
    
        # Convert to Hz per neuron
        N_neurons = result.Ne + result.Ni   # = 7
        firing_mean_hz = firing_mean / (dt * N_neurons)
        firing_sem_hz  = firing_sem  / (dt * N_neurons)
    
        # Window (200 ms to end)
        time_axis = np.arange(start_index, end_index) * dt * 1000.0
        firing_mean_win = firing_mean_hz[start_index:end_index]
        firing_sem_win  = firing_sem_hz[start_index:end_index]
    
        # Save
        np.savez_compressed(
            f"{output}/avg_firing_gen{chosen_gen}.npz",
            time_axis=time_axis,
            firing_mean=firing_mean_win,
            firing_sem=firing_sem_win,
        )
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, firing_mean_win, color="blue", label="Firing rate (Hz/neuron)")
        plt.fill_between(
            time_axis,
            firing_mean_win - firing_sem_win,
            firing_mean_win + firing_sem_win,
            color="blue", alpha=0.2
        )
        plt.axhline(params["firing_rate_target"], color="grey", linestyle="--", linewidth=1.5,
                    label=f"Target = {FIRING_RATE_TARGET} Hz")
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing rate (Hz per neuron)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output}/avg_firing_gen{chosen_gen}.png")
        plt.close()

    # Averaged PSD (I_sum)
    if len(psd_trials) > 0:
        psd_array = np.stack(psd_trials)
        psd_mean = np.mean(psd_array, axis=0)
        psd_sem = np.std(psd_array, axis=0, ddof=1) / np.sqrt(psd_array.shape[0])
        band_mask = (final_faxis >= 0) & (final_faxis <= 0.5)
        f_band = final_faxis[band_mask]
        psd_mean_band = psd_mean[band_mask]
        psd_sem_band = psd_sem[band_mask]
        peak_freq = f_band[np.argmax(psd_mean_band)]
        
        np.savez_compressed(
            f"{output}/avg_psd_gen{chosen_gen}.npz",
            freq=f_band,
            psd_mean=psd_mean_band,
            psd_sem=psd_sem_band
        )

        plt.figure(figsize=(10, 6))
        plt.plot(f_band, psd_mean_band, color="black", label="Mean PSD")
        plt.fill_between(f_band, psd_mean_band - psd_sem_band, psd_mean_band + psd_sem_band, color="gray", alpha=0.3)
        plt.axvline(params["psd_peak_target"], color="black", linestyle="--", linewidth=2, label=f"Target = {params['psd_peak_target']} Hz")
        plt.axvline(peak_freq, color="red", linestyle="--", linewidth=2, label=f"Peak = {peak_freq:.3f} Hz")
        plt.xlim(0, 1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power ($\\mu V^2$/Hz)")
        plt.tight_layout()
        plt.savefig(f"{output}/avg_psd_gen{chosen_gen}.png")
        plt.close()

    params["best_gen_final"] = chosen_gen
    params["best_candidate_final"] = {"ge": final_ge, "gi": final_gi}
    with open(f"{output}/params.json", "w") as f:
        json.dump(params, f, indent=2)


# Main: run grid over three target pairs

if __name__ == "__main__":

    POP_SIZE = 50
    GENERATIONS = 25
    TRIALS_PER_CANDIDATE = 1
    FIRING_RATE_TARGET = 0.09
    PSD_TARGET = 0.195
    MAX_ERROR = 0.3

    base_out = "SNN/MyCode/results/organoids"
    ensure_dir(base_out)

    t0 = time.time()
    label = "organoid"
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(base_out, f"{label}_{time_str}")
    ensure_dir(output_path)

    try:
        res, params = snn_firing_test(
            firing_rate_target=float(FIRING_RATE_TARGET),
            psd_peak_target=float(PSD_TARGET),
            max_error=float(MAX_ERROR),
            generations=GENERATIONS,
            pop_size=POP_SIZE,
            trials_per_candidate=TRIALS_PER_CANDIDATE
        )
        params["run_label"] = label
        params["run_time_str"] = time_str
        params["training_time_sec"] = time.time() - t0
        serialse_results_isum(res, params, output_path)
        print(f"Completed: {label}  (time {params['training_time_sec']:.1f}s)")
    except Exception as e:
        print(f"Run failed for {label}: {e}")
        with open(os.path.join(output_path, "error.txt"), "w") as ef:
            ef.write(str(e))