# This script runs a full NSGA-III optimization pipeline for an
# organoid-inspired spiking neural network model.
#
# Key features:
#   - Uses I_sum (summed synaptic current) to compute PSD
#   - Restricts PSD analysis to 0–0.5 Hz band (ultra-low frequency)
#   - Optimizes 3 parameters: ge, gi, sparsity
#   - Two objectives: firing-rate RMSE, PSD-peak RMSE
#   - Includes:
#       * NSGA-III optimization
#       * Full frontier extraction
#       * Validated re-evaluation with medians + SEM
#       * Convergence detection
#       * Bootstrap evaluation
#       * Diagnostic plots (firing, PSD, Pareto)
#
# This is a streamlined version of your cortex model runner,
# adapted for organoid dynamics and ultra-low-frequency PSD.

from typing import Dict, List, Tuple, Any   # type hints for clarity

import os                                   # filesystem operations
import json                                 # saving parameters
import time                                 # timing GA runs
import datetime                             # timestamping output folders

import numpy as np                          # numerical operations
import pandas as pd                         # DataFrame utilities
import matplotlib.pyplot as plt             # plotting
from mpl_toolkits.mplot3d import Axes3D     # 3D Pareto plotting
# pymoo: multi-objective evolutionary algorithms
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
# Import the organoid SNN model
from snnmoo.organoid_snn import SNN, SNNFirings


# Utility helpers

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def rfft_psd_peak(I_sum: np.ndarray, start_index: int, end_index: int, dt: float = 1.0/1000.0):
    """
    Compute PSD peak frequency from I_sum using rFFT,
    but ONLY within the ultra-low-frequency band 0–0.5 Hz.

    Parameters:
        I_sum       : summed synaptic current time series
        start_index : start of analysis window (discard transients)
        end_index   : end of analysis window
        dt          : timestep (1 ms)

    Returns:
        peak_freq : dominant frequency in 0–0.5 Hz band
        faxis     : full frequency axis
        Sxx       : full PSD spectrum
    """
    # Extract analysis window and remove DC offset
    X = I_sum[start_index:end_index] - np.mean(I_sum[start_index:end_index])
    N = len(X)
    if N <= 1:
        return 0.0, np.array([0.0]), np.array([0.0])
    # Apply Hann window to reduce spectral leakage
    xf = np.fft.rfft(X * np.hanning(N))
    Sxx = np.real((2 * dt**2 / (N * dt)) * (xf * np.conj(xf))) # Compute PSD
    faxis = np.fft.rfftfreq(N, d=dt) # Frequency axis

    # Restrict to 0–0.5 Hz band
    mask = (faxis >= 0) & (faxis <= 0.5)
    f_band = faxis[mask]
    S_band = Sxx[mask]

    if len(S_band) == 0:
        return 0.0, faxis, Sxx
    # Peak frequency in the restricted band
    peak_freq = float(f_band[np.argmax(S_band)])
    return peak_freq, faxis, Sxx


# Constants for reproducibility and meta history

SEED_BASE_EVAL = 1000     # base seed for GA evaluation trials
SEED_LOG = 12345          # seed for logging the best individual each generation
SEED_BOOTSTRAP = 20000    # seed for validated re-evaluation and bootstrap trials
SEED_PLOT = 30000         # seed for generating /firing-rate/PSD plots

# META_HISTORY stores raw RMSEs and parameter values for each
# generation during GA evaluation. This is essential for:
#   - reconstructing raw RMSEs
#   - validated re-evaluation
#   - convergence detectionn
META_HISTORY: List[List[Dict[str, Any]]] = []


# GA PROBLEM DEFINITION
# snn_firing_test() sets up and runs the NSGA‑III optimization
# for the organoid SNN model.
#
# Objectives:
#   1. Normalized firing‑rate RMSE
#   2. Normalized PSD‑peak RMSE (0–0.5 Hz band)
#
# Variables (3):
#   ge, gi, sparsity
#
# Constraints:
#   Each normalized RMSE must be < max_error
#
# Returns:
#   res    → pymoo result object (full GA history)
#   params → dictionary of run parameters for serialization
def snn_firing_test(
    firing_rate_target: float,
    psd_peak_target: float,
    max_error: float,
    generations: int = 25,
    pop_size: int = 50,
    trials_per_candidate: int = 1,
) -> Tuple[Any, Dict]:
    """Create and run the SNN multi-objective optimization problem using I_sum PSD only."""

    META_HISTORY.clear() # Clear global metadata store before each GA run

    class SNNProblem(Problem):
        """
        Defines the multi‑objective optimization problem for pymoo.
        """
        def __init__(self) -> None:
            super().__init__(
                n_var=3,          # ge, gi, sparsity
                n_obj=2,          # firing-rate RMSE (norm), PSD peak RMSE (norm)
                n_constr=2,       # each normalized RMSE must be < max_error
                xl=np.array([0.7, 0.8, 0.8]), # lower bounds
                xu=np.array([1.1, 1.2, 1.0]), # upper bounds
            )

        def _evaluate(self, X: np.ndarray, out: Dict, *args: List, **kwargs: Dict) -> None:
            """
            Evaluate a batch of chromosomes X.
            For each chromosome:
              - Run SNN trials
              - Compute firing rate + PSD peak
              - Compute raw + normalized RMSEs
              - Build objective vector F
              - Build constraint vector G
              - Store metadata for later analysis
            """
            F = [] # objective vectors
            G = [] # constraint vectors
            meta_list: List[Dict[str, Any]] = [] # metadata for this generation
            # Seeds for reproducible trials
            trial_seeds = np.arange(trials_per_candidate) + SEED_BASE_EVAL
            dt = 1.0 / 1000.0
            start_index = int(0.2 / dt)     # discard first 200 ms
            end_index   = int(300.0 / dt)   # analyze up to 300 seconds

            for chromosome in X:
                # Extract parameters
                ge_val       = float(chromosome[0])
                gi_val       = float(chromosome[1])
                sparsity_val = float(chromosome[2])

                firing_vals: List[float] = []
                psd_vals:    List[float] = []

                for trial_idx in range(trials_per_candidate):
                    seed = int(trial_seeds[trial_idx])
                    # Instantiate SNN with candidate parameters
                    model = SNN(ge=ge_val, gi=gi_val, sparsity=sparsity_val)
                    result = model.run_network(seed=seed) # Run network
                    # Firing rate (Hz per neuron)
                    score = result.score()
                    firing_vals.append(score["average_firing"])
                    # PSD peak frequency (0–0.5 Hz)
                    peak_freq_isum, _, _ = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
                    psd_vals.append(peak_freq_isum)

                firing_vals_arr = np.array(firing_vals)
                psd_vals_arr    = np.array(psd_vals)
                # Handle empty arrays (should not happen)
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

                F.append([rmse_firing_rate_norm, rmse_psd_norm]) # Objective vector
                G.append([ # Constraint vector
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
            # Assign outputs for pymoo
            out["F"]    = np.array(F)
            out["G"]    = np.array(G)
            out["meta"] = meta_list
            # Save metadata for this generation
            try:
                META_HISTORY.append(meta_list)
            except Exception:
                if 'META_HISTORY' not in globals():
                    globals()['META_HISTORY'] = []
                globals()['META_HISTORY'].append(meta_list)

    problem = SNNProblem() # Instantiate problem

    ref_dirs = get_reference_directions( # Reference directions for 2‑objective NSGA‑III
        "das-dennis",
        n_dim=2,          # because we have 2 objectives in the script
        n_points=pop_size
    )
    # Configure NSGA‑III
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    # Run optimization
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
    """
    Save optimization results, plots, and parameters to disk.
    Also reconstructs raw RMSEs, extracts best individuals per
    generation, and prepares data for convergence detection.
    """

    ensure_dir(output) # Ensure output directory exists
    # Lists to store best-individual metrics per generation
    n_evals = []            # number of evaluations so far
    firing_err_vals = []    # normalized firing-rate RMSE from F
    psd_err_vals = []       # normalized PSD RMSE from F
    ge_vals = []            # best ge per generation
    gi_vals = []            # best gi per generation
    sparsity_vals = []      # best sparsity per generation
    flops_vals = []         # FLOPS for best individual
    psd_peak_freqs = []     # PSD peak frequency for best individual
    # Analysis window for PSD
    dt = 1.0 / 1000.0   
    start_index = int(0.2 / dt) # discard first 200 ms
    end_index   = int(300.0 / dt) # analyze up to 300 seconds

    # Extract best individual from each generation in history
    for e in res.history:
        n_evals.append(e.evaluator.n_eval) # Number of evaluations so far
        best_ind = e.opt[0] # Best individual (pymoo sorts opt by objective sum)
        # Normalized RMSEs from objective vector F
        firing_err_vals.append(float(best_ind.F[0]))
        psd_err_vals.append(float(best_ind.F[1]))
        # Parameter values
        ge_vals.append(float(best_ind.X[0]))
        gi_vals.append(float(best_ind.X[1]))
        sparsity_vals.append(float(best_ind.X[2]))
        # Re-run best individual with logging seed for consistency  
        model = SNN(
            ge=float(best_ind.X[0]),
            gi=float(best_ind.X[1]),
            sparsity=float(best_ind.X[2])
        )
        result = model.run_network(seed=SEED_LOG)
        flops_vals.append(result.flops) # FLOPS for diagnostics
        # PSD peak frequency (0–0.5 Hz)
        peak_freq, _, _ = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
        psd_peak_freqs.append(peak_freq)
    # Build DataFrame summarizing best individuals per generation
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
        # Try to get metadata directly from pymoo individual
        try:
            ind_meta = gen.opt[0].get("meta", None)
        except Exception:
            ind_meta = None

        if ind_meta is not None:
            # Metadata stored directly in individual
            m = ind_meta[0] if isinstance(ind_meta, list) else ind_meta
            raw_firing_rate.append(m.get("rmse_firing_rate_raw", np.nan))
            raw_psd_isum.append(m.get("rmse_psd_isum_raw", np.nan))
        else:
            # Fallback: use global META_HISTORY
            try:
                batch_meta = META_HISTORY[gen_idx - 1]
                best_meta = batch_meta[0] if isinstance(batch_meta, list) else batch_meta
                raw_firing_rate.append(best_meta.get("rmse_firing_rate_raw", np.nan))
                raw_psd_isum.append(best_meta.get("rmse_psd_isum_raw", np.nan))
            except Exception:
                raw_firing_rate.append(np.nan)
                raw_psd_isum.append(np.nan)
    # Add raw RMSEs to DataFrame
    d["firing_rate_raw"]        = raw_firing_rate
    d["psd_peak_error_raw_isum"] = raw_psd_isum
    # Compute normalized RMSEs
    eps = 1e-9
    d["rmse_firing_rates_n"] = d["firing_rate_raw"]        / (params["firing_rate_target"] + eps)
    d["rmse_psd_n"]          = d["psd_peak_error_raw_isum"] / (params["psd_peak_target"]    + eps)
    # Composite RMSE metrics
    d["rmse_l2"]        = np.sqrt(d["rmse_firing_rates_n"].fillna(0)**2 + d["rmse_psd_n"].fillna(0)**2)
    d["rmse_mean"]      = (d["rmse_firing_rates_n"].fillna(0) + d["rmse_psd_n"].fillna(0)) / 2.0
    d["rmse_max"]       = d[["rmse_firing_rates_n", "rmse_psd_n"]].max(axis=1)
    d["rmse_l2_smooth"] = d["rmse_l2"].rolling(window=3, min_periods=1).mean() # Smoothed L2 RMSE for convergence detection
    # Save opt.csv
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
    
    # VALIDATED RE‑EVALUATION & CONVERGENCE DETECTION
    # After extracting best individuals per generation, we now:
    #   1. Re-evaluate top-K candidates with fresh seeds
    #   2. Compute medians + SEM for firing + PSD
    #   3. Compute composite RMSE
    #   4. Detect the "proven" best generation
    #   5. Fall back to stability window if needed
    #   6. Select final best candidate for bootstrap evaluation
    TOP_K = 3                 # number of candidates to re-evaluate per generation
    N_REEVAL = 5              # number of fresh trials per candidate
    SMOOTH_WINDOW = 5         # smoothing window for convergence curve
    SUSTAIN_WINDOW = 3        # required sustained improvement window
    STABLE_RUN = 5            # fallback stability window
    REL_CHANGE_THRESH = 0.01  # threshold for relative change in stability window
    N_BOOTSTRAP_FINAL = 10    # number of final bootstrap trials

    def reeval_trials(ge, gi, sparsity, n_trials, seed_base):
        """
        Re-run a candidate multiple times with new seeds.
        Returns arrays of:
            - firing rates
            - PSD peak frequencies
        """
        firings = []
        psd     = []
        for t in range(n_trials):
            seed = int(seed_base + t)
            model = SNN(ge=ge, gi=gi, sparsity=sparsity) # Recreate SNN with candidate parameters
            r = model.run_network(seed=seed)
            s = r.score()
            firings.append(s["average_firing"]) # Average firing rate (Hz per neuron)
            peak, _, _ = rfft_psd_peak(r.I_sum, start_index, end_index, dt=dt) # PSD peak frequency (0–0.5 Hz)
            psd.append(peak)
        return {"firings": np.array(firings), "psd": np.array(psd)}
    
    
    gen_validated = []
    for gen_idx, gen in enumerate(res.history, start=1):
        pop_sorted = sorted(gen.opt, key=lambda ind: float(np.sum(ind.F))) # Sort individuals by sum of objectives (lower = better)
        topk = pop_sorted[:TOP_K] # Select top-K candidates
        candidates = []
        for ind in topk:
            ge        = float(ind.X[0])
            gi        = float(ind.X[1])
            sparsity  = float(ind.X[2])
            # Re-evaluate with fresh seeds
            trials = reeval_trials(ge, gi, sparsity, n_trials=N_REEVAL,
                                   seed_base=SEED_BOOTSTRAP + gen_idx*1000)
            # Skip if no data (should not happen)
            if trials["firings"].size == 0 or trials["psd"].size == 0:
                continue
            # Compute medians
            med_firing = float(np.median(trials["firings"]))
            # Compute SEMs
            sem_firing = float(np.std(trials["firings"], ddof=1)/np.sqrt(len(trials["firings"]))) if len(trials["firings"]) > 1 else float("nan")
            med_psd    = float(np.median(trials["psd"]))
            sem_psd    = float(np.std(trials["psd"], ddof=1)/np.sqrt(len(trials["psd"]))) if len(trials["psd"]) > 1 else float("nan")
            # Absolute RMSEs
            rmse_firing = abs(med_firing - params["firing_rate_target"])
            rmse_psd    = abs(med_psd    - params["psd_peak_target"])
            # Normalized RMSEs
            eps = 1e-9
            rn_firing = rmse_firing / (params["firing_rate_target"] + eps)
            rn_psd    = rmse_psd    / (params["psd_peak_target"]    + eps)
            composite = float(np.sqrt(rn_firing**2 + rn_psd**2)) # Composite RMSE (L2 norm)
            comp_sem  = float(np.sqrt( # Composite SEM (propagated)
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
            # Save raw trial data for this candidate
            np.savez_compressed(
                f"{output}/gen{gen_idx}_ge{ge:.3f}_gi{gi:.3f}_trials.npz",
                firings=trials["firings"],
                psd=trials["psd"]
            )
        # Best candidate = lowest composite RMSE
        best_cand = min(candidates, key=lambda c: c["composite"]) if candidates else None
        gen_validated.append({"generation": gen_idx, "best": best_cand}) # Store validated result for this generation
    # Build smoothed composite RMSE curve for convergence detection
    comps = np.array([g["best"]["composite"] if g["best"] is not None else np.nan for g in gen_validated])
    smoothed = pd.Series(comps).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values

    chosen_gen = None
    proven_flag = False

    for idx, entry in enumerate(gen_validated):
        gen_idx = entry["generation"]
        best = entry["best"]
        if best is None:
            continue
        if gen_idx <= SUSTAIN_WINDOW: # Must be beyond sustain window
            continue
        if not np.isclose(smoothed[idx], np.nanmin(smoothed), atol=1e-12): # Must be global minimum of smoothed curve
            continue
        prev_idxs = list(range(max(0, idx - SUSTAIN_WINDOW), idx)) # Must improve over previous SUSTAIN_WINDOW generations
        if len(prev_idxs) == 0:
            continue
        prev_vals = smoothed[prev_idxs]
        if not all(smoothed[idx] + 1e-12 < pv - best["comp_sem"] for pv in prev_vals):
            continue
        prev_best = gen_validated[prev_idxs[-1]]["best"] # Must not worsen medians beyond SEM
        if prev_best is None:
            continue
        curr_best = best
        if (curr_best["med_firing"] > prev_best["med_firing"] + (curr_best["sem_firing"] if not np.isnan(curr_best["sem_firing"]) else 0)) or \
           (abs(curr_best["med_psd"] - params["psd_peak_target"]) >
            abs(prev_best["med_psd"] - params["psd_peak_target"]) + (curr_best["sem_psd"] if not np.isnan(curr_best["sem_psd"]) else 0)):
            continue
        chosen_gen = gen_idx # Proven best generation found
        proven_flag = True
        break

    if not proven_flag:
        stable_start = None
        # Look for stable window of length STABLE_RUN
        for i in range(len(smoothed) - STABLE_RUN + 1):
            window = smoothed[i:i+STABLE_RUN]
            # Skip windows with NaNs
            if np.any(np.isnan(window)):
                continue
            # Relative change between consecutive points
            rel_changes = np.abs(np.diff(window)) / (np.maximum(np.abs(window[:-1]), 1e-12))
            # If all changes < threshold → stable
            if np.all(rel_changes < REL_CHANGE_THRESH):
                stable_start = i
                break
        if stable_start is not None:
            chosen_gen = int(gen_validated[stable_start + STABLE_RUN - 1]["generation"])
            params["best_gen_validated"] = False
        else:
            # Final fallback: global minimum of smoothed curve
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
    # FINAL BOOTSTRAP EVALUATION-
    # After selecting the best generation (chosen_gen), we now:
    #   1. Extract its best candidate parameters
    #   2. Re-run the model N_BOOTSTRAP_FINAL times
    #   3. Save bootstrap results for downstream analysis
    # Extract final chosen parameters
    final_ge       = params["best_candidate"]["ge"]
    final_gi       = params["best_candidate"]["gi"]
    final_sparsity = params["best_candidate"]["sparsity"]
    # Run bootstrap trials with fresh seeds
    final_trials = reeval_trials(
        final_ge, final_gi, final_sparsity,
        n_trials=N_BOOTSTRAP_FINAL,
        seed_base=SEED_BOOTSTRAP + 999999
    )
    # Save bootstrap results
    np.savez_compressed(
        f"{output}/bestgen_{chosen_gen}_final_trials.npz",
        firings=final_trials["firings"],
        psd=final_trials["psd"]
    )
    # Store bootstrap metadata
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
    # Save convergence curves for plotting and diagnostics
    np.savez_compressed(
        f"{output}/rmse_convergence_raw.npz",
        generation=d["generation"].values,
        rmse_l2=d["rmse_l2"].values,
        rmse_l2_smooth=d["rmse_l2_smooth"].values,
        firing_error=d["firing_error"].values,
        psd_error=d["psd_peak_error"].values,
        chosen_gen=chosen_gen
    )
    # Save final parameters to JSON
    with open(f"{output}/params.json", "w") as f:
        json.dump(params, f, indent=2)
    # PLOTTING (Pareto, Firing Rate, PSD)
    #2D Pareto Frontier Plot
    try:
        pf = frontier_df
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        # Scatter plot of all individuals across all generations
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

    # Storage for trial data
    firing_series = []
    psd_trials = []
    N_PLOT_TRIALS = 5

    for trial_idx in range(N_PLOT_TRIALS):
        seed = SEED_PLOT + trial_idx
        # Recreate SNN with final parameters
        model = SNN(ge=best_ge, gi=best_gi, sparsity=best_sparsity)
        result = model.run_network(seed=seed)
        # Compute firing statistics
        s = result.score()
        spikes = result._firings_to_spikes_at_time()
        # Append spike count time series
        firing_series.append(spikes["num_fired"].values)
        firing_array = np.stack(firing_series)
        firing_mean  = np.mean(firing_array, axis=0)
        firing_sem   = np.std(firing_array, axis=0, ddof=1) / np.sqrt(firing_array.shape[0])
        
        time_axis = np.arange(start_index, end_index) * dt * 1000.0
        firing_mean_win = firing_mean[start_index:end_index]
        firing_sem_win  = firing_sem[start_index:end_index]

        # Compute PSD for this trial
        peak, faxis, Sxx = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
        # Save raw trial data
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
    # Global plotting style for publication-quality figures
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
    # Averaged Firing Rate (Hz per neuron)
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
        # Target firing rate line
        plt.axhline(params["firing_rate_target"], color="grey", linestyle="--", linewidth=1.5,
                    label=f"Target = {FIRING_RATE_TARGET} Hz")
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing rate (Hz per neuron)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output}/avg_firing_gen{chosen_gen}.png")
        plt.close()

    # Averaged PSD (I_sum, 0–0.5 Hz band)
    if len(psd_trials) > 0:
        psd_array = np.stack(psd_trials)
        psd_mean = np.mean(psd_array, axis=0)
        psd_sem = np.std(psd_array, axis=0, ddof=1) / np.sqrt(psd_array.shape[0])
        band_mask = (final_faxis >= 0) & (final_faxis <= 0.5) # Restrict to 0–0.5 Hz band
        f_band = final_faxis[band_mask]
        psd_mean_band = psd_mean[band_mask]
        psd_sem_band = psd_sem[band_mask]
        peak_freq = f_band[np.argmax(psd_mean_band)] # Peak frequency in averaged PSD
        
        np.savez_compressed( # Save NPZ
            f"{output}/avg_psd_gen{chosen_gen}.npz",
            freq=f_band,
            psd_mean=psd_mean_band,
            psd_sem=psd_sem_band
        )
        # Plot averaged PSD
        plt.figure(figsize=(10, 6))
        plt.plot(f_band, psd_mean_band, color="black", label="Mean PSD")
        plt.fill_between(f_band, psd_mean_band - psd_sem_band, psd_mean_band + psd_sem_band, color="gray", alpha=0.3)
        # Target frequency line
        plt.axvline(params["psd_peak_target"], color="black", linestyle="--", linewidth=2, label=f"Target = {params['psd_peak_target']} Hz")
        # Peak frequency line
        plt.axvline(peak_freq, color="red", linestyle="--", linewidth=2, label=f"Peak = {peak_freq:.3f} Hz")
        plt.xlim(0, 1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power ($\\mu V^2$/Hz)")
        plt.tight_layout()
        plt.savefig(f"{output}/avg_psd_gen{chosen_gen}.png")
        plt.close()
    # Store final chosen generation + parameters
    params["best_gen_final"] = chosen_gen
    params["best_candidate_final"] = {"ge": final_ge, "gi": final_gi}
    # Overwrite params.json with final additions
    with open(f"{output}/params.json", "w") as f:
        json.dump(params, f, indent=2)


# SECTION 9 — MAIN RUNNER BLOCK
# This block executes when the script is run directly.
# It:
#   1. Defines GA hyperparameters
#   2. Creates a timestamped output directory
#   3. Runs the NSGA‑III optimization
#   4. Serializes all results (opt.csv, frontier, plots, params.json)
#   5. Handles errors and logs them
if __name__ == "__main__":
    # GA hyperparameters
    POP_SIZE = 50
    GENERATIONS = 25
    TRIALS_PER_CANDIDATE = 1
    FIRING_RATE_TARGET = 0.09 # Target firing rate (Hz per neuron)
    PSD_TARGET = 0.195 # Target PSD peak frequency (Hz)
    MAX_ERROR = 0.3 # Maximum allowed normalized RMSE for constraints

    base_out = "SNN/MyCode/results/organoids" # Output directory setup
    ensure_dir(base_out)

    t0 = time.time() # start timing the full run
    label = "organoid" # label for this run
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create timestamped output folder
    output_path = os.path.join(base_out, f"{label}_{time_str}")
    ensure_dir(output_path)
    # Run GA + serialization
    try: 
        # Run NSGA‑III optimization
        res, params = snn_firing_test( 
            firing_rate_target=float(FIRING_RATE_TARGET),
            psd_peak_target=float(PSD_TARGET),
            max_error=float(MAX_ERROR),
            generations=GENERATIONS,
            pop_size=POP_SIZE,
            trials_per_candidate=TRIALS_PER_CANDIDATE
        )
        # Add metadata to params
        params["run_label"] = label
        params["run_time_str"] = time_str
        params["training_time_sec"] = time.time() - t0
        # Serialize results (opt.csv, frontier, plots, params.json)
        serialse_results_isum(res, params, output_path)
        print(f"Completed: {label}  (time {params['training_time_sec']:.1f}s)") # Completion message
    except Exception as e:
        # Error handling
        print(f"Run failed for {label}: {e}")
        # Save error message to file for debugging
        with open(os.path.join(output_path, "error.txt"), "w") as ef:
            ef.write(str(e))