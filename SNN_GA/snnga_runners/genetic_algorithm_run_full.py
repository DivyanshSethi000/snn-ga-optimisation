# GA trial script (I_sum PSD only) with grid over three firing-rate targets and sparsities.
# Synaptic-sum code removed; PSD computed from result.I_sum only.
# Retains plotting, post-hoc re-eval, bootstrap diagnostics, validated best-gen selection,
# and convergence detection based on medians and SEMs (no statistical tests required).

from typing import Dict, List, Tuple, Any

import os
import sys
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

from snnmoo.snn import SNN, SNNFirings


# Utility helpers

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def rfft_psd_peak(I_sum: np.ndarray, start_index: int, end_index: int, dt: float = 1.0/1000.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (peak_freq, faxis, Sxx) computed with consistent dt using rfft."""
    X = I_sum[start_index:end_index] - np.mean(I_sum[start_index:end_index])
    N = len(X)
    if N <= 1:
        return 0.0, np.array([0.0]), np.array([0.0])
    xf = np.fft.rfft(X * np.hanning(N))
    Sxx = np.real((2 * dt**2 / (N * dt)) * (xf * np.conj(xf)))
    faxis = np.fft.rfftfreq(N, d=dt)
    peak_freq = float(faxis[np.argmax(Sxx)])
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
    ex_target_arg: float,
    in_target_arg: float,
    psd_peak_target: float,
    max_error: float,
    generations: int = 25,
    pop_size: int = 45,
    trials_per_candidate: int = 15,
) -> Tuple[Any, Dict]:
    """Create and run the SNN multi-objective optimization problem using I_sum PSD only."""

    # Clear META_HISTORY for this run to avoid mixing runs
    META_HISTORY.clear()

    class SNNProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=5,          # ge, gi, sparsity, thalamic_ex, thalamic_in (was 3 now 5)
                n_obj=3,          # normalized ex RMSE, in RMSE, PSD peak RMSE
                n_constr=3,       # each normalized RMSE must be < max_error
                xl=np.array([0.0, 0.0, 0.2, 0.0, 0.0]), #lower bounds of n_var
                xu=np.array([2.0, 2.0, 1.0, 10.0, 10.0]), #upper bounds of n_var
            )

        def _evaluate(self, X: np.ndarray, out: Dict, *args: List, **kwargs: Dict) -> None:
            F = []
            G = []
            meta_list: List[Dict[str, Any]] = []

            trial_seeds = np.arange(trials_per_candidate) + SEED_BASE_EVAL
            dt = 1.0 / 1000.0
            start_index = int(0.2 / dt)   # 200 ms
            end_index   = int(1.0 / dt)   # 1000 ms

            for chromosome in X:
                ge_val = float(chromosome[0])
                gi_val = float(chromosome[1])
                sparsity_val = float(chromosome[2])
                th_ex_val     = float(chromosome[3])
                th_in_val     = float(chromosome[4])

                ex_vals: List[float] = []
                in_vals: List[float] = []
                psd_vals: List[float] = []

                for trial_idx in range(trials_per_candidate):
                    seed = int(trial_seeds[trial_idx])
                    model = SNN(
                        ge=ge_val,
                        gi=gi_val,
                        sparsity=sparsity_val,
                        thalamic_ex=th_ex_val,
                        thalamic_in=th_in_val
                    )
                    result = model.run_network(seed=seed)

                    score = result.score()
                    ex_vals.append(score["ex_firing"])
                    in_vals.append(score["in_firing"])

                    peak_freq_isum, _, _ = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
                    psd_vals.append(peak_freq_isum)

                ex_vals_arr = np.array(ex_vals)
                in_vals_arr = np.array(in_vals)
                psd_vals_arr = np.array(psd_vals)

                # Guard against empty arrays
                if ex_vals_arr.size == 0 or in_vals_arr.size == 0 or psd_vals_arr.size == 0:
                    rmse_ex_raw        = float("inf")
                    rmse_in_raw        = float("inf")
                    rmse_psd_isum_raw  = float("inf")
                    ex_mean            = float("inf")
                    in_mean            = float("inf")
                    psd_mean           = float("inf")
                else:
                    ex_mean   = float(ex_vals_arr.mean())
                    in_mean   = float(in_vals_arr.mean())
                    psd_mean  = float(psd_vals_arr.mean())
        
                    rmse_ex_raw       = float(np.sqrt(np.mean((ex_vals_arr  - ex_target_arg)   ** 2)))
                    rmse_in_raw       = float(np.sqrt(np.mean((in_vals_arr  - in_target_arg)   ** 2)))
                    rmse_psd_isum_raw = float(np.sqrt(np.mean((psd_vals_arr - psd_peak_target) ** 2)))

                eps = 1e-9
                rmse_ex_norm  = rmse_ex_raw       / (ex_target_arg   + eps)
                rmse_in_norm  = rmse_in_raw       / (in_target_arg   + eps)
                rmse_psd_norm = rmse_psd_isum_raw / (psd_peak_target + eps)
                
                F.append([rmse_ex_norm, rmse_in_norm, rmse_psd_norm])
                G.append([
                    rmse_ex_norm  - max_error,
                    rmse_in_norm  - max_error,
                    rmse_psd_norm - max_error
                ])


                meta_list.append({
                    "rmse_ex_raw": rmse_ex_raw,
                    "rmse_in_raw": rmse_in_raw,
                    "rmse_psd_isum_raw": rmse_psd_isum_raw,
                    "ex_mean": ex_mean,
                    "in_mean": in_mean,
                    "psd_mean": psd_mean,
                    "ge": ge_val,
                    "gi": gi_val,
                    "sparsity": sparsity_val,
                    "thalamic_ex": th_ex_val,
                    "thalamic_in": th_in_val
                })
        
            out["F"]    = np.array(F)
            out["G"]    = np.array(G)
            out["meta"] = meta_list

            # Append meta_list to module-level META_HISTORY so it is always available after minimize()
            try:
                META_HISTORY.append(meta_list)
            except Exception:
                # defensive fallback
                if 'META_HISTORY' not in globals():
                    globals()['META_HISTORY'] = []
                globals()['META_HISTORY'].append(meta_list)

    problem = SNNProblem()

    # NSGA-III reference directions for 3 objectives
    ref_dirs = get_reference_directions(
        "das-dennis",
        n_dim=3,          # because you have 3 objectives
        n_points=pop_size
    )

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    t_ga_start = time.time()
    
    res = minimize(
        problem,
        algorithm,
        ("n_gen", generations),
        verbose=True,
        save_history=True,
        seed=1
    )

    t_ga_end = time.time()
    print(f"[TIMING] GA optimization took {t_ga_end - t_ga_start:.2f} seconds")
    sys.stdout.flush()


    params = {
        "ex_target": ex_target_arg,
        "in_target": in_target_arg,
        "psd_peak_target": psd_peak_target,
        "max_error": max_error,
        "generations": generations,
        "pop_size": pop_size,
        "trials_per_candidate": trials_per_candidate,
        "psd_mode": "isum"
    }

    return res, params


# Serialization, diagnostics, and plotting 

def serialise_results_isum(res: Any, params: Dict, output: str) -> None:
    """Save optimization results, plots, and parameters to disk and pick best generation."""

    ensure_dir(output)

    n_evals = []
    ex_vals = []
    inhib_vals = []
    psd_vals = []
    ge_vals = []
    gi_vals = []
    sparsity_vals = []
    th_ex_vals = []
    th_in_vals = []
    flops_vals = []
    psd_peak_freqs = []

    dt = 1.0 / 1000.0
    start_index = int(0.2 / dt)
    end_index = int(1.0 / dt)

    # Iterate over history and collect best individual per generation
    for gen_idx, e in enumerate(res.history, start=1):
        print(f"[GA] Generation {gen_idx}/{params['generations']} completed.")
        sys.stdout.flush()
    
        if len(ex_vals) > 0:
            print(f"     Best normalized RMSEs: EX={ex_vals[-1]:.4f}, IN={inhib_vals[-1]:.4f}, PSD={psd_vals[-1]:.4f}")
            sys.stdout.flush()

        n_evals.append(e.evaluator.n_eval)
        best_ind = e.opt[0]

        ex_vals.append(float(best_ind.F[0]))
        inhib_vals.append(float(best_ind.F[1]))
        psd_vals.append(float(best_ind.F[2]))

        ge_vals.append(float(best_ind.X[0]))
        gi_vals.append(float(best_ind.X[1]))
        sparsity_vals.append(float(best_ind.X[2]))
        th_ex_vals.append(float(best_ind.X[3]))
        th_in_vals.append(float(best_ind.X[4]))

        model = SNN(ge=float(best_ind.X[0]), gi=float(best_ind.X[1]), sparsity=float(best_ind.X[2]), thalamic_ex=float(best_ind.X[3]), thalamic_in=float(best_ind.X[4]))
        result = model.run_network(seed=SEED_LOG)
        flops_vals.append(result.flops)

        peak_freq, _, _ = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
        psd_peak_freqs.append(peak_freq)

        d = pd.DataFrame({
        "generation": list(range(1, len(n_evals) + 1)),
        "n_evals": n_evals,
        "ex_error": ex_vals,
        "inhib_error": inhib_vals,
        "psd_peak_error": psd_vals,
        "ge": ge_vals,
        "gi": gi_vals,
        "sparsity": sparsity_vals,
        "thalamic_ex": th_ex_vals,
        "thalamic_in": th_in_vals,
        "flops": flops_vals,
        "psd_peak_freq": psd_peak_freqs,
    })


    # Extract raw RMSEs from history meta if available, fallback to META_HISTORY

    raw_ex = []
    raw_in = []
    raw_psd_isum = []
    for gen_idx, gen in enumerate(res.history, start=1):
        print(f"[REEVAL] Re-evaluating generation {gen_idx}/{params['generations']}...")
        sys.stdout.flush()
        ind_meta = None
        try:
            ind_meta = gen.opt[0].get("meta", None)
        except Exception:
            ind_meta = None

        if ind_meta is not None:
            m = ind_meta[0] if isinstance(ind_meta, list) else ind_meta
            raw_ex.append(m.get("rmse_ex_raw", np.nan))
            raw_in.append(m.get("rmse_in_raw", np.nan))
            raw_psd_isum.append(m.get("rmse_psd_isum_raw", np.nan))
        else:
            # fallback to module-level META_HISTORY (indexing: generation -> meta_list)
            try:
                batch_meta = META_HISTORY[gen_idx - 1]   # zero-based index
                best_meta = batch_meta[0] if isinstance(batch_meta, list) else batch_meta
                raw_ex.append(best_meta.get("rmse_ex_raw", np.nan))
                raw_in.append(best_meta.get("rmse_in_raw", np.nan))
                raw_psd_isum.append(best_meta.get("rmse_psd_isum_raw", np.nan))
            except Exception:
                raw_ex.append(np.nan); raw_in.append(np.nan); raw_psd_isum.append(np.nan)

    d["ex_error_raw"] = raw_ex
    d["inhib_error_raw"] = raw_in
    d["psd_peak_error_raw_isum"] = raw_psd_isum

    eps = 1e-9
    d["rmse_ex_n"]  = d["ex_error_raw"]  / (params["ex_target"]  + eps)
    d["rmse_in_n"]  = d["inhib_error_raw"] / (params["in_target"] + eps)
    d["rmse_psd_n"] = d["psd_peak_error_raw_isum"] / (params["psd_peak_target"] + eps)

    d["rmse_l2"] = np.sqrt(d["rmse_ex_n"].fillna(0)**2 + d["rmse_in_n"].fillna(0)**2 + d["rmse_psd_n"].fillna(0)**2)
    d["rmse_mean"] = (d["rmse_ex_n"].fillna(0) + d["rmse_in_n"].fillna(0) + d["rmse_psd_n"].fillna(0)) / 3.0
    d["rmse_max"] = d[["rmse_ex_n","rmse_in_n","rmse_psd_n"]].max(axis=1)
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
                "thalamic_ex": float(ind.X[3]),
                "thalamic_in": float(ind.X[4]),
                "ex_firing_error_norm": float(f[0]),
                "inhib_firing_error_norm": float(f[1]),
                "psd_peak_error_norm": float(f[2])
            })
    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(f"{output}/frontier_full.csv", index=False)


    # Validated post-hoc selection and convergence detection

    TOP_K = 3
    N_REEVAL = 5
    SMOOTH_WINDOW = 5
    SUSTAIN_WINDOW = 3
    STABLE_RUN = 5
    REL_CHANGE_THRESH = 0.01   # 1% relative change
    N_BOOTSTRAP_FINAL = 10

    def reeval_trials(ge, gi, sparsity, th_ex, th_in, n_trials, seed_base):
        exs, ins, psd = [], [], []
        for t in range(n_trials):
            seed = int(seed_base + t)
            model = SNN(
                ge=ge,
                gi=gi,
                sparsity=sparsity,
                thalamic_ex=th_ex,
                thalamic_in=th_in
            )
            r = model.run_network(seed=seed)
            s = r.score()
            exs.append(s["ex_firing"])
            ins.append(s["in_firing"])
            peak, _, _ = rfft_psd_peak(r.I_sum, start_index, end_index, dt=dt)
            psd.append(peak)
        return {"exs": np.array(exs), "ins": np.array(ins), "psd": np.array(psd)}
    
    t_reeval_start = time.time()

    gen_validated = []
    for gen_idx, gen in enumerate(res.history, start=1):
        pop_sorted = sorted(gen.opt, key=lambda ind: float(np.sum(ind.F)))
        topk = pop_sorted[:TOP_K]
        candidates = []
        for i, ind in enumerate(topk):
            ge = float(ind.X[0])
            gi = float(ind.X[1])
            sparsity = float(ind.X[2])
            th_ex = float(ind.X[3])
            th_in = float(ind.X[4])
        
            print(f"    Candidate {i+1}/{TOP_K}: ge={ge:.3f}, gi={gi:.3f}, sp={sparsity:.3f}, th_ex={th_ex:.3f}, th_in={th_in:.3f}")
            sys.stdout.flush()

            trials = reeval_trials(ge, gi, sparsity, th_ex, th_in, n_trials=N_REEVAL, seed_base=SEED_BOOTSTRAP + gen_idx*1000)
            # guard against empty trial arrays
            if trials["exs"].size == 0 or trials["ins"].size == 0 or trials["psd"].size == 0:
                continue
            med_ex = float(np.median(trials["exs"]))
            sem_ex = float(np.std(trials["exs"], ddof=1)/np.sqrt(len(trials["exs"]))) if len(trials["exs"]) > 1 else float("nan")
            med_in = float(np.median(trials["ins"]))
            sem_in = float(np.std(trials["ins"], ddof=1)/np.sqrt(len(trials["ins"]))) if len(trials["ins"]) > 1 else float("nan")
            med_psd = float(np.median(trials["psd"]))
            sem_psd = float(np.std(trials["psd"], ddof=1)/np.sqrt(len(trials["psd"]))) if len(trials["psd"]) > 1 else float("nan")

            rmse_ex = abs(med_ex - params["ex_target"])
            rmse_in = abs(med_in - params["in_target"])
            rmse_psd = abs(med_psd - params["psd_peak_target"])

            eps = 1e-9
            rn_ex = rmse_ex / (params["ex_target"] + eps)
            rn_in = rmse_in / (params["in_target"] + eps)
            rn_psd = rmse_psd / (params["psd_peak_target"] + eps)
            composite = float(np.sqrt(rn_ex**2 + rn_in**2 + rn_psd**2))
            print(f"[REEVAL] Gen {gen_idx} | ge={ge:.3f} gi={gi:.3f} sp={sparsity:.3f} "
              f"th_ex={th_ex:.3f} th_in={th_in:.3f} | Composite RMSE = {composite:.4f}")
            sys.stdout.flush()
            comp_sem = float(np.sqrt(
                ( (sem_ex/(params["ex_target"]+eps))**2 if not np.isnan(sem_ex) else 0.0 ) +
                ( (sem_in/(params["in_target"]+eps))**2 if not np.isnan(sem_in) else 0.0 ) +
                ( (sem_psd/(params["psd_peak_target"]+eps))**2 if not np.isnan(sem_psd) else 0.0 )
            ))

            candidates.append({
                "ind": ind, "ge": ge, "gi": gi, "sparsity": float(ind.X[2]), "thalamic_ex": th_ex, "thalamic_in": th_in, "composite": composite, "comp_sem": comp_sem,
                "med_ex": med_ex, "sem_ex": sem_ex, "med_in": med_in, "sem_in": sem_in,
                "med_psd": med_psd, "sem_psd": sem_psd, "trials": trials
            })
            
            np.savez_compressed(
                f"{output}/gen{gen_idx}_ge{ge:.3f}_gi{gi:.3f}_sp{sparsity:.3f}_thex{th_ex:.3f}_thin{th_in:.3f}trials.npz",
                exs=trials["exs"],
                ins=trials["ins"],
                psd=trials["psd"],
                ge=ge,
                gi=gi,
                sparsity=sparsity,
                thalamic_ex=th_ex,
                thalamic_in=th_in
            )


        best_cand = min(candidates, key=lambda c: c["composite"]) if candidates else None
        gen_validated.append({"generation": gen_idx, "best": best_cand})

    # Build smoothed composite series
    comps = np.array([g["best"]["composite"] if g["best"] is not None else np.nan for g in gen_validated])
    smoothed = pd.Series(comps).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values
    
    t_reeval_end = time.time()
    print(f"[TIMING] Re-evaluation phase took {t_reeval_end - t_reeval_start:.2f} seconds")
    sys.stdout.flush()
    

    # Convergence detection: find earliest generation where smoothed composite is stable for STABLE_RUN gens
    chosen_gen = None
    proven_flag = False

    # Strict proven selection
    for idx, entry in enumerate(gen_validated):
        gen_idx = entry["generation"]
        best = entry["best"]
        if best is None:
            continue
        if gen_idx <= SUSTAIN_WINDOW:
            continue
        # require this generation to be the smoothed minimum
        if not np.isclose(smoothed[idx], np.nanmin(smoothed), atol=1e-12):
            continue
        prev_idxs = list(range(max(0, idx - SUSTAIN_WINDOW), idx))
        if len(prev_idxs) == 0:
            continue
        prev_vals = smoothed[prev_idxs]
        # require improvement greater than comp_sem vs each previous
        if not all(smoothed[idx] + 1e-12 < pv - best["comp_sem"] for pv in prev_vals):
            continue
        # require medians not worse than previous by more than SEM
        prev_best = gen_validated[prev_idxs[-1]]["best"]
        if prev_best is None:
            continue
        curr_best = best
        if (curr_best["med_ex"] > prev_best["med_ex"] + (curr_best["sem_ex"] if not np.isnan(curr_best["sem_ex"]) else 0)) or \
           (curr_best["med_in"] > prev_best["med_in"] + (curr_best["sem_in"] if not np.isnan(curr_best["sem_in"]) else 0)) or \
           (abs(curr_best["med_psd"] - params["psd_peak_target"]) > abs(prev_best["med_psd"] - params["psd_peak_target"]) + (curr_best["sem_psd"] if not np.isnan(curr_best["sem_psd"]) else 0)):
            continue
        chosen_gen = gen_idx
        proven_flag = True
        break

    # Fallback: convergence by stability or argmin
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
    # Extract chosen generation entry
    chosen_entry = next((g for g in gen_validated if g["generation"] == chosen_gen), None)
    
    if chosen_entry is None or chosen_entry["best"] is None:
        # Fallback: use GA's best recorded generation
        best_idx = int(d["rmse_l2"].idxmin())
        chosen_gen = int(d.loc[best_idx, "generation"])
        best_history = res.history[chosen_gen - 1]
        best_ind = best_history.opt[0]
    
        best_ge        = float(best_ind.X[0])
        best_gi        = float(best_ind.X[1])
        best_sparsity  = float(best_ind.X[2])
        best_th_ex     = float(best_ind.X[3])
        best_th_in     = float(best_ind.X[4])
    
        params["best_candidate"] = {
            "ge": best_ge,
            "gi": best_gi,
            "sparsity": best_sparsity,
            "thalamic_ex": best_th_ex,
            "thalamic_in": best_th_in,
            "composite": None
        }
    
    else:
        chosen_best = chosen_entry["best"]
    
        params["best_candidate"] = {
            "ge": chosen_best["ge"],
            "gi": chosen_best["gi"],
            "sparsity": chosen_best["sparsity"],
            "thalamic_ex": chosen_best["thalamic_ex"],
            "thalamic_in": chosen_best["thalamic_in"],
            "composite": chosen_best["composite"]
        }
    
    # Final bootstrap evaluation
    final_ge        = params["best_candidate"]["ge"]
    final_gi        = params["best_candidate"]["gi"]
    final_sparsity  = params["best_candidate"]["sparsity"]
    final_th_ex     = params["best_candidate"]["thalamic_ex"]
    final_th_in     = params["best_candidate"]["thalamic_in"]
    
    t_boot_start = time.time()
    print(f"[BOOTSTRAP] Running {N_BOOTSTRAP_FINAL} bootstrap trials on chosen candidate...")
    sys.stdout.flush()
    print(f"    Params: ge={final_ge:.3f}, gi={final_gi:.3f}, sp={final_sparsity:.3f}, "
      f"th_ex={final_th_ex:.3f}, th_in={final_th_in:.3f}")
    sys.stdout.flush()

    final_trials = reeval_trials(
        final_ge, final_gi, final_sparsity,
        final_th_ex, final_th_in,
        n_trials=N_BOOTSTRAP_FINAL,
        seed_base=SEED_BOOTSTRAP + 999999
    )
    
    np.savez_compressed(
        f"{output}/bestgen_{chosen_gen}_final_trials.npz",
        exs=final_trials["exs"],
        ins=final_trials["ins"],
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
        ex_error=d["ex_error"].values,
        inhib_error=d["inhib_error"].values,
        psd_error=d["psd_peak_error"].values,
        chosen_gen=chosen_gen
    )

    # Save params and validated metadata
    with open(f"{output}/params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    t_boot_end = time.time()
    print(f"[TIMING] Final bootstrap took {t_boot_end - t_boot_start:.2f} seconds")
    sys.stdout.flush()
    print("[BOOTSTRAP] Completed.")
    sys.stdout.flush()



    # Plots (RMSE convergence, Pareto, raster, firing rates, PSD)

    plt.figure(figsize=(8, 4))
    if "rmse_l2" in d.columns:
        plt.plot(d["generation"], d["rmse_l2"], label="rmse_l2 (norm)")
        plt.plot(d["generation"], d["rmse_l2_smooth"], label="rmse_l2_smooth", linestyle="--")
    plt.plot(d["generation"], d["ex_error"], label="ex_error (norm)")
    plt.plot(d["generation"], d["inhib_error"], label="inhib_error (norm)")
    plt.plot(d["generation"], d["psd_peak_error"], label="psd_peak_error (norm)")
    plt.axvline(chosen_gen, color="green", linestyle="--", label=f"chosen_gen={chosen_gen}")
    plt.xlabel("Generation")
    plt.ylabel("Normalized RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}/rmse_convergence.png")
    plt.close()

    try:
        pf = frontier_df
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pf["ex_firing_error_norm"], pf["inhib_firing_error_norm"], pf["psd_peak_error_norm"],
                   c='blue', s=20, alpha=0.6)
        ax.set_xlabel("EX error (norm)")
        ax.set_ylabel("IN error (norm)")
        ax.set_zlabel("PSD error (norm)")
        plt.tight_layout()
        plt.savefig(f"{output}/pareto_3d_full_history.png")
        plt.close()
    except Exception:
        pass

    # Use the final chosen candidate for plotting
    best_ge = params["best_candidate"]["ge"]
    best_gi = params["best_candidate"]["gi"]
    best_sparsity = params["best_candidate"]["sparsity"]
    best_th_ex = params["best_candidate"]["thalamic_ex"]
    best_th_in = params["best_candidate"]["thalamic_in"]


    ex_series = []
    in_series = []
    psd_trials = []
    results_for_raster = []
    for trial_idx in range(15):
        seed = SEED_PLOT + trial_idx
        model = SNN(
            ge=best_ge,
            gi=best_gi,
            sparsity=best_sparsity,
            thalamic_ex=best_th_ex,
            thalamic_in=best_th_in
        )
        result = model.run_network(seed=seed)
        s = result.score()
        spikes = result._firings_to_spikes_at_time()
        ex_series.append(spikes["excitatory"].values)
        in_series.append(spikes["inhibitory"].values)
        peak, faxis, Sxx = rfft_psd_peak(result.I_sum, start_index, end_index, dt=dt)
    
        np.savez_compressed(
            f"{output}/gen{chosen_gen}_trial{trial_idx}_raw.npz",
            I_sum=result.I_sum,
            spikes_ex=spikes["excitatory"].values,
            spikes_in=spikes["inhibitory"].values,
            faxis=faxis,
            psd=Sxx,
            ex_firing=s["ex_firing"],
            in_firing=s["in_firing"],
            ge=best_ge,
            gi=best_gi,
            sparsity=best_sparsity,
            thalamic_ex=best_th_ex,
            thalamic_in=best_th_in
        )

    
        np.savez_compressed(
            f"{output}/gen{chosen_gen}_trial{trial_idx}_raster.npz",
            firings_time=np.array(result.firings_time, dtype=object),
            firings_neuron=np.array(result.firings_neuron, dtype=object)
        )
    
        psd_trials.append(Sxx)
        results_for_raster.append(result)
    
    final_faxis = faxis.copy()

    # --- choose a single trial for raster, e.g. trial 0 ---
    best_raster_result = results_for_raster[0]
    
    plt.figure(figsize=(8, 4))
    
    firings_list = []
    if hasattr(best_raster_result, "firings_neuron") and best_raster_result.firings_neuron:
        for neuron_id, spike_times in enumerate(best_raster_result.firings_neuron):
            for t in spike_times:
                firings_list.append((neuron_id, t))
    else:
        for t, fired in enumerate(best_raster_result.firings_time):
            for neuron_id in fired:
                firings_list.append((neuron_id, t))
    
    if firings_list:
        firings = np.array(firings_list)
        mask = (firings[:, 1] >= start_index) & (firings[:, 1] < end_index)
        firings = firings[mask]
        if len(firings) > 0:
            neuron_ids = firings[:, 0]
            times_ms = firings[:, 1] * dt * 1000
            colors = np.where(neuron_ids >= 800, "red", "blue")
            plt.scatter(times_ms, neuron_ids, s=2, c=colors)
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron #")
    plt.title(f"Spike Raster (Gen {chosen_gen}, single trial)")
    plt.tight_layout()
    plt.savefig(f"{output}/spike_raster_gen{chosen_gen}_single_trial.png")
    plt.close()

    # Averaged firing rates
    if len(ex_series) > 0 and len(in_series) > 0:
        ex_array = np.stack(ex_series)
        in_array = np.stack(in_series)
        ex_mean = np.mean(ex_array, axis=0)
        in_mean = np.mean(in_array, axis=0)
        ex_sem = np.std(ex_array, axis=0, ddof=1) / np.sqrt(ex_array.shape[0])
        in_sem = np.std(in_array, axis=0, ddof=1) / np.sqrt(in_array.shape[0])

        time_axis = np.arange(start_index, end_index) * dt * 1000
        ex_mean_win = ex_mean[start_index:end_index]
        in_mean_win = in_mean[start_index:end_index]
        ex_sem_win = ex_sem[start_index:end_index]
        in_sem_win = in_sem[start_index:end_index]
        
        np.savez_compressed(
            f"{output}/avg_firing_rates_gen{chosen_gen}.npz",
            time_axis=time_axis,
            ex_mean=ex_mean_win,
            in_mean=in_mean_win,
            ex_sem=ex_sem_win,
            in_sem=in_sem_win
        )

        plt.figure(figsize=(8, 4))
        plt.plot(time_axis, ex_mean_win, color="blue", label="Excitatory firing (counts)")
        plt.fill_between(time_axis, ex_mean_win - ex_sem_win, ex_mean_win + ex_sem_win, color="blue", alpha=0.2)
        plt.plot(time_axis, in_mean_win, color="red", label="Inhibitory firing (counts)")
        plt.fill_between(time_axis, in_mean_win - in_sem_win, in_mean_win + in_sem_win, color="red", alpha=0.2)
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing (counts per step)")
        plt.title(f"Averaged Firing Rates (Gen {chosen_gen})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output}/avg_firing_rates_gen{chosen_gen}.png")
        plt.close()

    # Averaged PSD (I_sum)
    if len(psd_trials) > 0:
        psd_array = np.stack(psd_trials)
        psd_mean = np.mean(psd_array, axis=0)
        psd_sem = np.std(psd_array, axis=0, ddof=1) / np.sqrt(psd_array.shape[0])
        band_mask = (final_faxis >= 0) & (final_faxis <= 100)
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



        plt.figure(figsize=(8, 4))
        plt.plot(f_band, psd_mean_band, color="black", label="Mean PSD")
        plt.fill_between(f_band, psd_mean_band - psd_sem_band, psd_mean_band + psd_sem_band, color="gray", alpha=0.3)
        plt.axvline(peak_freq, color="red", linestyle="--", alpha=0.7)
        plt.text(peak_freq + 1, psd_mean_band.max() * 0.9, f"Peak = {peak_freq:.1f} Hz", color="red")
        plt.xlim(0, 100)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Averaged PSD (Gen {chosen_gen})")
        plt.tight_layout()
        plt.savefig(f"{output}/avg_psd_gen{chosen_gen}.png")
        plt.close()

    params["best_gen_final"] = chosen_gen
    params["best_candidate_final"] = {"ge": final_ge, "gi": final_gi}
    with open(f"{output}/params.json", "w") as f:
        json.dump(params, f, indent=2)


# Main: run grid over three target pairs

if __name__ == "__main__":
    firing_rates = [
        (2, 10),
        (10, 2),
        (2, 2)
    ]

    POP_SIZE = 45
    GENERATIONS = 25
    TRIALS_PER_CANDIDATE = 15
    PSD_TARGET = 50.0
    MAX_ERROR = 5.0

    base_out = "SNN/MyCode/results/snn_ga_full/nsga_3/run_1"
    ensure_dir(base_out)

    for (ex_t, in_t) in firing_rates:
        t0 = time.time()
        label = f"ex{ex_t}_in{in_t}"
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(base_out, f"{label}_{time_str}")
        ensure_dir(output_path)
        print(f"\nRunning: {label}  -> output: {output_path}\n")
        sys.stdout.flush()

        try:
            res, params = snn_firing_test(
                ex_target_arg=float(ex_t),
                in_target_arg=float(in_t),
                psd_peak_target=float(PSD_TARGET),
                max_error=float(MAX_ERROR),
                generations=GENERATIONS,
                pop_size=POP_SIZE,
                trials_per_candidate=TRIALS_PER_CANDIDATE
            )
            params["run_label"] = label
            params["run_time_str"] = time_str
            params["training_time_sec"] = time.time() - t0
            serialise_results_isum(res, params, output_path)
            print(f"Completed: {label}  (time {params['training_time_sec']:.1f}s)")
            sys.stdout.flush()
        except Exception as e:
            print(f"Run failed for {label}: {e}")
            sys.stdout.flush()
            with open(os.path.join(output_path, "error.txt"), "w") as ef:
                ef.write(str(e))