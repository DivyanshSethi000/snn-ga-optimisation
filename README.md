# Multi-Objective Optimisation with Oscillatory Dynamics in Spontaneous and Decision Spiking Neural Networks

## Abstract
Spiking neural networks (SNNs) can be used for implementing cost-efficient artificial intelligence computing or mechanistic modelling of experimentally observed neural data. In the latter, fitting neural data with recurrent SNNs (RSNNs) remains a challenge. Importantly, given that neuronal network oscillations are known to play important roles in neural functions, fitting specific RSNN oscillation frequencies with neural firing rates has yet to be fully explored. In this work, we extended our previous application of genetic algorithm (GA), specifically non-dominated sorting GA (NSGA-III), on sensitive Izhikevich neuron-based RSNNs by optimising their connectivity parameters to target emergent neuronal (sub)population firing rates and network oscillation frequencies. We evaluated this, via RMSEs on a Pareto frontier, on spontaneously active simulated RSNN model and low-activation brain organoid, followed by a simulated RSNN model with transient decision dynamics. In all cases, the models comprised spontaneously firing cortical excitatory and inhibitory neurons. We showed that NSGA-III could readily optimise for multiple network firing rates and dominant network oscillation frequencies, and for the decision-making model, for activity patterns in different time epochs. Notably, dominant oscillation frequencies were found to be more parameter sensitive, but firing rates were more robustly met. We also identified low-activity regime for decision-making. Overall, we have successfully demonstrated the implementation of multi-objective GA optimisation on RSNNs’ and brain organoid’s neural firing rates and oscillations.

## Overview
This repository contains a complete framework for multi‑objective genetic optimisation of biologically inspired spiking neural networks (SNNs).

The project includes:

- Three independent SNN models:
  - Cortex model (balanced E/I firing + oscillation frequency)
  - Organoid model (ultra‑slow oscillations + low firing)
  - Decision‑making model (two‑choice attractor network)

- Three NSGA‑III optimisation pipelines

- Three post‑processing pipelines for visualisation and validation

## How the Pipeline Works
### 1. SNN Model Simulation
Each model defines:
- Neuron dynamics (Izhikevich RS/FS)
- AMPA/NMDA/GABA synapses
- Random recurrent connectivity
- Stimulus or background drive
- Spike recording + PSD computation

The models return:
- Spike rasters
- Population spike counts
- Synaptic current trace (I_sum)
- PSD peak frequency
- Trial metadata

### 2. NSGA‑III Optimisation
Each GA runner:
- Defines a multi‑objective problem
- Runs NSGA‑III for a fixed number of generations
- Evaluates each candidate across multiple trials
- Computes normalised RMSE objectives

Saves:
- frontier_full.csv
- raw_data_best.npz
- best_solution.json
- ga_history.npz
These files are the only required inputs for post‑processing.

### 3. Post‑Processing & Visualisation
Each post‑processing script:
- Loads the GA outputs
- Recomputes validation metrics
- Generates publication‑ready figures:

Cortex:
- 3D Pareto frontier
- Raster plots
- Firing‑rate curves
- PSD curves

Organoid:
- Frontier plot
- 1‑second binned firing rate
- Ultra‑slow PSD peak detection

Decision‑making:
- 3D Pareto frontier
- Best‑trial raster
- Epoch‑wise PSD insets
- E1‑win / E2‑win averaged firing rates

All scripts are fully inline‑commented for readability.

### 4. Pipeline Execution & File Dependencies
Each of the three SNN pipelines in this project — Cortex, Organoid, and Decision‑Making — follows the same execution pattern.
Although the models differ, the workflow is identical and reproducible.

#### 1. Model Files
Each pipeline begins with a dedicated SNN model file:
- snn.py (cortex model)
- organoid_snn.py (organoid model)
- two_choice_decision_snn.py (decision-making model)

These files define the neuron dynamics, synaptic mechanisms, connectivity, and simulation loop.
They are imported by the GA runner scripts.

#### 2. GA Runner Scripts (Test → Full)
For each model, there are two GA runner scripts:

A test runner (short, low-generation run)
- genetic_algorithm_run_test.py
- organoid_ga_run_test.py
- decision_making_ga_run_test.py

A full runner (long, full NSGA‑III optimisation)
- genetic_algorithm_run_full.py
- organoid_ga_run_full.py
- decision_making_ga_run_full.py

These are the only files submitted to NI-HPC's Kelvin2 via your SLURM job script.
The model files are imported automatically when the job runs.

The workflow is:
- Upload the test runner to NI-HPC clusters through accessing your Kelvin2 account which has the python 3.10.5 version installed along with their site packages for the virtual environment along with the initialisation files in the snnmoo folder present.
- Submit a job using your SLURM script using k2-sandbox, k2-hipri or k2-medpri partitions according to the runner, using the first one to run all the test runners; using the second one for the decision making runner; using the third one for the genetic algorithm and organoid runners.
- Verify that the test runner file runs without erros
- Once validated, submit the full runner to Kelvin.
- Let the full NSGA‑III optimisation complete.

#### 3. Results Folder (Kelvin Output)
Each full GA run produces a bunch of outputs in certain folders, download those output folders and store them in the below folder paths
- results/snn_ga_1k/
- results/snn_ga_organoid/
- results/snn_ga_decision_making/

Inside each folder, the GA runner saves files  required for post‑processing. Everything needed for reproduction is saved automatically only needing to be downloaded.

#### 4. Post‑Processing Notebooks (Local Analysis)
Once the Kelvin job finishes, you download the results folder to your local machine.

Then you open the corresponding post‑processing script:
- post_processing_ga_1k.py
- post_processing_organoid.py
- post_processing_decision_making.py

Each script is structured like a Jupyter notebook and requires only two things:
- The correct path to the results folder
- The correct case-specific filenames (e.g., avg_psd_gen25.npz, gen20_trial10_raster.npz)

These post-processing scripts:
- Do not re-run the SNN
- Do not re-run the GA
- Only analyse the saved NPZ/CSV files
- Reproduce the exact figures you generated:
    -  Pareto frontiers
    -  Raster plots
    -  Firing-rate curves
    -  PSD curves
    -  Decision-making panels (Firing Rate + PSD insets)

This ensures full reproducibility of the results.