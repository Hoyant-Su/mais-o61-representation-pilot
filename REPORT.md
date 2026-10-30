# Representation selection in the O61 pilot

Haoyang Su · Fudan University and Shanghai Innovation Institute · October 30, 2026

How much does a network's output spectrum tell us about its neurons? This contribution runs the complete finite experiment requested in [MAIS-O61](https://github.com/Math-for-AI-Safety/MAIS/blob/main/open-problems/MAIS-O61.md): 24 configurations, 100 initialization seeds each, for a total of 2,400 trained networks. The data include every final checkpoint and every prespecified outcome. The discussion is at [MAIS issue 46](https://github.com/Math-for-AI-Safety/MAIS/issues/46).

In all 600 C5 width-eight runs, both frequencies entered the key set at the prescribed threshold. That did not imply that the character components explained all centered logits, or that every neuron was pure. For example, C5 with ReLU, width eight and no decay had mean character mass 0.474 and mean pure-neuron fraction 0.139, despite both frequencies entering the key set in all 100 seeds. The three readouts measure different properties.

## Protocol

The groups are C5 and S3. The activations are quadratic and ReLU, widths are two and eight, and decay coefficients are 0, 0.0001 and 0.01. Every parameter starts as an independent Gaussian with standard deviation one divided by the square root of the group order. The loss is mean cross-entropy over the full multiplication table plus lambda times the squared Euclidean parameter norm. It is optimized with Adam at learning rate 0.001, beta coefficients 0.9 and 0.999, and numerical stabilizer 1e-8. This is ordinary Adam on the penalized objective, with no decoupled decay. ReLU has derivative zero at zero.

All computations use float64. Each run is limited to 20,000 updates. It stops sooner when both change measures remain below 1e-7 for 500 consecutive updates. Loss change is divided by max(1, absolute previous loss), and parameter displacement is divided by max(1, previous parameter norm). This relative-change convention was stated in the issue before the experiment. A run that reaches the limit without satisfying the rule is retained and counted as nonconvergent under that rule. This label does not assert mathematical divergence.

Seeds are 0 through 99. Each initialization uses NumPy's default generator with entropy [20261030, group index, width, seed], followed by independent normal draws of shape (3, width, group order). The same initialization is reused across decay coefficients and activations for a given group, width and seed. Seeds are independent within a configuration. Comparisons between configurations are paired by initialization.

C5 elements are integers modulo five. S3 elements are permutations of (0, 1, 2) in lexicographic order, with multiplication p times q defined by p(q(k)). The identity is the first element. The data are the complete multiplication tables, generated exactly from these definitions. Reported accuracy is accuracy on that full training table, not held-out generalization accuracy.

## Measured quantities

The observables follow the definitions in [MAIS-A5](https://github.com/Math-for-AI-Safety/MAIS/tree/main/agendas/A5). Logits are centered over candidate outputs for each input pair. For each irreducible character, its correlation with the centered logits is divided by both norms. Squared absolute correlations are summed over each conjugate pair. The key threshold 0.005 applies to the square root of this sum. The spectral count is the smallest number of classes capturing 95% of the total character mass. It is not the number capturing 95% of all logit variance.

Neuron purity uses the sum of the centered energies of all three weight vectors. A neuron is pure for a class when at least 95% of that energy projects into its isotypic component. A neuron with zero centered energy is pure for no class. The S3 sign and standard projectors have ranks one and four respectively. Multiplicity divides the number of pure neurons by the full width, including mixed neurons in the denominator.

Each key-set probability has a pointwise 95% Wilson interval over the 100 seeds. These intervals quantify Monte Carlo uncertainty for this finite algorithm and stopping convention. They are not simultaneous intervals across the grid and do not estimate an infinite-time or infinite-width law. No hypothesis tests or model selection are performed.

## Results

All 2,400 runs produced finite checkpoints. Of these, 467 met the stopping rule and 1,933 reached the update limit. None of the 1,200 ReLU runs met the stopping rule. The latter observation must be read alongside the table accuracy: many runs classified every group product correctly while continuing to change in parameter space.

The tables show the number of converged and perfectly classifying runs out of 100. The final column is the mean fraction of neurons pure for either nontrivial class.

### C5

| Activation | Width | Decay | Converged | Perfect table accuracy | Mean accuracy | Mean character mass | Mean pure fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| quadratic | 2 | 0 | 0 | 50 | 0.859 | 0.162 | 0.240 |
| quadratic | 2 | 0.0001 | 6 | 50 | 0.846 | 0.163 | 0.165 |
| quadratic | 2 | 0.01 | 92 | 2 | 0.719 | 0.199 | 0.430 |
| quadratic | 8 | 0 | 0 | 100 | 1.000 | 0.633 | 0.251 |
| quadratic | 8 | 0.0001 | 64 | 100 | 1.000 | 0.801 | 0.579 |
| quadratic | 8 | 0.01 | 68 | 100 | 1.000 | 0.757 | 0.651 |
| relu | 2 | 0 | 0 | 0 | 0.394 | 0.094 | 0.010 |
| relu | 2 | 0.0001 | 0 | 0 | 0.412 | 0.105 | 0.005 |
| relu | 2 | 0.01 | 0 | 0 | 0.315 | 0.106 | 0.210 |
| relu | 8 | 0 | 0 | 64 | 0.982 | 0.474 | 0.139 |
| relu | 8 | 0.0001 | 0 | 84 | 0.994 | 0.576 | 0.152 |
| relu | 8 | 0.01 | 0 | 12 | 0.904 | 0.578 | 0.395 |

### S3

| Activation | Width | Decay | Converged | Perfect table accuracy | Mean accuracy | Mean character mass | Mean pure fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| quadratic | 2 | 0 | 0 | 27 | 0.877 | 0.101 | 0.630 |
| quadratic | 2 | 0.0001 | 0 | 27 | 0.854 | 0.112 | 0.660 |
| quadratic | 2 | 0.01 | 93 | 0 | 0.603 | 0.243 | 0.655 |
| quadratic | 8 | 0 | 0 | 100 | 1.000 | 0.586 | 0.657 |
| quadratic | 8 | 0.0001 | 74 | 100 | 1.000 | 0.729 | 0.819 |
| quadratic | 8 | 0.01 | 70 | 100 | 1.000 | 0.721 | 0.780 |
| relu | 2 | 0 | 0 | 0 | 0.336 | 0.087 | 0.640 |
| relu | 2 | 0.0001 | 0 | 0 | 0.319 | 0.097 | 0.525 |
| relu | 2 | 0.01 | 0 | 0 | 0.252 | 0.146 | 0.975 |
| relu | 8 | 0 | 0 | 53 | 0.982 | 0.365 | 0.508 |
| relu | 8 | 0.0001 | 0 | 79 | 0.992 | 0.448 | 0.631 |
| relu | 8 | 0.01 | 0 | 3 | 0.829 | 0.454 | 0.956 |

### Same-frequency probability at C5 width two

The event requires both neurons to be pure for the same frequency. It does not require perfect classification and is reported without conditioning on convergence or accuracy.

| Activation | Decay | Observed count | Probability | 95% Wilson interval |
|---|---:|---:|---:|---:|
| quadratic | 0 | 13/100 | 0.13 | [0.078, 0.210] |
| quadratic | 0.0001 | 10/100 | 0.10 | [0.055, 0.174] |
| quadratic | 0.01 | 35/100 | 0.35 | [0.264, 0.447] |
| relu | 0 | 0/100 | 0.00 | [0.000, 0.037] |
| relu | 0.0001 | 0/100 | 0.00 | [0.000, 0.037] |
| relu | 0.01 | 2/100 | 0.02 | [0.006, 0.070] |

For quadratic activation, the zero-decay estimate is 0.13, while at decay 0.01 it is 0.35. At the latter setting only two runs classified the entire C5 table correctly. A shared pure frequency therefore should not be interpreted as a guarantee of a successful classifier. The complete four-outcome key-set distribution, including zero-count outcomes, is in `results/key_set_probabilities.csv`.

## Numerical validation

`validate.py` checks group composition, character orthogonality, the isotypic projections, exact pure and mixed character spectra, zero logits, constant neurons and covariance under a C5 automorphism. The analytical gradients and 50 Adam updates are compared with an independent implementation using PyTorch autograd and `torch.optim.Adam`.

Four complete CPU replays additionally reproduce seed 17 at width eight and decay 0.0001, one for each group and activation. Each replay uses the same number of updates as its saved run, from a fresh initialization. The largest relative parameter difference from the compiled GPU implementation was 1.04e-8. Per-run differences are supplied in `results/replay_*.json`. The replays use the saved stopping iteration, so they validate the trajectory through that iteration rather than independently testing the stopping decision.

The production computation batches independent networks and evaluates their gradients without averaging over the seed dimension. Four GPUs handled the four group/activation combinations concurrently. Each combination took approximately 35 seconds, including compilation and checkpoint writing. Model and optimizer states were saved every 1,000 updates so an interrupted run can continue deterministically. The released checkpoints contain the final parameters and stopping metadata, with no model downloads or private data required.

## Reproduction

The tested environment was Python 3.12, NumPy 2.2.6, SciPy 1.17.0 and PyTorch 2.6.0 with CUDA 12.4 on NVIDIA H100 GPUs. One GPU is sufficient if the four combinations are run sequentially. A few GB of device memory are sufficient. Docker is not required.

Run from this directory:

```sh
python validate.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --group C5 --activation quadratic --resume /tmp/mais-o61-resume
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --group C5 --activation relu --resume /tmp/mais-o61-resume
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --group S3 --activation quadratic --resume /tmp/mais-o61-resume
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --group S3 --activation relu --resume /tmp/mais-o61-resume
python analyze.py
python replay.py --group C5 --activation quadratic
```

The training commands skip completed checkpoint files. To regenerate every fit, first select an empty directory using `--output`, with a separate resume directory. Point the analysis at those checkpoints or place them in `results/checkpoints`. `analyze.py` also reproduces all summary tables directly from the released checkpoints without training. The replay command can be repeated with each group and activation.

`experiment.json` specifies the fixed protocol. `results/runs.csv` contains the 2,400 per-seed observations, `results/summary.csv` contains 24 configuration summaries, and `results/key_set_probabilities.csv` contains 96 key-set probabilities and intervals. The eight checkpoint archives each hold 300 networks, ordered first by decay coefficient and then by seed. Their `theta` arrays have shape (300, 3, width, group order), where the second dimension stores u, v and w. `stopped` is the first update satisfying the stopping rule, or zero when the limit was reached. `last_step` records the final batch update.

## Interpretation and contribution

The pilot supplies the requested finite computation and its reproducible data. It does not identify the limiting selection law, establish a universal representation-selection mechanism, or test the large-width conjecture. Adam dynamics and its stopping convention are distinct from the agenda's formal gradient-flow ensemble. The empirical distinction between character visibility, character mass and neuron purity is the main feature of these results that a proposed selection theory should explain.

This implementation and report were prepared with AI assistance under Haoyang Su's contribution. The mathematical definitions and experimental grid come from MAIS-O61 and MAIS-A5. No authorship or acceptance decision by the MAIS maintainers is implied by this release. The contribution is offered for review under their published policy.

Source definitions: [MAIS-O61](https://github.com/Math-for-AI-Safety/MAIS/blob/main/open-problems/MAIS-O61.md), [MAIS-A5](https://github.com/Math-for-AI-Safety/MAIS/tree/main/agendas/A5), and [contribution policy](https://github.com/Math-for-AI-Safety/MAIS/blob/main/CONTRIBUTING.md).

Code, generated data and this report are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
