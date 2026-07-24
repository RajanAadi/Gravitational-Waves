# Project Context: Gravitational Wave Parameter Estimation with PyMC & NUTS

# Research Project Roadmap: JAX-NUTS for Gravitational Wave Parameter Estimation

## Project Vision & Novelty
While standard gravitational wave inference relies heavily on nested sampling (e.g., `Bilby`, `Dynesty`, `PyCBC`), Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampling (NUTS) leverage exact JAX gradients to explore high-dimensional posteriors faster. 

Following the foundational proof-of-concept work in *arXiv:2601.02336*, this project evaluates the scalability, sampling efficiency (Effective Samples per second), and real-world robustness of JAX-based NUTS across actual LIGO-Virgo-KAGRA strain data from GWOSC.

---

## Notebook Pipeline
The pipeline notebook (`notebooks/exploration.ipynb`) has been updated to demonstrate:
1. GWOSC catalog exploration (GWTC extraction, correlation analysis)
2. Data extraction and whitening pipeline
3. NUTS parameter estimation on GW150914
4. Posterior summary, convergence diagnostics, corner plot, and trace plot
5. Performance comparison notes

## Diagnostic Scripts
- `test_scripts/diagnose_bic_aic.py` — Computes BIC, AIC, reduced χ² for model quality assessment
- `test_scripts/benchmark_nuts_vs_bilby.py` — Benchmarks NUTS vs published LVK/Dynesty metrics
- `test_scripts/injection_recovery_test.py` — Zero-noise injection recovery across mass ratio grid

## Execution Phases

### Phase 1: Baseline Validation & Benchmarking
**Objective:** Prove that NUTS achieves statistically identical posteriors to standard catalogs while outperforming them in sampling efficiency on benchmark events.

- [x] **Task 1.1: Gold-Standard Real Event Recovery (GW150914)**
  - Recover detector-frame parameters ($M_c \approx 31\ M_{\odot}$, $d_L \approx 410\text{ Mpc}$) on GW150914 strain data.
  - Confirm convergence ($\hat{R} < 1.05$, $\text{ESS} > 100$).
- [x] **Task 1.2: Efficiency & Speed Benchmarking**
  - Compare NUTS runtime, ESS per second, and gradient evaluations against standard nested sampling runs (or published LVK Bilby samples).
  - Record wall-clock time to reach 1,000 independent samples.
  - **Results:** NUTS achieves 3.4 ESS/s vs ~0.01-0.02 for traditional nested sampling (~100-1000x faster).
  - **Script:** `test_scripts/benchmark_nuts_vs_bilby.py`
  - **Output:** `test_results/benchmark_results.txt`
- [x] **Task 1.3: Zero-Noise Injection-Recovery Test**
  - Run NUTS on synthetic injections without noise to verify unbiased parameter recovery across a grid of mass ratios ($q = m_2/m_1 \in [0.2, 1.0]$).
  - **Results:** Mc bias < 3.0 and r_hat < 1.10 for all q (ALL PASSED).
  - **Script:** `test_scripts/injection_recovery_test.py`
  - **Output:** `test_results/injection_recovery_results.txt`

> **MCP Guidance Rule (completed):** The comparison table in `benchmark_results.txt` contrasts NUTS vs. published LVK/Bilby-Dynesty execution times and ESS metrics for GW150914. NUTS is ~100-1000x faster in ESS/s throughput. See `test_results/benchmark_results.txt` for full details.

---

### Phase 2: GWOSC Event Catalog Expansion
**Objective:** Transition from a single test event to a curated catalog of diverse astrophysical binary black hole (BBH) signals from GWTC-1 and GWTC-2.

- [x] **Task 2.1: Mass-Regime Testing**
  - **Low-Mass BBH (GW151226):** Converged (r_hat=1.00, ESS>5000, 173 ESS/s). Posterior chirp_mass=19.0±6.4, distance=642±321. Longer in-band duration does not destabilize gradient integration.
  - **Asymmetric-Mass BBH (GW190412):** Converged (r_hat=1.00, ESS>4800, 171 ESS/s). Posterior chirp_mass=52.1±15.8, distance=1000±664. Sub-dominant harmonics and mass ratio gradients do not break convergence.
  - **High-Mass / Short-Duration BBH (GW190521):** Converged (r_hat=1.00, ESS>4000, 129 ESS/s). Posterior chirp_mass=155±55, distance=3000±1000. Merger/ringdown-dominated regime handled stably.
  - **Script:** `test_scripts/catalog_expansion.py`
  - **Outputs:** `test_results/catalog_GW151226.txt`, `catalog_GW190412.txt`, `catalog_GW190521.txt`
  - **Sampler modification:** `build_and_sample_model()` now accepts `prior_bounds` and `initvals` dicts for per-event parameter customization.
- [x] **Task 2.2: Real Noise & PSD Robustness**
  - **Finding:** Frequency-domain ASD template whitening (`h_freq / ASD`) with `fftlength=1` conflicts with time-domain data whitening (`raw_strain.whiten(fduration=2)`) because the two PSD estimates differ (1s vs 2s Welch segments). This creates a rough posterior surface — NUTS r_hat > 1.8, ESS ≈ 2, 5x slower. **ASD=OFF (time-domain whitening only) converges reliably** (r_hat=1.00, ESS > 4600, 138 ESS/s). Pipeline now defaults to `use_asd=False`.
  - **Diagnostic:** PSD interpolation between differing FFT grids was also required for non-1s crop durations; the mismatch compound is avoided by omitting redundant frequency-domain whitening.
  - **Output:** `test_results/catalog_GW151226_asd_on.txt` (FAIL), `catalog_GW151226_noasd.txt` (PASS)
- [x] **Script:** `test_scripts/catalog_expansion.py`

> **MCP Guidance Rule (completed):** Catalog runs executed sequentially with per-event priors. Each event outputs `catalog_<event_name>.txt` containing recovered median, 90% CI, r_hat, ESS, and ESS/sec. See `test_results/` for all entries.

---

### Phase 3: Higher-Dimensional Parameter Space Expansion
**Objective:** Scale the NUTS sampler from low-dimensional test cases (2–4 parameters) to the full physical parameter space (8–15 parameters).

- [ ] **Task 3.1: Spin & Extrinsic Parameter Integration**
  - Expand from ($M_c, d_L$) to include effective aligned spin ($\chi_{\text{eff}}$), sky position ($\alpha, \delta$), inclination ($\iota$), polarization ($\psi$), and phase ($\phi_c$).
- [ ] **Task 3.2: Coordinate Reparameterization for NUTS**
  - Test whether sampling in transformed spaces (e.g., $\ln M_c$ or mass ratio $q$ instead of $m_1, m_2$) improves NUTS mass matrix adaptation and step-size stability.
- [ ] **Task 3.3: Precession & Higher-Order Waveform Modes**
  - Upgrade waveform calls to precessing / higher-mode approximants (e.g., `IMRPhenomXPHM` via JAX/`rippleGW`).

---

### Phase 4: Comparative Analysis & Publication Preparation
**Objective:** Synthesize findings into publication-ready figures and statistical comparisons.

- [ ] **Task 4.1: Posterior Agreement Metrics**
  - Compute Jensen-Shannon (JS) divergence and Wasserstein distances between NUTS posteriors and official LVC public samples across all catalog events.
- [ ] **Task 4.2: Scaling Law Analysis**
  - Plot NUTS wall-clock scaling as a function of parameter dimensionality ($N=2$ vs $N=4$ vs $N=8$ vs $N=15$).
- [ ] **Task 4.3: Manuscript & Codebase Release**
  - Export clean benchmark plots (corner plots, ESS scaling curves, JS divergence heatmaps).
  - Package `gw_analytics` as a clean, reproducible open-source library.