# Project Context: Gravitational Wave Parameter Estimation with PyMC & NUTS

## The Goal
Estimate the parameters (chirp mass and luminosity distance) of the GW150914 black hole merger using a No-U-Turn Sampler (NUTS) pipeline via PyMC, utilizing whitened LIGO data from GWOSC.

## Current Repository Setup
- Located at: `~/Documents/GravWaves`
- Contains: `gw_analytics.py`, Jupyter Notebook, and environment configurations.

## Troubleshooting History & Lessons Learned
1. **ArviZ/gwpy Backend Crash:** `az.plot_trace()` was crashing because `gwpy` injected its own plotting engine into the global environment, causing ArviZ to look for a non-existent `gwpy` backend. 
2. **Prior Boundary Pile-up:** When using a basic mathematical sine-wave template, the NUTS sampler completely rejected the data and fled to the absolute boundaries of the Uniform priors (Mass = 10/50, Distance = 100/2000). 
3. **The Physics Reality:** An algebraic toy equation cannot match the non-linear, relativistic phase evolution of real General Relativity data. The likelihood heavily penalizes phase mismatches, forcing the sampler to run to the corners to minimize the mathematical penalty.

## Active Code Architecture (GWNUTSSampler Fixes Injected)
- **Data Scaling:** Raw strain data and theoretical templates are multiplied by `1e21` to prevent floating-point gradient underflow.
- **Noise Hardcoding:** Because the data is whitened, background noise standard deviation (`sigma`) is locked to `1.0`.
- **Time Alignment:** A `tc` (time of coalescence) uniform prior parameter was added to allow phase shifting.

## Next Steps
We are completely abandoning the toy algebraic waveform equation. The next phase of development is to integrate a genuinely relativistic, differentiable waveform engine using either:
- **Option A:** A custom PyTensor wrapper (`Op`) using finite differences over legacy `LALSuite` C-bindings.
- **Option B (Preferred):** A modern, natively differentiable JAX-based waveform library like `ripples` connected via a JAX-PyMC bridge.

---

## JAX Waveform Integration Plan (Steps 1–6)

**Option B was chosen.** Dependencies installed: `jax==0.11.0`, `jaxlib==0.11.0`, `rippleGW==0.2.1`, `numpyro==0.21.0`.

### Verified ripplegw 0.2.1 API (sourced directly from installed package)
- **Python module name:** `ripplegw` (NOT `ripple`)
- **Correct function:** `gen_IMRPhenomD` from `ripplegw.waveforms.IMRPhenomD`
- **Signature:** `gen_IMRPhenomD(f, params, f_ref)` → complex frequency-domain strain `h0`
- **`params` array — 7 elements in order:**
  1. `Mchirp` — Chirp mass [solar masses]
  2. `eta` — Symmetric mass ratio (0 < eta ≤ 0.25; equal mass → 0.25)
  3. `chi1` — Aligned spin of primary [-1, 1]
  4. `chi2` — Aligned spin of secondary [-1, 1]
  5. `D` — Luminosity distance [Mpc]
  6. `tc` — Time of coalescence [seconds] (applied as linear phase shift)
  7. `phic` — Phase at coalescence [radians]
- `gen_IMRPhenomD_hphc(f, params, f_ref)` takes an 8-element array (adds `inclination`) and returns `(hp, hc)`.

### Implementation Steps

**Step 1 (DONE):** Install `jax`, `jaxlib`, `rippleGW`, `numpyro` into venv; update `requirements.txt`.

**Step 2:** Enable JAX float64 at the top of `samplers.py`:
```python
from jax import config
config.update("jax_enable_x64", True)
```

**Step 3:** Add missing `__init__` constructor to `GWNUTSSampler` storing `time_array`, `observed_strain`, `noise_sigma`, `scale_factor`.

**Step 4:** Define `_jax_waveform(Mc, eta, D, tc, freqs, N)` — a pure JAX function that calls `gen_IMRPhenomD`, zero-pads frequencies outside the sensitive band, and returns the irfft time-domain template.

**Step 5:** Wrap it with `@as_jax_op` (from `pytensor.link.jax`) so PyMC can differentiate through it. Frequency grid and array length are closed over as static constants.

**Step 6:** Replace the toy waveform math in `build_and_sample_model` with the PyTensor Op. Switch sampler from `pm.sample()` to `pymc.sampling.jax.sample_numpyro_nuts()` for end-to-end JAX/XLA compilation.

### Design Decisions Requiring User Input (Stop Point)
After this implementation the pipeline requires two physical choices before running on real GW150914 data:

1. **eta prior:** Currently fixing `eta = 0.25` (equal-mass simplification). GW150914 true values are m1≈36, m2≈29 M☉ → eta≈0.247, so this is valid. Alternatively, infer eta with `pm.Uniform("eta", 0.1, 0.25)` for a more complete posterior.
2. **Frequency band:** The irfft template is computed over the full rfft grid. The whitened LIGO data is only sensitive in ~20–1024 Hz. The likelihood should mask or zero out frequencies outside this band on both data and template to avoid fitting to out-of-band noise.