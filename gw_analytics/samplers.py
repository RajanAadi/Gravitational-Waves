# ============================================================
# Steps 2–6: JAX-Differentiable Waveform Integration
# ============================================================
#
# Step 2: Enable float64 globally for JAX.
# LIGO strain numbers are ~1e-21; float32 (7 sig-figs) causes
# catastrophic cancellation in the phase integrals.
# This MUST be set before any jax or ripplegw import.
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pymc.sampling.jax as pm_jax
import pytensor.tensor as pt

# Step 5: @as_jax_op bridges a pure JAX function into PyTensor's
# symbolic graph so that PyMC's gradient engine can differentiate
# through it end-to-end.
from pytensor import wrap_jax

# Step 4: The actual relativistic waveform model (verified API).
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

# Physical reference frequency [Hz]: lowest frequency in the
# sensitive LIGO band. ripple uses this to anchor the phase.
F_LOWER = 20.0
F_UPPER = 1024.0


# ============================================================
# Step 4: Pure JAX waveform function
# ============================================================
def _make_jax_waveform(freqs_jax: jnp.ndarray, N: int, dt: float, scale_factor: float, noise_asd_jax: jnp.ndarray = None):
    """
    Returns a closure `jax_waveform(Mc, D, tc, phic)` that avoids low-frequency 
    singularities by only passing valid, sorted frequencies to the engine,
    then padding the outputs dynamically. Optionally performs frequency-domain
    whitening if noise_asd_jax is provided.
    """
    # Convert to numpy temporarily to calculate the static slice index on the CPU
    freqs_np = np.array(freqs_jax)
    idx_start = int(np.searchsorted(freqs_np, F_LOWER))
    
    # Create a strictly monotonic, safe frequency array starting exactly at F_LOWER
    freqs_valid = jnp.array(freqs_np[idx_start:])
    
    # Full-sized in-band mask for the final output array
    in_band = (freqs_jax >= F_LOWER) & (freqs_jax <= F_UPPER)

    @jax.jit
    def jax_waveform(Mc, D, tc, phic):
        # ripple params: [Mchirp, eta, chi1, chi2, D_mpc, tc, phic]
        # Use 0.25 - 1e-12 instead of exactly 0.25 to prevent division-by-zero inside
        # ripplegw's mass conversion function (Mc_eta_to_ms) during backpropagation.
        params = jnp.array([Mc, 0.25 - 1e-12, 0.0, 0.0, D, tc, phic])

        # 1. Compute the waveform ONLY on the physically safe, sorted frequency grid
        h_freq_valid = gen_IMRPhenomD(freqs_valid, params, F_LOWER)

        # 2. Differentiably pad the front with complex zeros for the 0 to F_LOWER bins
        padding = jnp.zeros(idx_start, dtype=jnp.complex128)
        h_freq_full = jnp.concatenate([padding, h_freq_valid])

        # 3. Whiten the template in the frequency domain if ASD is provided
        if noise_asd_jax is not None:
            # Prevent division by zero.
            safe_asd = jnp.where(noise_asd_jax > 1e-30, noise_asd_jax, 1.0)
            h_freq_whitened = h_freq_full / safe_asd
            h_freq_masked = jnp.where(in_band, h_freq_whitened, 0.0 + 0.0j)
        else:
            h_freq_masked = jnp.where(in_band, h_freq_full, 0.0 + 0.0j)

        # 4. Transform to real time-domain strain
        h_time = jnp.fft.irfft(h_freq_masked, n=N) / dt

        # Apply discrete-to-continuous normalization factor for whitening
        if noise_asd_jax is not None:
            h_time = h_time * jnp.sqrt(2.0 * dt)

        return h_time * scale_factor

    return jax_waveform


# ============================================================
# Step 5: PyTensor Op via @as_jax_op
# ============================================================
def _make_pytensor_op(time_array: np.ndarray, scale_factor: float, asd: np.ndarray = None):
    """
    Builds and returns a PyTensor-compatible waveform Op backed by JAX.

    @as_jax_op wraps the JAX function so that:
      - PyMC's symbolic graph can include it as a node.
      - Gradients flow through it automatically via JAX autodiff
        (used by the NUTS sampler to compute the Hamiltonian gradient).

    Args:
        time_array:   1D numpy array of time samples from the data.
        scale_factor: Multiplicative factor applied to the template
                      (matching the 1e21 scaling applied to observed data).
        asd:          Optional 1D numpy array containing the noise ASD.
    Returns:
        A callable PyTensor Op accepting scalar (Mc, D, tc, phic) PyTensor vars.
    """
    N = len(time_array)
    dt = float(time_array[1] - time_array[0])
    freqs_np = np.fft.rfftfreq(N, d=dt)
    freqs_jax = jnp.array(freqs_np)
    asd_jax = jnp.array(asd) if asd is not None else None

    # Build the pure JAX waveform closure
    _jax_fn = _make_jax_waveform(freqs_jax, N, dt, scale_factor, asd_jax)

    @wrap_jax
    def waveform_op(Mc, D, tc, phic):
        return _jax_fn(Mc, D, tc, phic)

    return waveform_op


# ============================================================
# Steps 3 & 6: Updated GWNUTSSampler
# ============================================================
class GWNUTSSampler:
    # Step 3: __init__ was previously missing, causing AttributeError
    # when the notebook tried to access self.time_array etc.
    def __init__(
        self,
        time_array: np.ndarray,
        observed_strain: np.ndarray,
        noise_sigma: float = 1.0,
        scale_factor: float = 1e21,
        asd: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        time_array     : 1D array of time stamps [seconds], zero-referenced.
        observed_strain: 1D array of whitened, scaled strain values.
        noise_sigma    : Noise standard deviation of the whitened data.
                         After whitening sigma=1.0; after 1e21 scaling still 1.0
                         because both data and template are scaled equally.
        scale_factor   : Applied to both data (externally) and template
                         (internally) to prevent float64 gradient underflow.
                         Must match the factor applied to observed_strain.
        asd            : Optional 1D array containing the noise ASD.
        """
        self.time_array = np.asarray(time_array, dtype=np.float64)
        self.observed_strain = np.asarray(observed_strain, dtype=np.float64)
        self.noise_sigma = float(noise_sigma)
        self.scale_factor = float(scale_factor)
        self.asd = np.asarray(asd, dtype=np.float64) if asd is not None else None
        self.idata = None

    # Step 6: Rebuilt model using the JAX-differentiable waveform Op.
    def build_and_sample_model(
        self,
        draws: int = 3000,
        tune: int = 2000,
        chains: int = 2,
        prior_bounds: dict = None,
        initvals: dict = None,
    ):
        """
        Construct the PyMC model with an IMRPhenomD waveform template
        and sample using NumPyro's NUTS (JAX-compiled, gradient-aware).

        Parameters
        ----------
        draws : int
            Number of posterior draws per chain.
        tune : int
            Number of tuning (warmup) steps per chain.
        chains : int
            Number of independent MCMC chains.
        prior_bounds : dict, optional
            Custom prior bounds keyed by parameter name.
            Each value is a (lower, upper) tuple.
            Defaults to GW150914-appropriate bounds.
        initvals : dict, optional
            Custom initial values keyed by parameter name.
            Defaults to GW150914-appropriate initvals.

        Returns
        -------
        idata : ArviZ InferenceData object with posterior samples.
        """
        if prior_bounds is None:
            prior_bounds = {}

        if initvals is None:
            initvals = {}

        # Build the static PyTensor Op once (frequency grid computed here)
        waveform_op = _make_pytensor_op(self.time_array, self.scale_factor, self.asd)

        with pm.Model() as model:
            # Chirp mass
            lb, ub = prior_bounds.get("chirp_mass", (28.0, 35.0))
            chirp_mass = pm.Uniform("chirp_mass", lower=lb, upper=ub)

            # Luminosity distance
            lb, ub = prior_bounds.get("luminosity_distance", (600.0, 1200.0))
            lum_dist = pm.Uniform("luminosity_distance", lower=lb, upper=ub)

            # Time of coalescence
            lb, ub = prior_bounds.get("tc", (0.525, 0.532))
            tc = pm.Uniform("tc", lower=lb, upper=ub)

            # Phase of coalescence: full periodic range (-π, π)
            phic = pm.Uniform("phic", lower=-np.pi, upper=np.pi)

            # JAX-differentiable relativistic waveform template
            scaled_template = waveform_op(chirp_mass, lum_dist, tc, phic)

            # Likelihood: whitened strain ~ N(template, sigma=1)
            pm.Normal(
                "likelihood",
                mu=scaled_template,
                sigma=self.noise_sigma,
                observed=self.observed_strain,
            )

            # Default initvals for each parameter
            default_initvals = {
                "chirp_mass": 31.5,
                "luminosity_distance": 900.0,
                "tc": 0.5285,
                "phic": 0.0,
            }
            default_initvals.update(initvals)

            self.idata = pm_jax.sample_numpyro_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,
                progressbar=True,
                initvals=default_initvals,
            )

        return self.idata