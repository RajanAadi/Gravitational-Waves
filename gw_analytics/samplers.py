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
def _make_jax_waveform(freqs_jax: jnp.ndarray, N: int, dt: float, scale_factor: float):
    """
    Returns a closure `jax_waveform(Mc, D, tc)` that is a pure JAX
    function. Closing over the static frequency grid and array length
    avoids retracing the JAX JIT on every call.

    The function:
      1. Builds the 7-element ripple params vector for a face-on,
         equal-mass, non-spinning binary (eta=0.25, chi1=chi2=0, phic=0).
      2. Calls gen_IMRPhenomD to get the complex frequency-domain strain.
      3. Zeros out frequencies outside [F_LOWER, F_UPPER] so the
         likelihood only sees in-band physics.
      4. Converts to a real time-domain template via irfft.
      5. Applies the scale_factor to match the pre-scaled observed data.
    """
    # In-band boolean mask — computed once at construction time.
    in_band = (freqs_jax >= F_LOWER) & (freqs_jax <= F_UPPER)

    @jax.jit
    def jax_waveform(Mc, D, tc):
        # ripple params: [Mchirp, eta, chi1, chi2, D_mpc, tc, phic]
        # Fixing eta=0.25 (equal-mass), non-spinning, phic=0.
        params = jnp.array([Mc, 0.25, 0.0, 0.0, D, tc, 0.0])

        # Generate complex frequency-domain strain h(f)
        h_freq = gen_IMRPhenomD(freqs_jax, params, F_LOWER)

        # Zero out contributions outside the sensitive band
        h_freq_masked = jnp.where(in_band, h_freq, 0.0 + 0.0j)

        # irfft: complex freq-domain → real time-domain, length N
        h_time = jnp.fft.irfft(h_freq_masked, n=N) / dt

        # Apply scale factor to match the 1e21-scaled observed data
        return h_time * scale_factor

    return jax_waveform


# ============================================================
# Step 5: PyTensor Op via @as_jax_op
# ============================================================
def _make_pytensor_op(time_array: np.ndarray, scale_factor: float):
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
    Returns:
        A callable PyTensor Op accepting scalar (Mc, D, tc) PyTensor vars.
    """
    N = len(time_array)
    dt = float(time_array[1] - time_array[0])
    freqs_np = np.fft.rfftfreq(N, d=dt)
    freqs_jax = jnp.array(freqs_np)

    # Build the pure JAX waveform closure
    _jax_fn = _make_jax_waveform(freqs_jax, N, dt, scale_factor)

    @wrap_jax
    def waveform_op(Mc, D, tc):
        return _jax_fn(Mc, D, tc)

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
        """
        self.time_array = np.asarray(time_array, dtype=np.float64)
        self.observed_strain = np.asarray(observed_strain, dtype=np.float64)
        self.noise_sigma = float(noise_sigma)
        self.scale_factor = float(scale_factor)
        self.idata = None

    # Step 6: Rebuilt model using the JAX-differentiable waveform Op.
    def build_and_sample_model(self, draws: int = 1500, tune: int = 1000, chains: int = 2):
        """
        Construct the PyMC model with an IMRPhenomD waveform template
        and sample using NumPyro's NUTS (JAX-compiled, gradient-aware).

        Returns
        -------
        idata : ArviZ InferenceData object with posterior samples.
        """
        # Build the static PyTensor Op once (frequency grid computed here)
        waveform_op = _make_pytensor_op(self.time_array, self.scale_factor)

        with pm.Model() as model:
            # ----------------------------------------------------------
            # Priors
            # ----------------------------------------------------------
            # Chirp mass: GW150914 true value ~28.3 M☉
            chirp_mass = pm.Uniform("chirp_mass", lower=10.0, upper=50.0)

            # Luminosity distance: GW150914 true value ~410 Mpc
            lum_dist = pm.Uniform("luminosity_distance", lower=100.0, upper=2000.0)

            # Time of coalescence: kept narrow around the known merger time.
            # The whitened/cropped data is zero-referenced so tc ~ 0.5 s
            # puts the merger near the centre of the 1-second window.
            tc = pm.Uniform("tc", lower=0.4, upper=0.6)

            # ----------------------------------------------------------
            # Step 6: JAX-differentiable relativistic waveform template
            # Replaces the toy algebraic sine wave entirely.
            # ----------------------------------------------------------
            scaled_template = waveform_op(chirp_mass, lum_dist, tc)

            # ----------------------------------------------------------
            # Likelihood: whitened strain ~ N(template, sigma=1)
            # ----------------------------------------------------------
            pm.Normal(
                "likelihood",
                mu=scaled_template,
                sigma=self.noise_sigma,
                observed=self.observed_strain,
            )

            # ----------------------------------------------------------
            # Step 6: NumPyro NUTS — compiles the full model to JAX/XLA.
            # This replaces pm.sample() and uses automatic differentiation
            # through the ripplegw waveform for gradient computation.
            # ----------------------------------------------------------
            self.idata = pm_jax.sample_numpyro_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.90,
                progressbar=True,
            )

        return self.idata