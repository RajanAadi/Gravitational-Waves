import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "bic_aic_diagnosis.txt"), "a", encoding="utf-8")

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import jax.numpy as jnp
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gw_analytics.samplers import GWNUTSSampler, _make_pytensor_op


def compute_bic_aic(event_name, detector='H1', duration=1.0):
    print(f"Fetching GPS metadata for {event_name}...")
    gps_time = event_gps(event_name)

    print(f"Downloading and whitening strain data for {detector}...")
    raw_strain = TimeSeries.fetch_open_data(detector, gps_time - 2.5, gps_time + 2.5, cache=True)
    whitened_strain = raw_strain.whiten(fduration=2)
    cropped_strain = whitened_strain.crop(gps_time - (duration/2), gps_time + (duration/2))

    time_array = cropped_strain.times.value - cropped_strain.times.value[0]
    raw_strain_array = cropped_strain.value

    print("Computing noise PSD and ASD...")
    psd = raw_strain.psd(fftlength=1)
    asd_array = np.sqrt(psd.value)

    nuts_engine = GWNUTSSampler(
        time_array=time_array,
        observed_strain=raw_strain_array,
        noise_sigma=1.0,
        scale_factor=1.0,
        asd=asd_array,
    )

    inference_data = nuts_engine.build_and_sample_model(draws=1500, tune=1000, chains=4)

    summary = az.summary(inference_data, var_names=["chirp_mass", "luminosity_distance", "tc", "phic"])
    print("\n--- Posterior Summary ---")
    print(summary)

    posterior_means = {
        "chirp_mass": float(summary.loc["chirp_mass", "mean"]),
        "luminosity_distance": float(summary.loc["luminosity_distance", "mean"]),
        "tc": float(summary.loc["tc", "mean"]),
        "phic": float(summary.loc["phic", "mean"]),
    }

    print("\n--- BIC / AIC Computation ---")
    n = len(time_array)
    k = 4

    waveform_op = _make_pytensor_op(time_array, 1.0, asd_array)

    Mc_pt = pt.constant(jnp.array(posterior_means["chirp_mass"]))
    D_pt = pt.constant(jnp.array(posterior_means["luminosity_distance"]))
    tc_pt = pt.constant(jnp.array(posterior_means["tc"]))
    phic_pt = pt.constant(jnp.array(posterior_means["phic"]))

    template_at_mean = waveform_op(Mc_pt, D_pt, tc_pt, phic_pt).eval()

    residuals = raw_strain_array - template_at_mean
    sigma = 1.0
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2) / sigma**2

    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    print(f"Number of data points (n): {n}")
    print(f"Number of parameters (k):  {k}")
    print(f"Log-likelihood at posterior mean: {log_likelihood:.1f}")
    print(f"AIC:  {aic:.1f}")
    print(f"BIC:  {bic:.1f}")
    print(f"chi2 (sum of squared residuals): {np.sum(residuals**2):.2f}")
    print(f"Reduced chi2 (chi2 / (n - k)): {np.sum(residuals**2) / (n - k):.4f}")

    return inference_data, {"aic": aic, "bic": bic, "log_likelihood": log_likelihood, "chi2": np.sum(residuals**2)}


if __name__ == "__main__":
    data, metrics = compute_bic_aic('GW150914', detector='H1', duration=1.0)
    print("\n========================================")
    print("BIC/AIC diagnostics complete.")
    print(f"Results saved to test_results/bic_aic_diagnosis.txt")
