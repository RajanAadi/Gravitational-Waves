import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "test_pymc_likelihood.txt"), "w", encoding="utf-8")

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pymc as pm
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gw_analytics.samplers import GWNUTSSampler

def test_likelihood_with_phic():
    event_name = 'GW150914'
    detector = 'H1'
    duration = 1.0

    print("Fetching GPS metadata...")
    gps_time = event_gps(event_name)

    print("Downloading and whitening strain data...")
    raw_strain = TimeSeries.fetch_open_data(detector, gps_time - 2.5, gps_time + 2.5, cache=True)
    whitened_strain = raw_strain.whiten(fduration=2)
    cropped_strain = whitened_strain.crop(gps_time - (duration/2), gps_time + (duration/2))

    time_array = cropped_strain.times.value - cropped_strain.times.value[0]
    scaled_strain = cropped_strain.value

    print("Computing noise PSD and ASD...")
    psd = raw_strain.psd(fftlength=1)
    asd_array = np.sqrt(psd.value)

    print("Initializing GWNUTSSampler...")
    sampler = GWNUTSSampler(
        time_array=time_array,
        observed_strain=scaled_strain,
        noise_sigma=1.0,
        scale_factor=1.0,
        asd=asd_array
    )

    waveform_op = sampler.build_and_sample_model.__globals__['_make_pytensor_op'](
        sampler.time_array, sampler.scale_factor, sampler.asd
    )

    with pm.Model() as model:
        chirp_mass = pm.Uniform("chirp_mass", lower=20.0, upper=35.0)
        lum_dist = pm.Uniform("luminosity_distance", lower=200.0, upper=1500.0)
        tc = pm.Uniform("tc", lower=0.51, upper=0.55)
        phic = pm.Uniform("phic", lower=-np.pi, upper=np.pi)

        scaled_template = waveform_op(chirp_mass, lum_dist, tc, phic)

        pm.Normal(
            "likelihood",
            mu=scaled_template,
            sigma=sampler.noise_sigma,
            observed=sampler.observed_strain,
        )

    logp = model.compile_logp()

    def transform(val, lower, upper):
        return float(np.log((val - lower) / (upper - val)))

    def to_chi2(lp, N):
        return -2 * lp - N * np.log(2 * np.pi)

    N = len(time_array)

    print("\n--- Compiled PyMC Log-Likelihood Evaluation (with phic) ---")

    pt_noise = {
        "chirp_mass_interval__": transform(28.3, 20.0, 35.0),
        "luminosity_distance_interval__": transform(1499.0, 200.0, 1500.0),
        "tc_interval__": transform(0.53, 0.51, 0.55),
        "phic_interval__": transform(0.0, -np.pi, np.pi),
    }
    lp_noise = logp(pt_noise)
    print(f"Noise-only (D~1500, Mc=28.3, tc=0.53, phic=0): LogL = {lp_noise:.4f}, Chi2 = {to_chi2(lp_noise, N):.2f}")

    pt_runaway = {
        "chirp_mass_interval__": transform(25.6, 20.0, 35.0),
        "luminosity_distance_interval__": transform(1499.0, 200.0, 1500.0),
        "tc_interval__": transform(0.49, 0.51, 0.55),
        "phic_interval__": transform(0.0, -np.pi, np.pi),
    }
    lp_runaway = logp(pt_runaway)
    print(f"Runaway    (D~1500, Mc=25.6, tc=0.49, phic=0): LogL = {lp_runaway:.4f}, Chi2 = {to_chi2(lp_runaway, N):.2f}")

    pt_physical = {
        "chirp_mass_interval__": transform(28.3, 20.0, 35.0),
        "luminosity_distance_interval__": transform(871.0, 200.0, 1500.0),
        "tc_interval__": transform(0.5295, 0.51, 0.55),
        "phic_interval__": transform(-0.7854, -np.pi, np.pi),
    }
    lp_physical = logp(pt_physical)
    print(f"Physical   (D=871, Mc=28.3, tc=0.5295, phic=-0.785): LogL = {lp_physical:.4f}, Chi2 = {to_chi2(lp_physical, N):.2f}")

    pt_physical_phic0 = {
        "chirp_mass_interval__": transform(28.3, 20.0, 35.0),
        "luminosity_distance_interval__": transform(871.0, 200.0, 1500.0),
        "tc_interval__": transform(0.5295, 0.51, 0.55),
        "phic_interval__": transform(0.0, -np.pi, np.pi),
    }
    lp_physical_phic0 = logp(pt_physical_phic0)
    print(f"Physical/phic=0 (D=871, Mc=28.3, tc=0.5295):     LogL = {lp_physical_phic0:.4f}, Chi2 = {to_chi2(lp_physical_phic0, N):.2f}")

    pt_optimal = {
        "chirp_mass_interval__": transform(31.7, 20.0, 35.0),
        "luminosity_distance_interval__": transform(814.0, 200.0, 1500.0),
        "tc_interval__": transform(0.5281, 0.51, 0.55),
        "phic_interval__": transform(2.0, -np.pi, np.pi),
    }
    lp_optimal = logp(pt_optimal)
    print(f"Optimal    (D=814, Mc=31.7, tc=0.5281, phic=2.0): LogL = {lp_optimal:.4f}, Chi2 = {to_chi2(lp_optimal, N):.2f}")

    print("\n--- Summary ---")
    print(f"Physical mode is {lp_physical - lp_noise:.1f} nats more probable than noise-only")
    print(f"Optimal (sampler) mode is {lp_optimal - lp_noise:.1f} nats more probable than noise-only")

if __name__ == "__main__":
    test_likelihood_with_phic()
