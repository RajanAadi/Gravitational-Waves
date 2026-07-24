import sys
import os

# Redirect standard output to append to sampler_test.txt in the test_results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "sampler_test.txt"), "a", encoding="utf-8")

# Find the parent directory (GravWaves root) and add it to Python's search path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import arviz as az
import corner
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gw_analytics import GWNUTSSampler

def run_event_sampler(event_name, detector='H1', duration=1.0):
    print(f"Fetching GPS metadata for {event_name}...")
    try:
        gps_time = event_gps(event_name)
    except ValueError:
        print(f"Error: Could not find event.")
        return None
        
    print(f"Downloading and whitening strain data for {detector}...")
    raw_strain = TimeSeries.fetch_open_data(detector, gps_time - 2.5, gps_time + 2.5, cache=True)
    whitened_strain = raw_strain.whiten(fduration=2)
    cropped_strain = whitened_strain.crop(gps_time - (duration/2), gps_time + (duration/2))
    
    time_array = cropped_strain.times.value - cropped_strain.times.value[0]
    raw_strain_array = cropped_strain.value
    
    # --- FIX 1 & 2: SCALING AND NOISE ---
    # Since the data and template are both fully whitened, their physical strain values
    # are normalized to units of noise standard deviation (O(1)). Therefore, we can
    # safely use a SCALE_FACTOR of 1.0 without any risk of gradient underflow.
    SCALE_FACTOR = 1.0
    scaled_strain = raw_strain_array * SCALE_FACTOR
    
    # Because we whitened the data, the background noise standard deviation is exactly 1
    # (Since we scaled the data, we will scale the template to match, so sigma stays 1.0)
    noise_sigma = 1.0 
    
    # Compute the noise ASD to whiten the template in the frequency domain.
    # The PSD is calculated with fftlength=1 second to match the 1.0s cropped duration and frequency spacing (df = 1.0 Hz)
    print("Computing noise PSD and ASD...")
    psd = raw_strain.psd(fftlength=1)
    asd_array = np.sqrt(psd.value)
    
    print(f"Initializing NUTS engine for {event_name} with template whitening...")
    nuts_engine = GWNUTSSampler(
        time_array=time_array,
        observed_strain=scaled_strain,
        noise_sigma=noise_sigma,
        scale_factor=1.0,
        asd=asd_array
    )
    
    inference_data = nuts_engine.build_and_sample_model(draws=1500, tune=1000, chains=2)
    
    return inference_data

# 1. Run the function on the Gold Standard event
event_data = run_event_sampler('GW150914', detector='H1', duration=1.0)

# 2. Lock ArviZ's global configuration to strictly use Matplotlib
az.rcParams["plot.backend"] = "matplotlib"

# 3. Print the statistics
summary = az.summary(event_data, var_names=["chirp_mass", "luminosity_distance", "tc", "phic"], ci_prob=0.94)
print("\n--- Final Parameter Estimates ---")
print(summary)

# Extract the NUTS posterior samples for your parameters
posterior_samples = event_data.posterior
mass_samples = posterior_samples["chirp_mass"].values.flatten()
distance_samples = posterior_samples["luminosity_distance"].values.flatten()

# Stack them into a 2D array for the corner plot
samples_2d = np.vstack([mass_samples, distance_samples]).T

# Generate the publication-ready Corner Plot
fig = corner.corner(
    samples_2d, 
    labels=[r"Chirp Mass ($M_\odot$)", r"Luminosity Distance (Mpc)"],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    quantiles=[0.03, 0.5, 0.97], # Highlights your 94% HDI bounds
    color="#0072C1" # LIGO standard blue
)

plt.savefig(os.path.join(test_results_dir, "corner_plot.png"), dpi=300)
plt.close()