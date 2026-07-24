import sys
import os
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

def diagnose():
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
    scaled_observed_strain = cropped_strain.value * 1e21
    
    N = len(time_array)
    dt = float(time_array[1] - time_array[0])
    freqs_np = np.fft.rfftfreq(N, d=dt)
    
    # Compute ASD
    print("Computing PSD/ASD from raw strain...")
    psd = raw_strain.psd(fftlength=1)
    asd = np.sqrt(psd.value)
    
    print(f"Data length: {N}, dt: {dt}, freqs length: {len(freqs_np)}")
    print(f"Observed strain (scaled) max: {np.max(np.abs(scaled_observed_strain)):.4e}")
    print(f"Observed strain (scaled) mean: {np.mean(scaled_observed_strain):.4e}")
    print(f"Observed strain (scaled) std: {np.std(scaled_observed_strain):.4e}")
    
    # Generate template at true parameters of GW150914
    # true parameters: Mc = 28.3, eta = 0.247, D = 410, tc = 0.5
    Mc_true = 28.3
    eta_true = 0.247
    D_true = 410.0
    tc_true = 0.5
    
    idx_start = int(np.searchsorted(freqs_np, 20.0))
    freqs_valid = freqs_np[idx_start:]
    
    params = np.array([Mc_true, eta_true, 0.0, 0.0, D_true, tc_true, 0.0])
    
    print("Generating unwhitened template...")
    h_freq_valid = gen_IMRPhenomD(jnp.array(freqs_valid), jnp.array(params), 20.0)
    
    padding = np.zeros(idx_start, dtype=np.complex128)
    h_freq_full = np.concatenate([padding, np.array(h_freq_valid)])
    
    # Check raw template amplitude
    print(f"Raw frequency template max amplitude: {np.max(np.abs(h_freq_full)):.4e}")
    
    # Transform unwhitened template to time domain
    h_time_unwhitened = np.fft.irfft(h_freq_full, n=N) / dt
    print(f"Unwhitened time-domain template (unscaled) max: {np.max(np.abs(h_time_unwhitened)):.4e}")
    print(f"Unwhitened time-domain template (scaled 1e21) max: {np.max(np.abs(h_time_unwhitened * 1e21)):.4e}")
    
    # Whiten template in frequency domain
    safe_asd = np.where(asd > 1e-30, asd, 1.0)
    h_freq_whitened = h_freq_full / safe_asd
    
    # Bandpass mask (20Hz to 1024Hz)
    in_band = (freqs_np >= 20.0) & (freqs_np <= 1024.0)
    h_freq_whitened_masked = np.where(in_band, h_freq_whitened, 0.0 + 0.0j)
    
    h_time_whitened = np.fft.irfft(h_freq_whitened_masked, n=N) / dt
    h_time_whitened_normalized = h_time_whitened * np.sqrt(2.0 * dt)
    print(f"Whitened time-domain template (unscaled) max: {np.max(np.abs(h_time_whitened)):.4e}")
    print(f"Whitened time-domain template (normalized with sqrt(2.0*dt)) max: {np.max(np.abs(h_time_whitened_normalized)):.4e}")
    
    # Let's write these diagnostics to a file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    test_results_dir = os.path.join(parent_dir, "test_results")
    log_path = os.path.join(test_results_dir, "whitening_diagnostics.txt")
    with open(log_path, "w") as f_out:
        f_out.write("WHITENING DIAGNOSTICS REPORT\n")
        f_out.write("=" * 60 + "\n")
        f_out.write(f"Observed strain (unscaled) max: {np.max(np.abs(cropped_strain.value)):.8e}\n")
        f_out.write(f"Observed strain (unscaled) std: {np.std(cropped_strain.value):.8e}\n\n")
        
        f_out.write(f"Unwhitened time template (unscaled) max: {np.max(np.abs(h_time_unwhitened)):.8e}\n\n")
        
        f_out.write(f"Whitened time template (unscaled, no sqrt) max: {np.max(np.abs(h_time_whitened)):.8e}\n")
        f_out.write(f"Whitened time template (normalized with sqrt(2.0*dt)) max: {np.max(np.abs(h_time_whitened_normalized)):.8e}\n")
        
        # Calculate chi2 mismatch at true parameters
        residual_no_sqrt = cropped_strain.value - h_time_whitened
        chi2_no_sqrt = np.sum(residual_no_sqrt**2)
        f_out.write(f"\nChi2 at True Parameters (unscaled, no sqrt): {chi2_no_sqrt:.8e}\n")
        
        residual_normalized = cropped_strain.value - h_time_whitened_normalized
        chi2_normalized = np.sum(residual_normalized**2)
        f_out.write(f"Chi2 at True Parameters (normalized with sqrt): {chi2_normalized:.8e}\n")

if __name__ == "__main__":
    diagnose()
