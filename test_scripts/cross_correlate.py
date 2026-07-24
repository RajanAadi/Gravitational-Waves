import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

def analyze():
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
    observed = cropped_strain.value
    
    N = len(time_array)
    dt = float(time_array[1] - time_array[0])
    freqs_np = np.fft.rfftfreq(N, d=dt)
    freqs_jax = jnp.array(freqs_np)
    
    # Compute ASD
    psd = raw_strain.psd(fftlength=1)
    asd = np.sqrt(psd.value)
    asd_jax = jnp.array(asd)
    safe_asd = jnp.where(asd_jax > 1e-30, asd_jax, 1.0)
    
    idx_start = int(np.searchsorted(freqs_np, 20.0))
    freqs_valid = freqs_jax[idx_start:]
    in_band = (freqs_np >= 20.0) & (freqs_np <= 1024.0)
    
    padding = np.zeros(idx_start, dtype=np.complex128)
    
    # Generate template at D = 1.0 Mpc, Mc = 28.3, tc = 0.0, phic = 0.0
    # Let's generate a template with tc = 0.0 and phic = 0.0
    params = np.array([28.3, 0.247, 0.0, 0.0, 1.0, 0.0, 0.0])
    h_freq_valid = gen_IMRPhenomD(freqs_valid, params, 20.0)
    h_freq_full = np.concatenate([padding, np.array(h_freq_valid)])
    
    # Whiten template
    h_freq_whitened = h_freq_full / safe_asd
    h_freq_whitened_masked = np.where(in_band, h_freq_whitened, 0.0 + 0.0j)
    
    # Transform to time domain
    h_ref = np.fft.irfft(h_freq_whitened_masked, n=N) / dt
    h_ref_normalized = h_ref * np.sqrt(2.0 * dt)
    
    # Perform cross-correlation using numpy.correlate (or FFT)
    # observed is d(t), h_ref_normalized is h(t) for D=1 Mpc
    # The correlation is c(tau) = sum_t d(t) * h(t + tau)
    # We want to find the shift tau that maximizes c(tau)
    
    # To handle circular boundaries correctly, let's use scipy or numpy correlate in 'same' mode or roll
    correlations = []
    shifts = np.arange(-N//2, N//2)
    
    for shift in shifts:
        # Roll the template to represent h(t - shift * dt)
        h_rolled = np.roll(h_ref_normalized, shift)
        corr = np.sum(observed * h_rolled)
        correlations.append(corr)
        
    correlations = np.array(correlations)
    
    # Find peak correlation
    peak_idx = np.argmax(correlations)
    peak_shift = shifts[peak_idx]
    peak_time_shift = peak_shift * dt
    peak_val = correlations[peak_idx]
    
    print("\n--- Cross-Correlation Results ---")
    print(f"Max correlation value (overlap sum): {peak_val:.4f}")
    print(f"Optimal shift index: {peak_shift} samples")
    print(f"Optimal shift time (tc): {peak_time_shift:.6f} seconds")
    
    # Calculate best-fit D and corresponding Chi2 at this optimal shift
    h_best = np.roll(h_ref_normalized, peak_shift)
    den = np.sum(h_best**2)
    alpha = peak_val / den
    best_d = 1.0 / alpha if alpha > 0 else 2000.0
    
    print(f"Analytical Best-fit D at peak: {best_d:.2f} Mpc")
    
    h_fit = h_best / best_d
    chi2 = np.sum((observed - h_fit)**2)
    print(f"Chi2 at Best-fit parameters: {chi2:.4f} (expected ~4096)")
    
    # Plotting and saving the cross-correlation to a text file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    test_results_dir = os.path.join(parent_dir, "test_results")
    log_path = os.path.join(test_results_dir, "correlation_results.txt")
    with open(log_path, "w") as f_out:
        f_out.write("CROSS-CORRELATION ANALYSIS REPORT\n")
        f_out.write("=" * 60 + "\n")
        f_out.write(f"Max correlation: {peak_val:.6e}\n")
        f_out.write(f"Optimal shift time: {peak_time_shift:.6f} seconds\n")
        f_out.write(f"Analytical Best-fit D: {best_d:.2f} Mpc\n")
        f_out.write(f"Chi2 at best-fit: {chi2:.4f}\n\n")
        
        # Save a portion of the correlation function around the peak
        f_out.write("Correlation function around peak:\n")
        f_out.write(f"{'Shift (s)':>10} | {'Correlation':>12}\n")
        f_out.write("-" * 25 + "\n")
        for idx in range(max(0, peak_idx - 10), min(len(shifts), peak_idx + 11)):
            f_out.write(f"{shifts[idx]*dt:10.6f} | {correlations[idx]:12.6f}\n")

if __name__ == "__main__":
    analyze()
