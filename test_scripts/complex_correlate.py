import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
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
    
    # 1. Generate h0 (phic = 0.0)
    params_0 = np.array([28.3, 0.247, 0.0, 0.0, 1.0, 0.0, 0.0])
    h0_freq_valid = gen_IMRPhenomD(freqs_valid, params_0, 20.0)
    h0_freq_full = np.concatenate([padding, np.array(h0_freq_valid)])
    h0_freq_whitened = h0_freq_full / safe_asd
    h0_freq_whitened_masked = np.where(in_band, h0_freq_whitened, 0.0 + 0.0j)
    h0_ref = np.fft.irfft(h0_freq_whitened_masked, n=N) / dt
    h0_normalized = h0_ref * np.sqrt(2.0 * dt)
    
    # 2. Generate h_pi_2 (phic = pi/2)
    params_pi_2 = np.array([28.3, 0.247, 0.0, 0.0, 1.0, 0.0, np.pi/2])
    hpi2_freq_valid = gen_IMRPhenomD(freqs_valid, params_pi_2, 20.0)
    hpi2_freq_full = np.concatenate([padding, np.array(hpi2_freq_valid)])
    hpi2_freq_whitened = hpi2_freq_full / safe_asd
    hpi2_freq_whitened_masked = np.where(in_band, hpi2_freq_whitened, 0.0 + 0.0j)
    hpi2_ref = np.fft.irfft(hpi2_freq_whitened_masked, n=N) / dt
    hpi2_normalized = hpi2_ref * np.sqrt(2.0 * dt)
    
    shifts = np.arange(-N//2, N//2)
    best_corr = -1.0
    best_shift = None
    best_phic = None
    
    for shift in shifts:
        # Roll the templates
        h0_rolled = np.roll(h0_normalized, shift)
        hpi2_rolled = np.roll(hpi2_normalized, shift)
        
        X = np.sum(observed * h0_rolled)
        Y = np.sum(observed * hpi2_rolled)
        
        corr_max = np.sqrt(X**2 + Y**2)
        
        if corr_max > best_corr:
            best_corr = corr_max
            best_shift = shift
            best_phic = np.arctan2(Y, X)
            
    # Compute physical parameters
    best_time_shift = best_shift * dt
    # If shift is negative, map it to positive equivalent
    best_tc = best_time_shift if best_time_shift >= 0 else (1.0 + best_time_shift)
    
    # Reconstruct the best fit template
    h0_best = np.roll(h0_normalized, best_shift)
    hpi2_best = np.roll(hpi2_normalized, best_shift)
    h_best = h0_best * np.cos(best_phic) + hpi2_best * np.sin(best_phic)
    
    # Analytical D optimization
    den = np.sum(h_best**2)
    alpha = best_corr / den
    best_d = 1.0 / alpha
    
    h_fit = h_best / best_d
    chi2 = np.sum((observed - h_fit)**2)
    
    print("\n--- Complex Cross-Correlation Results ---")
    print(f"Max correlation value (overlap sum): {best_corr:.4f}")
    print(f"Optimal shift index: {best_shift} samples")
    print(f"Optimal shift time: {best_time_shift:.6f} seconds")
    print(f"Optimal tc (within 0-1s window): {best_tc:.6f} seconds")
    print(f"Optimal phic: {best_phic:.4f} radians")
    print(f"Analytical Best-fit D: {best_d:.2f} Mpc (True ~410)")
    print(f"Chi2 at Best-fit parameters: {chi2:.4f} (expected ~4096)")
    
    # Let's save a detailed report of this complex analysis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    test_results_dir = os.path.join(parent_dir, "test_results")
    log_path = os.path.join(test_results_dir, "complex_correlation_results.txt")
    with open(log_path, "w") as f_out:
        f_out.write("COMPLEX CROSS-CORRELATION ANALYSIS REPORT\n")
        f_out.write("=" * 60 + "\n")
        f_out.write(f"Max correlation: {best_corr:.4e}\n")
        f_out.write(f"Optimal shift index: {best_shift} samples\n")
        f_out.write(f"Optimal shift time: {best_time_shift:.6f} s\n")
        f_out.write(f"Optimal tc (in window): {best_tc:.6f} s\n")
        f_out.write(f"Optimal phic: {best_phic:.4f} rad\n")
        f_out.write(f"Analytical Best-fit D: {best_d:.2f} Mpc\n")
        f_out.write(f"Chi2 at best-fit: {chi2:.4f}\n")

if __name__ == "__main__":
    analyze()
