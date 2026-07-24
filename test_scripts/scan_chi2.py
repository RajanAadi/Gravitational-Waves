import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

def scan():
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
    
    # Compute ASD
    psd = raw_strain.psd(fftlength=1)
    asd = np.sqrt(psd.value)
    safe_asd = np.where(asd > 1e-30, asd, 1.0)
    
    idx_start = int(np.searchsorted(freqs_np, 20.0))
    freqs_valid = freqs_np[idx_start:]
    in_band = (freqs_np >= 20.0) & (freqs_np <= 1024.0)
    
    padding = np.zeros(idx_start, dtype=np.complex128)
    
    # Grid parameters
    mcs = np.linspace(25.0, 31.0, 13) # 25.0, 25.5, ..., 31.0
    tcs = np.linspace(0.40, 0.50, 101) # 0.400, 0.401, ..., 0.500
    
    print(f"Scanning Mc across {len(mcs)} values and tc across {len(tcs)} values...")
    
    best_chi2 = float('inf')
    best_mc = None
    best_tc = None
    best_d = None
    
    results = []
    
    # Let's fix D to various values and see the minimum Chi2
    for Mc_val in mcs:
        for tc_val in tcs:
            # We will also optimize D analytically!
            # Since the template h(t) scales as 1/D, we can write:
            # h(t) = (1/D) * h_ref(t) where h_ref(t) is computed with D = 1.0 Mpc.
            # Then we want to minimize: sum (d - (1/D) * h_ref)^2
            # Let alpha = 1/D. The minimum w.r.t alpha is:
            # alpha = sum(d * h_ref) / sum(h_ref^2)
            # D_opt = 1 / alpha
            
            # 1. Generate template at D = 1.0 Mpc
            params = np.array([Mc_val, 0.247, 0.0, 0.0, 1.0, tc_val, 0.0])
            h_freq_valid = gen_IMRPhenomD(jnp.array(freqs_valid), jnp.array(params), 20.0)
            h_freq_full = np.concatenate([padding, np.array(h_freq_valid)])
            
            # Whiten
            h_freq_whitened = h_freq_full / safe_asd
            h_freq_whitened_masked = np.where(in_band, h_freq_whitened, 0.0 + 0.0j)
            
            # Time domain
            h_ref = np.fft.irfft(h_freq_whitened_masked, n=N) / dt
            h_ref_normalized = h_ref * np.sqrt(2.0 * dt)
            
            # Analytical D optimization
            num = np.sum(observed * h_ref_normalized)
            den = np.sum(h_ref_normalized**2)
            
            if den > 1e-15:
                alpha = num / den
                # Ensure alpha is positive (otherwise signal is anti-correlated, physically we want positive amplitude)
                if alpha > 0:
                    D_opt = 1.0 / alpha
                    # Clamp D_opt to prior range [100, 2000]
                    D_opt = np.clip(D_opt, 100.0, 2000.0)
                else:
                    D_opt = 2000.0 # Force small amplitude
            else:
                D_opt = 2000.0
                
            # Compute Chi2 at this D_opt
            h_final = h_ref_normalized / D_opt
            chi2 = np.sum((observed - h_final)**2)
            
            results.append((Mc_val, tc_val, D_opt, chi2))
            
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_mc = Mc_val
                best_tc = tc_val
                best_d = D_opt
                
    # Print the summary
    print("\n--- Grid Search Results ---")
    print(f"Minimum Chi2: {best_chi2:.4f} (expected ~4096)")
    print(f"Best Chirp Mass: {best_mc:.2f} M_sun (True ~28.3)")
    print(f"Best tc: {best_tc:.4f} s (Centered at 0.5s)")
    print(f"Best Luminosity Distance: {best_d:.2f} Mpc (True ~410)")
    
    # Save a slice of the grid around the minimum
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    test_results_dir = os.path.join(parent_dir, "test_results")
    log_path = os.path.join(test_results_dir, "scan_results.txt")
    with open(log_path, "w") as f_out:
        f_out.write("CHI2 GRID SCAN REPORT\n")
        f_out.write("=" * 60 + "\n")
        f_out.write(f"Best parameters: Mc = {best_mc:.2f}, tc = {best_tc:.4f}, D = {best_d:.2f} | Chi2 = {best_chi2:.4f}\n\n")
        f_out.write("Sample results (sorted by Chi2):\n")
        f_out.write(f"{'Mc':>6} | {'tc':>6} | {'D_opt':>8} | {'Chi2':>10}\n")
        f_out.write("-" * 40 + "\n")
        sorted_results = sorted(results, key=lambda x: x[3])
        for r in sorted_results[:30]:
            f_out.write(f"{r[0]:6.2f} | {r[1]:6.4f} | {r[2]:8.2f} | {r[3]:10.4f}\n")

if __name__ == "__main__":
    scan()
