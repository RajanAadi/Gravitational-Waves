import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD
from scipy.optimize import minimize

def optimize_gw150914_with_phase():
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
    observed = jnp.array(cropped_strain.value)
    
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
    in_band = (freqs_jax >= 20.0) & (freqs_jax <= 1024.0)
    
    padding = jnp.zeros(idx_start, dtype=jnp.complex128)
    
    @jax.jit
    def compute_chi2(params_to_opt):
        # params_to_opt: [Mc, D, tc, phic]
        Mc_val, D_val, tc_val, phic_val = params_to_opt
        
        # ripple params: [Mchirp, eta, chi1, chi2, D_mpc, tc, phic]
        params = jnp.array([Mc_val, 0.247, 0.0, 0.0, D_val, tc_val, phic_val])
        h_freq_valid = gen_IMRPhenomD(freqs_valid, params, 20.0)
        h_freq_full = jnp.concatenate([padding, h_freq_valid])
        
        # Whiten
        h_freq_whitened = h_freq_full / safe_asd
        h_freq_whitened_masked = jnp.where(in_band, h_freq_whitened, 0.0 + 0.0j)
        
        # Time domain
        h_time = jnp.fft.irfft(h_freq_whitened_masked, n=N) / dt
        h_time_normalized = h_time * jnp.sqrt(2.0 * dt)
        
        # Chi2
        residual = observed - h_time_normalized
        return jnp.sum(residual**2)

    # Let's define the value and grad function for optimization
    loss_and_grad = jax.value_and_grad(compute_chi2)
    
    def scipy_objective(x):
        val, grad = loss_and_grad(x)
        return float(val), np.array(grad, dtype=np.float64)

    # Initial guesses: try different tc and phic combinations
    initial_tcs = [0.43, 0.44, 0.45]
    initial_phics = [0.0, np.pi/2, np.pi, -np.pi/2]
    
    best_res = None
    best_val = float('inf')
    
    print("\nRunning L-BFGS-B optimization with phase (phic) parameter...")
    for init_tc in initial_tcs:
        for init_phic in initial_phics:
            init_guess = np.array([28.3, 410.0, init_tc, init_phic])
            bounds = [(20.0, 40.0), (100.0, 2000.0), (0.4, 0.6), (-np.pi, np.pi)]
            
            res = minimize(
                scipy_objective,
                init_guess,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 200}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
                print(f"New Best! Chi2: {res.fun:.4f} | Mc: {res.x[0]:.2f}, D: {res.x[1]:.2f}, tc: {res.x[2]:.4f}, phic: {res.x[3]:.4f}")
            
    print("\n--- Optimization Summary (With Phase) ---")
    print(f"Minimum Chi2: {best_res.fun:.4f} (expected ~4096)")
    print(f"Optimized Chirp Mass: {best_res.x[0]:.4f} M_sun")
    print(f"Optimized Luminosity Distance: {best_res.x[1]:.4f} Mpc")
    print(f"Optimized tc: {best_res.x[2]:.6f} s")
    print(f"Optimized phic: {best_res.x[3]:.4f} rad")
    print(f"Scipy optimization message: {best_res.message}")

if __name__ == "__main__":
    optimize_gw150914_with_phase()
