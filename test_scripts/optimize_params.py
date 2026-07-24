import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD
from scipy.optimize import minimize

def optimize_gw150914():
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
        # params_to_opt: [Mc, D, tc]
        Mc_val, D_val, tc_val = params_to_opt
        
        # ripple params: [Mchirp, eta, chi1, chi2, D_mpc, tc, phic]
        # Using 0.247 for eta (true value for GW150914)
        params = jnp.array([Mc_val, 0.247, 0.0, 0.0, D_val, tc_val, 0.0])
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
        # Convert JAX arrays to standard float64 numpy arrays for scipy
        return float(val), np.array(grad, dtype=np.float64)

    # Initial guess: close to true parameters
    # Mc = 28.3, D = 410.0, tc = 0.43
    # Let's try multiple initial guesses for tc since it's highly multi-modal due to phase cycles
    initial_tcs = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47]
    best_res = None
    best_val = float('inf')
    
    print("\nRunning L-BFGS-B optimization for multiple initial tc guesses...")
    for init_tc in initial_tcs:
        init_guess = np.array([28.3, 410.0, init_tc])
        bounds = [(10.0, 50.0), (100.0, 2000.0), (0.4, 0.6)]
        
        res = minimize(
            scipy_objective,
            init_guess,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 200}
        )
        
        print(f"Initial tc: {init_tc:.2f} | Final Chi2: {res.fun:.4f} | Mc: {res.x[0]:.2f}, D: {res.x[1]:.2f}, tc: {res.x[2]:.4f}")
        
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
    print("\n--- Optimization Summary ---")
    print(f"Minimum Chi2: {best_res.fun:.4f} (expected ~4096)")
    print(f"Optimized Chirp Mass: {best_res.x[0]:.4f} M_sun")
    print(f"Optimized Luminosity Distance: {best_res.x[1]:.4f} Mpc")
    print(f"Optimized tc: {best_res.x[2]:.6f} s")
    print(f"Scipy optimization message: {best_res.message}")

if __name__ == "__main__":
    optimize_gw150914()
