import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

def run_isolation():
    N = 4096
    dt = 1.0 / 4096.0
    freqs_np = np.fft.rfftfreq(N, d=dt)
    freqs_jax = jnp.array(freqs_np)
    scale_factor = 1e21

    # Open isolation_results.txt in the test_results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    test_results_dir = os.path.join(parent_dir, "test_results")
    results_path = os.path.join(test_results_dir, "isolation_results.txt")
    
    with open(results_path, "w") as f_out:
        f_out.write("Frequency (Hz) | Forward Max | dMc Value | Is NaN?\n")
        f_out.write("-" * 55 + "\n")
        
        # Systematically sweep F_UPPER from 2048 down to 32 Hz in steps of 10 Hz
        frequencies = list(range(2048, 30, -10))
        # Add smaller 1 Hz steps near transition zones (from 50 down to 32)
        frequencies.extend(list(range(50, 31, -1)))
        # Sort them descending and remove duplicates
        frequencies = sorted(list(set(frequencies)), reverse=True)
        
        for f_up in frequencies:
            idx_start = int(np.searchsorted(freqs_np, 20.0))
            freqs_valid = jnp.array(freqs_np[idx_start:])
            in_band = (freqs_jax >= 20.0) & (freqs_jax <= float(f_up))
            
            # Target objective to evaluate jax.grad of
            def objective(Mc, D, tc):
                params = jnp.array([Mc, 0.25, 0.0, 0.0, D, tc, 0.0])
                h_freq_valid = gen_IMRPhenomD(freqs_valid, params, 20.0)
                padding = jnp.zeros(idx_start, dtype=jnp.complex128)
                h_freq_full = jnp.concatenate([padding, h_freq_valid])
                h_freq_masked = jnp.where(in_band, h_freq_full, 0.0 + 0.0j)
                h_time = jnp.fft.irfft(h_freq_masked, n=N) / dt
                return jnp.sum(h_time * scale_factor)
                
            grad_fn = jax.grad(objective, argnums=0)
            
            # Forward call to get the waveform max
            def get_fwd_max(Mc, D, tc):
                params = jnp.array([Mc, 0.25, 0.0, 0.0, D, tc, 0.0])
                h_freq_valid = gen_IMRPhenomD(freqs_valid, params, 20.0)
                padding = jnp.zeros(idx_start, dtype=jnp.complex128)
                h_freq_full = jnp.concatenate([padding, h_freq_valid])
                h_freq_masked = jnp.where(in_band, h_freq_full, 0.0 + 0.0j)
                h_time = jnp.fft.irfft(h_freq_masked, n=N) / dt
                return jnp.max(jnp.abs(h_time * scale_factor))
            
            try:
                fwd_max = get_fwd_max(30.0, 400.0, 0.5)
                dMc = grad_fn(30.0, 400.0, 0.5)
                is_nan = np.isnan(dMc)
                f_out.write(f"{f_up:14d} | {fwd_max:11.4e} | {dMc:9.4e} | {is_nan}\n")
            except Exception as e:
                f_out.write(f"{f_up:14d} | Error: {e}\n")

if __name__ == "__main__":
    run_isolation()
