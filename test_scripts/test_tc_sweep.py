import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

def test_tc():
    N = 4096
    dt = 1.0 / 4096.0
    freqs_np = np.fft.rfftfreq(N, d=dt)
    
    idx_start = int(np.searchsorted(freqs_np, 20.0))
    freqs_valid = freqs_np[idx_start:]
    padding = np.zeros(idx_start, dtype=np.complex128)
    
    # We will test various values of tc
    test_tcs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -0.1, -0.2, -0.5]
    
    print(f"{'tc parameter':>12} | {'Peak Time (s)':>14} | {'Peak Index':>10}")
    print("-" * 45)
    
    for tc_val in test_tcs:
        params = np.array([28.3, 0.247, 0.0, 0.0, 410.0, tc_val, 0.0])
        h_freq_valid = gen_IMRPhenomD(jnp.array(freqs_valid), jnp.array(params), 20.0)
        h_freq_full = np.concatenate([padding, np.array(h_freq_valid)])
        
        # Transform to time domain
        h_time = np.fft.irfft(h_freq_full, n=N) / dt
        
        # Find the peak of the waveform envelope in the time domain
        peak_idx = np.argmax(np.abs(h_time))
        peak_time = peak_idx * dt
        
        print(f"{tc_val:12.2f} | {peak_time:14.4f} | {peak_idx:10d}")

if __name__ == "__main__":
    test_tc()
