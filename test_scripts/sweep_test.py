import os
import sys

# Redirect standard output to a plain text file in the test_results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "sweep_test.txt"), "w", encoding="utf-8")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

# Parameters
N = 4096
dt = 1.0 / 4096.0
freqs_np = np.fft.rfftfreq(N, d=dt)
freqs_jax = jnp.array(freqs_np)
F_LOWER = 20.0

print("=== SWEEP TEST 1: MASKING ONLY (HIGH FREQS STILL EVALUATED IN RIIPLEGW) ===")
for f_up in [2048, 1000, 500, 300, 150, 100, 75, 50, 40, 30, 25]:
    idx_start = int(np.searchsorted(freqs_np, F_LOWER))
    freqs_valid = jnp.array(freqs_np[idx_start:])
    in_band = (freqs_jax >= F_LOWER) & (freqs_jax <= float(f_up))

    def test_waveform_masking(Mc, D, tc):
        params = jnp.array([Mc, 0.25, 0.0, 0.0, D, tc, 0.0])
        h_freq_valid = gen_IMRPhenomD(freqs_valid, params, F_LOWER)
        padding = jnp.zeros(idx_start, dtype=jnp.complex128)
        h_freq_full = jnp.concatenate([padding, h_freq_valid])
        h_freq_masked = jnp.where(in_band, h_freq_full, 0.0 + 0.0j)
        return jnp.fft.irfft(h_freq_masked, n=N) / dt

    def objective1(Mc, D, tc):
        return jnp.sum(test_waveform_masking(Mc, D, tc))

    grad_fn1 = jax.grad(objective1, argnums=(0, 1, 2))
    try:
        grads = grad_fn1(30.0, 400.0, 0.5)
        print(f"F_UPPER = {f_up:4d} Hz | grads: dMc={grads[0]}, dD={grads[1]}, dtc={grads[2]}")
    except Exception as e:
        print(f"F_UPPER = {f_up:4d} Hz | error: {e}")

print("\n=== SWEEP TEST 2: STRICT SLICING (HIGH FREQS EXCLUDED FROM RIPPLEGW) ===")
for f_up in [2048, 1000, 500, 300, 150, 100, 75, 50, 40, 30, 25]:
    idx_start = int(np.searchsorted(freqs_np, F_LOWER))
    idx_end = int(np.searchsorted(freqs_np, float(f_up)))
    
    # Slice the valid frequencies strictly between F_LOWER and F_UPPER
    freqs_valid = jnp.array(freqs_np[idx_start:idx_end])
    
    def test_waveform_slicing(Mc, D, tc):
        params = jnp.array([Mc, 0.25, 0.0, 0.0, D, tc, 0.0])
        h_freq_valid = gen_IMRPhenomD(freqs_valid, params, F_LOWER)
        
        # Pad below F_LOWER and above F_UPPER
        pad_low = jnp.zeros(idx_start, dtype=jnp.complex128)
        pad_high = jnp.zeros(len(freqs_np) - idx_end, dtype=jnp.complex128)
        
        h_freq_full = jnp.concatenate([pad_low, h_freq_valid, pad_high])
        return jnp.fft.irfft(h_freq_full, n=N) / dt

    def objective2(Mc, D, tc):
        return jnp.sum(test_waveform_slicing(Mc, D, tc))

    grad_fn2 = jax.grad(objective2, argnums=(0, 1, 2))
    try:
        grads = grad_fn2(30.0, 400.0, 0.5)
        print(f"F_UPPER = {f_up:4d} Hz | grads: dMc={grads[0]}, dD={grads[1]}, dtc={grads[2]}")
    except Exception as e:
        print(f"F_UPPER = {f_up:4d} Hz | error: {e}")
