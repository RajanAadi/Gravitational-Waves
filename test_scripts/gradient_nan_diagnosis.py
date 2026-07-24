import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "gradient_nan_diagnosis.txt"), "w", encoding="utf-8")

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

print("=" * 60)
print("ISSUE #1: Mc_eta_to_ms Gradient NaN Diagnosis")
print("=" * 60)

N = 4096
dt = 1.0 / 4096.0
freqs_np = np.fft.rfftfreq(N, d=dt)
idx_start = int(np.searchsorted(freqs_np, 20.0))
freqs_valid = jnp.array(freqs_np[idx_start:])

def test_eta_value(eta_val, label):
    print(f"\n--- Testing eta = {eta_val} ({label}) ---")

    def f_masses_sum(Mc):
        m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta_val]))
        return m1 + m2

    def f_wrap(Mc_val):
        return Mc_eta_to_ms(jnp.array([Mc_val, eta_val]))
    m1, m2 = f_wrap(30.0)
    print(f"  m1, m2 = {m1}, {m2}")
    try:
        dm1_dMc = jax.grad(lambda Mc: f_wrap(Mc)[0])(30.0)
        dm2_dMc = jax.grad(lambda Mc: f_wrap(Mc)[1])(30.0)
        has_nan = jnp.isnan(dm1_dMc) or jnp.isnan(dm2_dMc)
        print(f"  d(m1)/dMc = {dm1_dMc}, d(m2)/dMc = {dm2_dMc}")
        print(f"  Has NaN: {has_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")
        has_nan = True
    if not has_nan:
        def f_waveform_real(Mc, D, tc, phic):
            params = jnp.array([Mc, eta_val, 0.0, 0.0, D, tc, phic])
            h = gen_IMRPhenomD(freqs_valid, params, 20.0)
            return jnp.sum(jnp.abs(h) ** 2)
        grad_wf = jax.grad(f_waveform_real, argnums=(0, 1, 2, 3))
        grads = grad_wf(30.0, 400.0, 0.5, 0.0)
        ok = all(not np.isnan(float(g)) for g in grads)
        print(f"  Full waveform gradients: dMc={float(grads[0]):.2f}, dD={float(grads[1]):.2f}, dtc={float(grads[2]):.2e}, dphic={float(grads[3]):.2e}")
        print(f"  All finite: {ok}")

test_eta_value(0.25, "EXACT equal-mass (buggy)")
test_eta_value(0.25 - 1e-12, "OFFSET equal-mass (workaround)")

print("\n" + "=" * 60)
print("ROOT CAUSE: Mc_eta_to_ms has a division by (1 - 4*eta)")
print("At eta=0.25, sqrt(1 - 4*eta) = sqrt(0) = 0 -> division by zero")
print("The fix: use eta = 0.25 - 1e-12 instead of eta = 0.25")
print("=" * 60)
