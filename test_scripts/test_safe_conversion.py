import os
import sys

# Redirect standard output to a plain text file in the test_results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "test_safe_conversion.txt"), "w", encoding="utf-8")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from ripplegw.waveforms.IMRPhenomD import _gen_IMRPhenomD
from ripplegw.waveforms.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
from ripplegw.constants import MTSUN

# Parameters
N = 4096
dt = 1.0 / 4096.0
freqs_np = np.fft.rfftfreq(N, d=dt)
F_LOWER = 20.0
f_ref = F_LOWER

idx_start = int(np.searchsorted(freqs_np, F_LOWER))
f = jnp.array(freqs_np[idx_start:])

Mc = 30.0
eta = 0.25

def safe_Mc_eta_to_ms(m):
    Mchirp, eta_val = m
    M = Mchirp / (eta_val ** (3.0 / 5.0))
    # Use 1e-14/1e-15 epsilon inside square root to prevent division by zero in the derivative of sqrt
    val = M**2 - 4 * M**2 * eta_val
    m2 = (M - jnp.sqrt(jnp.maximum(val, 1e-14))) / 2.0
    m1 = M - m2
    return m1, m2

def check_grads():
    print("Testing safe mass conversion...")
    
    # 1. Conversion to m1, m2
    def get_intrinsic_safe(Mc_val):
        m1, m2 = safe_Mc_eta_to_ms(jnp.array([Mc_val, eta]))
        return jnp.array([m1, m2, 0.0, 0.0])
    
    intrinsic = get_intrinsic_safe(Mc)
    print("m1, m2, chi1, chi2:", intrinsic)
    grad_intrinsic = jax.jacobian(get_intrinsic_safe)(Mc)
    print("grad_intrinsic (d(m1,m2)/dMc):", grad_intrinsic)
    
    # Let's test the entire waveform gradient with safe conversion
    def safe_gen_IMRPhenomD(freqs, Mc_val, D, tc):
        m1, m2 = safe_Mc_eta_to_ms(jnp.array([Mc_val, eta]))
        theta_intrinsic = jnp.array([m1, m2, 0.0, 0.0])
        theta_extrinsic = jnp.array([D, tc, 0.0])
        coeffs = get_coeffs(theta_intrinsic)
        h0 = _gen_IMRPhenomD(freqs, theta_intrinsic, theta_extrinsic, coeffs, f_ref)
        return h0

    def objective(Mc_val, D, tc):
        h0 = safe_gen_IMRPhenomD(f, Mc_val, D, tc)
        return jnp.sum(h0.real)

    grad_fn = jax.grad(objective, argnums=(0, 1, 2))
    grads = grad_fn(Mc, 400.0, 0.5)
    print(f"Gradients with safe mass conversion [dMc, dD, dtc]:\n{grads}")

check_grads()
