import os
import sys

# Redirect standard output to a plain text file in the test_results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "diagnose_intermediate_grads.txt"), "w", encoding="utf-8")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD, get_IIb_raw_phase, Phase, Amp, _gen_IMRPhenomD
from ripplegw.waveforms.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
from ripplegw.conversions import Mc_eta_to_ms
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
params = jnp.array([Mc, eta, 0.0, 0.0, 400.0, 0.5, 0.0])

def check_step_grads():
    print("Evaluating step-by-step gradients with respect to Mc...")
    
    # 1. Conversion to m1, m2
    def get_intrinsic(Mc_val):
        m1, m2 = Mc_eta_to_ms(jnp.array([Mc_val, eta]))
        return jnp.array([m1, m2, 0.0, 0.0])
    
    print("Mc_val:", Mc)
    intrinsic = get_intrinsic(Mc)
    print("m1, m2, chi1, chi2:", intrinsic)
    grad_intrinsic = jax.jacobian(get_intrinsic)(Mc)
    print("grad_intrinsic (d(m1,m2)/dMc):", grad_intrinsic)
    
    # 2. Coeffs
    def get_coeffs_from_Mc(Mc_val):
        theta_intrinsic = get_intrinsic(Mc_val)
        return get_coeffs(theta_intrinsic)
    
    coeffs = get_coeffs_from_Mc(Mc)
    grad_coeffs = jax.jacobian(get_coeffs_from_Mc)(Mc)
    print("grad_coeffs has NaN:", jnp.isnan(grad_coeffs).any())
    
    # 3. Transition frequencies
    def get_transition_freqs_from_Mc(Mc_val):
        theta_intrinsic = get_intrinsic(Mc_val)
        coeffs = get_coeffs(theta_intrinsic)
        return get_transition_frequencies(theta_intrinsic, coeffs[5], coeffs[6])
    
    trans_freqs = get_transition_freqs_from_Mc(Mc)
    print("Transition freqs:", trans_freqs)
    grad_trans_freqs = jax.jacobian(get_transition_freqs_from_Mc)(Mc)
    print("grad_trans_freqs (d(f_trans)/dMc):", grad_trans_freqs)
    
    # 4. t0
    def get_t0_from_Mc(Mc_val):
        theta_intrinsic = get_intrinsic(Mc_val)
        coeffs = get_coeffs(theta_intrinsic)
        transition_freqs = get_transition_frequencies(theta_intrinsic, coeffs[5], coeffs[6])
        _, _, _, f4, f_RD, f_damp = transition_freqs
        M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * MTSUN
        # Note: t0 is computed as derivative of get_IIb_raw_phase w.r.t first arg (fM_s) at f4 * M_s
        # Let's see if this is differentiable w.r.t theta_intrinsic and coeffs
        t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta_intrinsic, coeffs, f_RD, f_damp)
        return t0
        
    t0 = get_t0_from_Mc(Mc)
    print("t0:", t0)
    grad_t0 = jax.grad(get_t0_from_Mc)(Mc)
    print("grad_t0 (dt0/dMc):", grad_t0)
    
    # 5. Psi (Phase at frequencies f)
    def get_Psi_from_Mc(Mc_val):
        theta_intrinsic = get_intrinsic(Mc_val)
        coeffs = get_coeffs(theta_intrinsic)
        transition_freqs = get_transition_frequencies(theta_intrinsic, coeffs[5], coeffs[6])
        return Phase(f, theta_intrinsic, coeffs, transition_freqs)
        
    Psi = get_Psi_from_Mc(Mc)
    grad_Psi = jax.jacobian(get_Psi_from_Mc)(Mc)
    print("grad_Psi has NaN:", jnp.isnan(grad_Psi).any())
    
    # 6. Psi_ref (Phase at f_ref)
    def get_Psi_ref_from_Mc(Mc_val):
        theta_intrinsic = get_intrinsic(Mc_val)
        coeffs = get_coeffs(theta_intrinsic)
        transition_freqs = get_transition_frequencies(theta_intrinsic, coeffs[5], coeffs[6])
        return Phase(f_ref, theta_intrinsic, coeffs, transition_freqs)
        
    Psi_ref = get_Psi_ref_from_Mc(Mc)
    grad_Psi_ref = jax.grad(get_Psi_ref_from_Mc)(Mc)
    print("grad_Psi_ref (dPsi_ref/dMc):", grad_Psi_ref)
    
    # 7. Total Phase (Psi - t0 * ((f * M_s) - Mf_ref) - Psi_ref)
    def get_total_phase_from_Mc(Mc_val):
        theta_intrinsic = get_intrinsic(Mc_val)
        coeffs = get_coeffs(theta_intrinsic)
        transition_freqs = get_transition_frequencies(theta_intrinsic, coeffs[5], coeffs[6])
        M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * MTSUN
        _, _, _, f4, f_RD, f_damp = transition_freqs
        t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta_intrinsic, coeffs, f_RD, f_damp)
        Psi = Phase(f, theta_intrinsic, coeffs, transition_freqs)
        Mf_ref = f_ref * M_s
        Psi_ref = Phase(f_ref, theta_intrinsic, coeffs, transition_freqs)
        Psi_shifted = Psi - (t0 * ((f * M_s) - Mf_ref) + Psi_ref)
        return Psi_shifted
        
    Psi_total = get_total_phase_from_Mc(Mc)
    grad_Psi_total = jax.jacobian(get_total_phase_from_Mc)(Mc)
    print("grad_Psi_total has NaN:", jnp.isnan(grad_Psi_total).any())
    if jnp.isnan(grad_Psi_total).any():
        # Find index of first NaN
        nan_indices = jnp.where(jnp.isnan(grad_Psi_total))[0]
        print("First NaN in grad_Psi_total is at index:", nan_indices[0], "freq:", f[nan_indices[0]])

check_step_grads()
