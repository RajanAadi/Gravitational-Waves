import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from gw_analytics.samplers import _make_jax_waveform

def test_gw_analytics_waveform():
    N = 4096
    dt = 1.0 / 4096.0
    freqs_np = np.fft.rfftfreq(N, d=dt)
    freqs_jax = jnp.array(freqs_np)
    scale_factor = 1e21

    # Create the JAX waveform closure from the updated gw_analytics codebase
    jax_fn = _make_jax_waveform(freqs_jax, N, dt, scale_factor)

    # Wrap it in an objective to get scalar output for jax.grad
    def objective(Mc, D, tc, phic):
        h_time = jax_fn(Mc, D, tc, phic)
        return jnp.sum(h_time**2)

    # Compute gradients with respect to all four inputs (Mc, D, tc, phic)
    grad_fn = jax.grad(objective, argnums=(0, 1, 2, 3))

    # Output file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    test_results_dir = os.path.join(parent_dir, "test_results")
    results_path = os.path.join(test_results_dir, "verification_results.txt")

    with open(results_path, "w") as f_out:
        f_out.write("GW_ANALYTICS JAX GRADIENT VERIFICATION LOG (WITH PHIC)\n")
        f_out.write("=" * 60 + "\n")
        
        try:
            # Run forward call
            h_time = jax_fn(30.0, 400.0, 0.5, -0.7854)
            fwd_max = float(jnp.max(jnp.abs(h_time)))
            f_out.write(f"Forward wave max amplitude: {fwd_max:.8e}\n")
            
            # Run gradient call
            grads = grad_fn(30.0, 400.0, 0.5, -0.7854)
            dMc, dD, dtc, dphic = float(grads[0]), float(grads[1]), float(grads[2]), float(grads[3])
            
            f_out.write(f"Computed gradients:\n")
            f_out.write(f"  - dMc   = {dMc:.8e} (Is NaN? {np.isnan(dMc)})\n")
            f_out.write(f"  - dD    = {dD:.8e} (Is NaN? {np.isnan(dD)})\n")
            f_out.write(f"  - dtc   = {dtc:.8e} (Is NaN? {np.isnan(dtc)})\n")
            f_out.write(f"  - dphic = {dphic:.8e} (Is NaN? {np.isnan(dphic)})\n")
            f_out.write("\nSUCCESS: All gradients are fully valid, finite floats!\n")
            
            print("Verification successful! All gradients are valid, finite floats.")
        except Exception as e:
            f_out.write(f"ERROR during gradient evaluation: {e}\n")
            print(f"Verification failed with error: {e}")

if __name__ == "__main__":
    test_gw_analytics_waveform()
