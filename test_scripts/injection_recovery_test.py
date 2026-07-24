import sys, os, time

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "injection_recovery_results.txt"), "w", encoding="utf-8")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
from gw_analytics.samplers import GWNUTSSampler, _make_jax_waveform

jax.config.update("jax_enable_x64", True)
az.rcParams["plot.backend"] = "matplotlib"

F_LOWER = 20.0
F_UPPER = 1024.0
N = 4096
DURATION = 1.0
FS = N / DURATION
dt = 1.0 / FS

GRID_QS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_DRAWS = 2000
N_TUNE = 1500
N_CHAINS = 2

INJECTED = {
    "chirp_mass": 30.0,
    "luminosity_distance": 870.0,
    "tc": 0.5295,
    "phic": 0.0,
}


def eta_from_q(q):
    return q / (1.0 + q) ** 2


def generate_injection_waveform(Mc, eta, D, tc, phic):
    freqs_np = np.fft.rfftfreq(N, d=dt)
    freqs_jax = jnp.array(freqs_np)
    idx_start = int(np.searchsorted(freqs_np, F_LOWER))
    freqs_valid = jnp.array(freqs_np[idx_start:])
    in_band = (freqs_jax >= F_LOWER) & (freqs_jax <= F_UPPER)

    params = jnp.array([Mc, eta, 0.0, 0.0, D, tc, phic])
    h_freq_valid = gen_IMRPhenomD(freqs_valid, params, F_LOWER)
    padding = jnp.zeros(idx_start, dtype=jnp.complex128)
    h_freq_full = jnp.concatenate([padding, h_freq_valid])
    h_freq_masked = jnp.where(in_band, h_freq_full, 0.0 + 0.0j)
    h_time = jnp.fft.irfft(h_freq_masked, n=N) / dt
    return np.asarray(h_time)


def run_injection_test(eta_true):
    """Inject a signal with given eta and see if NUTS recovers Mc."""
    q = eta_true_to_q(eta_true)
    template = generate_injection_waveform(
        INJECTED["chirp_mass"], eta_true,
        INJECTED["luminosity_distance"], INJECTED["tc"], INJECTED["phic"],
    )

    t_arr = np.arange(N) * dt

    engine = GWNUTSSampler(
        time_array=t_arr, observed_strain=template,
        noise_sigma=1.0, scale_factor=1.0, asd=None,
    )

    t0 = time.time()
    idata = engine.build_and_sample_model(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS)
    elapsed = time.time() - t0

    summary = az.summary(idata, var_names=["chirp_mass", "luminosity_distance", "tc", "phic"])
    mc_mean = float(summary.loc["chirp_mass", "mean"])
    mc_sd = float(summary.loc["chirp_mass", "sd"])
    mc_rhat = float(summary.loc["chirp_mass", "r_hat"])
    mc_ess = float(summary.loc["chirp_mass", "ess_bulk"])
    lum_mean = float(summary.loc["luminosity_distance", "mean"])
    lum_sd = float(summary.loc["luminosity_distance", "sd"])

    bias_mc = mc_mean - INJECTED["chirp_mass"]
    bias_lum = lum_mean - INJECTED["luminosity_distance"]

    return {
        "eta": eta_true,
        "q": q,
        "Mc_rec": mc_mean,
        "Mc_sd": mc_sd,
        "Mc_bias": bias_mc,
        "Mc_rhat": mc_rhat,
        "Mc_ess": int(mc_ess),
        "lum_rec": lum_mean,
        "lum_sd": lum_sd,
        "lum_bias": bias_lum,
        "time_s": round(elapsed, 1),
    }


def eta_true_to_q(eta):
    """Solve q / (1+q)^2 = eta for q in [0,1]."""
    disc = 1.0 - 4.0 * eta
    if disc < 0:
        return 1.0
    return (1.0 - np.sqrt(disc)) / (1.0 + np.sqrt(disc))


def print_report(results):
    print("=" * 90)
    print("ZERO-NOISE INJECTION-RECOVERY TEST — Mass Ratio Grid")
    print("=" * 90)
    print(f"Fixed injected values: Mc={INJECTED['chirp_mass']}, D={INJECTED['luminosity_distance']}, "
          f"tc={INJECTED['tc']}, phic={INJECTED['phic']}")
    print(f"Model: fixed eta=0.25 (equal-mass assumption)")
    print(f"Sampler: PyMC-NUTS, {N_DRAWS} draws + {N_TUNE} tune, {N_CHAINS} chains")
    print()
    header = f"{'q':<8} {'eta':<8} {'Mc_rec':<10} {'Mc_sd':<8} {'Mc_bias':<10} {'r_hat':<8} {'ESS':<8} {'D_rec':<10} {'D_bias':<10} {'time_s':<8}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['q']:<8.2f} {r['eta']:<8.4f} {r['Mc_rec']:<10.2f} {r['Mc_sd']:<8.2f} "
              f"{r['Mc_bias']:<+10.2f} {r['Mc_rhat']:<8.2f} {r['Mc_ess']:<8} "
              f"{r['lum_rec']:<10.1f} {r['lum_bias']:<+10.1f} {r['time_s']:<8}")
    print("-" * 90)

    all_pass = all(abs(r["Mc_bias"]) < 3.0 and r["Mc_rhat"] < 1.10 for r in results)
    print(f"\n  Criteria: |Mc_bias| < 3.0 AND r_hat < 1.10 for all q")
    print(f"  OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

    results = []
    for q in GRID_QS:
        eta = eta_from_q(q)
        print(f"\n  Running q={q:.1f} (eta={eta:.4f})...", end=" ", flush=True)
        try:
            r = run_injection_test(eta)
            results.append(r)
            print(f"Mc_rec={r['Mc_rec']:.2f}, bias={r['Mc_bias']:+.2f}, r_hat={r['Mc_rhat']:.2f}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\n")
    print_report(results)
    print("\nResults saved to test_results/injection_recovery_results.txt")
