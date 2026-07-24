import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
test_results_dir = os.path.join(parent_dir, "test_results")
sys.stdout = open(os.path.join(test_results_dir, "convergence_test.txt"), "w", encoding="utf-8")

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import arviz as az
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gw_analytics import GWNUTSSampler

def run_and_check_convergence():
    print("=" * 60)
    print("CONVERGENCE TEST: GW150914 NUTS Sampling Stability")
    print("=" * 60)

    gps_time = event_gps('GW150914')
    raw_strain = TimeSeries.fetch_open_data('H1', gps_time - 2.5, gps_time + 2.5, cache=True)
    whitened_strain = raw_strain.whiten(fduration=2)
    cropped_strain = whitened_strain.crop(gps_time - 0.5, gps_time + 0.5)

    time_array = cropped_strain.times.value - cropped_strain.times.value[0]
    scaled_strain = cropped_strain.value

    psd = raw_strain.psd(fftlength=1)
    asd_array = np.sqrt(psd.value)

    print(f"\nSampling parameters:")
    print(f"  chains = 2, draws = 3000, tune = 2000, target_accept = 0.95")

    nuts_engine = GWNUTSSampler(
        time_array=time_array,
        observed_strain=scaled_strain,
        noise_sigma=1.0,
        scale_factor=1.0,
        asd=asd_array,
    )

    inference_data = nuts_engine.build_and_sample_model(draws=3000, tune=2000, chains=2)

    az.rcParams["plot.backend"] = "matplotlib"
    summary = az.summary(inference_data, var_names=["chirp_mass", "luminosity_distance", "tc", "phic"], ci_prob=0.94)
    print("\n--- Final Parameter Estimates ---")
    print(summary)

    print("\n--- Per-Parameter r_hat ---")
    for param_name in ["chirp_mass", "luminosity_distance", "tc", "phic"]:
        if param_name in summary.index:
            r_val = float(summary.loc[param_name, "r_hat"])
            ess_val = float(summary.loc[param_name, "ess_bulk"])
            status = "OK" if r_val < 1.05 else "HIGH"
            print(f"  {param_name:20s}: r_hat = {r_val:.4f} ({status}), ESS_bulk = {ess_val:.0f}")

    r_hats = summary["r_hat"].dropna().astype(float)
    ess_bulk = summary["ess_bulk"].dropna().astype(float)

    print("\n--- Convergence Diagnostics ---")
    max_rhat = float(r_hats.max())
    min_ess_bulk = float(ess_bulk.min())
    print(f"Max r_hat: {max_rhat:.4f}")
    print(f"Min ess_bulk: {min_ess_bulk:.1f}")

    n_div = inference_data.sample_stats.get("divergences", None)
    if n_div is not None and hasattr(n_div, "values"):
        total_div = int(n_div.values.sum())
        print(f"Total divergences: {total_div}")
    else:
        total_div = 0
        print("Total divergences: 0 (no divergence data)")

    all_pass = True
    if max_rhat > 1.10:
        print(f"FAIL: r_hat > 1.10 ({max_rhat:.4f})")
        all_pass = False
    else:
        print(f"PASS: r_hat <= 1.10 ({max_rhat:.4f})")

    if min_ess_bulk < 100:
        print(f"FAIL: ess_bulk < 100 ({min_ess_bulk:.1f})")
        all_pass = False
    else:
        print(f"PASS: ess_bulk >= 100 ({min_ess_bulk:.1f})")

    if total_div > 0:
        print(f"FAIL: {total_div} divergences detected")
        all_pass = False
    else:
        print(f"PASS: 0 divergences")

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 60}")

    return inference_data, summary, all_pass

if __name__ == "__main__":
    run_and_check_convergence()
