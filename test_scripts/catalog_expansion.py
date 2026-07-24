"""
Phase 2: GWOSC Event Catalog Expansion

Tasks:
  2.1 — Mass-Regime Testing on GW151226, GW190412, GW190521
  2.2 — Real Noise & PSD Robustness (ASD-on vs ASD-off comparison)
"""

import sys, os, time, json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import arviz as az
az.rcParams["plot.backend"] = "matplotlib"

# gwpy subclasses matplotlib Axes; monkey-patch arviz_plots to remap gwpy->matplotlib
import arviz_plots.plot_collection as _pc
import arviz_plots.visuals as _visuals
_orig_bfo = _pc.backend_from_object
def _bfo_patch(obj, return_module=True):
    b = _orig_bfo(obj, return_module=False)
    if b == "gwpy":
        b = "matplotlib"
    if return_module:
        return __import__(f"arviz_plots.backend.{b}", fromlist=[""])
    return b
_pc.backend_from_object = _bfo_patch
_visuals.backend_from_object = _bfo_patch

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gw_analytics import GWNUTSSampler

OUTPUT_DIR = os.path.join(parent_dir, "test_results")

N_DRAWS = 2000
N_TUNE = 1500
N_CHAINS = 2


def fetch_and_prepare(event_name, detector='H1', duration=1.0):
    """Fetch and whiten strain data from GWOSC for a given event."""
    gps_time = event_gps(event_name)
    raw_strain = TimeSeries.fetch_open_data(
        detector, gps_time - 2.5, gps_time + 2.5, cache=True
    )
    whitened_strain = raw_strain.whiten(fduration=2)
    cropped_strain = whitened_strain.crop(
        gps_time - duration / 2, gps_time + duration / 2
    )

    time_array = cropped_strain.times.value - cropped_strain.times.value[0]
    scaled_strain = cropped_strain.value

    # Compute PSD on the longer raw data for a better noise estimate,
    # then interpolate onto the template's frequency grid.
    psd_long = raw_strain.psd(fftlength=1)
    asd_long = np.sqrt(psd_long.value)
    dt = float(cropped_strain.times.value[1] - cropped_strain.times.value[0])
    freqs_tmpl = np.fft.rfftfreq(len(cropped_strain), d=dt)
    freqs_psd = np.fft.rfftfreq(int(1.0 / dt), d=dt)
    from scipy import interpolate
    f_interp = interpolate.interp1d(
        freqs_psd, asd_long, kind="linear",
        bounds_error=False, fill_value=asd_long[-1],
    )
    asd_array = f_interp(freqs_tmpl)

    return {
        "gps_time": gps_time,
        "time_array": time_array,
        "scaled_strain": scaled_strain,
        "asd_array": asd_array,
        "duration": duration,
        "tc_center": duration / 2,
    }


def run_event(event_name, detector, duration, prior_bounds, initvals,
              use_asd=False, label=""):
    """Run the NUTS sampler on a single event.

    Note: use_asd=False by default. The data is already time-domain whitened
    via raw_strain.whiten(fduration=2). Applying a second frequency-domain
    ASD division (computed with fftlength=1) uses a different PSD estimate,
    creating a likelihood surface mismatch that degrades NUTS convergence.
    """
    """Run the NUTS sampler on a single event and return (summary, elapsed, idata)."""
    print(f"\n{'='*70}")
    print(f"  {label or event_name}  (detector={detector}, duration={duration}s)")
    print(f"{'='*70}")

    data = fetch_and_prepare(event_name, detector=detector, duration=duration)
    tc_center = data["tc_center"]

    # Override tc bounds with per-event values
    if "tc" not in prior_bounds:
        prior_bounds["tc"] = (tc_center - 0.025, tc_center + 0.025)
    if "tc" not in initvals:
        initvals["tc"] = tc_center

    engine = GWNUTSSampler(
        time_array=data["time_array"],
        observed_strain=data["scaled_strain"],
        noise_sigma=1.0,
        scale_factor=1.0,
        asd=data["asd_array"] if use_asd else None,
    )

    t0 = time.time()
    idata = engine.build_and_sample_model(
        draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
        prior_bounds=prior_bounds, initvals=initvals,
    )
    elapsed = time.time() - t0

    summary = az.summary(
        idata,
        var_names=["chirp_mass", "luminosity_distance", "tc", "phic"],
        ci_prob=0.90,
    )

    return summary, elapsed, idata


def write_catalog_report(event_name, summary, elapsed, idata,
                          prior_bounds, initvals, robustness_note=""):
    """Write a diagnostic summary to catalog_<event_name>.txt."""
    path = os.path.join(OUTPUT_DIR, f"catalog_{event_name}.txt")
    var_names = ["chirp_mass", "luminosity_distance", "tc", "phic"]

    lines = []
    lines.append(f"Catalog Entry: {event_name}")
    lines.append(f"{'='*60}")
    lines.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"Samples: {N_DRAWS} draws + {N_TUNE} tune x {N_CHAINS} chains")
    lines.append(f"Wall time: {elapsed:.1f}s")
    if robustness_note:
        lines.append(f"Note: {robustness_note}")
    lines.append("")

    lines.append("Prior Bounds:")
    for p in var_names:
        lb, ub = prior_bounds.get(p, ("-", "-"))
        lines.append(f"  {p:20s}: [{lb}, {ub}]")
    lines.append("")

    lines.append("Initvals:")
    for p in var_names:
        iv = initvals.get(p, "-")
        lines.append(f"  {p:20s}: {iv}")
    lines.append("")

    # Detect CI column names (hdi_5%/hdi_95% in older arviz, eti90_lb/eti90_ub in newer)
    ci_low_col = next(c for c in summary.columns if c.endswith("_lb") or c.endswith("_5%"))
    ci_high_col = next(c for c in summary.columns if c.endswith("_ub") or c.endswith("_95%"))

    lines.append("Posterior Summary (90% CI):")
    lines.append(f"{'Parameter':<25s} {'Mean':>10s} {'SD':>10s} {ci_low_col:>10s} "
                 f"{ci_high_col:>10s} {'r_hat':>8s} {'ess_bulk':>10s}")
    lines.append("-" * 85)
    all_ok = True
    for p in var_names:
        if p not in summary.index:
            continue
        row = summary.loc[p]
        mean = float(row["mean"])
        sd = float(row["sd"])
        hdi_low = float(row[ci_low_col])
        hdi_high = float(row[ci_high_col])
        r_hat = float(row["r_hat"])
        ess = int(row["ess_bulk"])
        r_status = "OK" if r_hat < 1.05 else "HIGH"
        if r_hat >= 1.05:
            all_ok = False
        lines.append(
            f"{p:<25s} {mean:>10.3f} {sd:>10.3f} {hdi_low:>10.3f} "
            f"{hdi_high:>10.3f} {r_hat:>8.3f} ({r_status}) {ess:>10d}"
        )

    lines.append("")
    min_ess = int(summary["ess_bulk"].min())
    lines.append(f"Minimum ESS: {min_ess}")
    lines.append(f"All r_hat < 1.05: {'PASS' if all_ok else 'FAIL'}")

    # ESS/s metric
    ess_per_sec = min_ess / elapsed if elapsed > 0 else 0
    lines.append(f"ESS/sec: {ess_per_sec:.2f}")
    lines.append(f"{'='*60}")

    report = "\n".join(lines)
    print(report)

    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\n  Wrote {path}")
    return all_ok


# =====================================================================
# Task 2.1: Mass-Regime Testing (three events)
# =====================================================================

def task_2_1():
    print("\n\n")
    print("#" * 70)
    print("#  Task 2.1: Mass-Regime Testing")
    print("#" * 70)

    events = [
        {
            "name": "GW151226",
            "detector": "H1",
            "duration": 1.0,
            "prior_bounds": {
                "chirp_mass": (8, 30),
                "luminosity_distance": (100, 1200),
            },
            "initvals": {
                "chirp_mass": 18.0,
                "luminosity_distance": 600.0,
            },
            "label": "Low-Mass BBH (GW151226)",
        },
        {
            "name": "GW190412",
            "detector": "H1",
            "duration": 1.0,
            "prior_bounds": {
                "chirp_mass": (25, 80),
                "luminosity_distance": (200, 2500),
            },
            "initvals": {
                "chirp_mass": 50.0,
                "luminosity_distance": 1200.0,
            },
            "label": "Asymmetric-Mass BBH (GW190412)",
        },
        {
            "name": "GW190521",
            "detector": "H1",
            "duration": 1.0,
            "prior_bounds": {
                "chirp_mass": (60, 250),
                "luminosity_distance": (500, 5000),
            },
            "initvals": {
                "chirp_mass": 150.0,
                "luminosity_distance": 2500.0,
            },
            "label": "High-Mass BBH (GW190521)",
        },
    ]

    all_passed = True
    for ev in events:
        try:
            summary, elapsed, idata = run_event(
                ev["name"], ev["detector"], ev["duration"],
                ev["prior_bounds"], ev["initvals"],
                label=ev["label"],
            )
            ok = write_catalog_report(
                ev["name"], summary, elapsed, idata,
                ev["prior_bounds"], ev["initvals"],
            )
            if not ok:
                all_passed = False
        except Exception as e:
            print(f"\n  ERROR on {ev['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\n  Task 2.1 overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# =====================================================================
# Task 2.2: Real Noise & PSD Robustness
# =====================================================================

def task_2_2():
    print("\n\n")
    print("#" * 70)
    print("#  Task 2.2: Real Noise & PSD Robustness")
    print("#" * 70)

    # Compare inference WITH and WITHOUT frequency-domain ASD whitening
    # on the same event to verify that PSD-based whitening does not
    # corrupt the JAX likelihood gradients.
    event_name = "GW151226"
    detector = "H1"
    duration = 1.0
    prior_bounds = {
        "chirp_mass": (8, 30),
        "luminosity_distance": (100, 1200),
    }
    initvals = {
        "chirp_mass": 18.0,
        "luminosity_distance": 600.0,
    }

    # Run WITH ASD (default pipeline)
    print("\n--- With ASD whitening ---")
    summary_on, elapsed_on, idata_on = run_event(
        event_name, detector, duration,
        prior_bounds, initvals,
        use_asd=True,
        label=f"{event_name} [ASD=ON]",
    )
    write_catalog_report(
        f"{event_name}_asd_on", summary_on, elapsed_on, idata_on,
        prior_bounds, initvals,
        robustness_note="ASD-based frequency-domain whitening enabled",
    )

    # Run WITHOUT ASD
    print("\n--- Without ASD whitening ---")
    prior_bounds_noasd = dict(prior_bounds)
    prior_bounds_noasd["tc"] = (duration / 2 - 0.025, duration / 2 + 0.025)
    initvals_noasd = dict(initvals)
    initvals_noasd["tc"] = duration / 2
    summary_off, elapsed_off, idata_off = run_event(
        event_name, detector, duration,
        prior_bounds_noasd, initvals_noasd,
        use_asd=False,
        label=f"{event_name} [ASD=OFF]",
    )
    write_catalog_report(
        f"{event_name}_noasd", summary_off, elapsed_off, idata_off,
        prior_bounds_noasd, initvals_noasd,
        robustness_note="No frequency-domain whitening (ASD disabled)",
    )

    # Compare parameters
    print("\n--- PSD Robustness Comparison ---")
    print(f"{'Parameter':<25s} {'ASD=ON Mean':>12s} {'ASD=OFF Mean':>13s} {'Δ (%)':>10s}")
    print("-" * 62)
    all_ok = True
    for p in ["chirp_mass", "luminosity_distance", "tc", "phic"]:
        m_on = float(summary_on.loc[p, "mean"])
        m_off = float(summary_off.loc[p, "mean"])
        rel_diff = (m_off - m_on) / abs(m_on) * 100 if abs(m_on) > 1e-10 else 0
        if abs(rel_diff) > 10:
            all_ok = False
        print(f"{p:<25s} {m_on:>12.3f} {m_off:>13.3f} {rel_diff:>9.2f}%")
    print(f"\n  Robustness check: {'PASS' if all_ok else 'FAIL'} (threshold: <10% variation)")
    print(f"  ASD=ON  elapsed: {elapsed_on:.1f}s")
    print(f"  ASD=OFF elapsed: {elapsed_off:.1f}s")

    return all_ok


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t_total = time.time()
    passed_21 = task_2_1()
    passed_22 = task_2_2()
    t_total = time.time() - t_total

    print(f"\n{'='*70}")
    print(f"  Phase 2 Summary")
    print(f"{'='*70}")
    print(f"  Task 2.1 (Mass-Regime Testing): {'PASS' if passed_21 else 'FAIL'}")
    print(f"  Task 2.2 (PSD Robustness):      {'PASS' if passed_22 else 'FAIL'}")
    print(f"  Total wall time: {t_total:.1f}s")
    print(f"{'='*70}")
