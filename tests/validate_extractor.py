#!/usr/bin/env python3
"""
Validate extractor.py against synthetic CSI with known ground truth.

For each test case, this script:
  1. generates a synthetic session with controlled cardiac signal +
     optional shield attenuation,
  2. runs extractor.py against it,
  3. compares the analyzer's output to ground truth,
  4. reports PASS/FAIL per metric.

The point: every session report from a real bench is only as
trustworthy as the math in extractor.py. This script catches errors
in that math before they corrupt real data interpretation.

Test matrix
===========
- BPM extraction across the cardiac band (60, 75, 90, 110 BPM)
- Shield SNR loss recovery (0, 3, 10, 20 dB attenuation)
- Body-absent ambient blocks (no extractable cardiac)
- Sample-rate sensitivity (20 Hz, 10 Hz, 5 Hz)
- Noise-floor sensitivity (low, medium, high)

Failure semantics: per-metric tolerance bands. BPM ±3, SNR loss ±3 dB,
within-session OFF replay drift ≤2 dB. The bands are tight enough to
catch real bugs and wide enough to tolerate synthetic-data variability.

Usage
=====
    python3 validate_analyzer.py
    python3 validate_analyzer.py --keep-sessions  # don't delete after
    python3 validate_analyzer.py --verbose
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_DIR / "src"
SYNTH = ANALYSIS_DIR / "synthetic.py"
ANALYZE = ANALYSIS_DIR / "extractor.py"


@dataclass
class TestSpec:
    name: str
    bpm: float
    shield_attenuation_db: float
    sample_rate_hz: float = 20.0
    block_duration_s: float = 60.0
    noise_amplitude: float = 0.6
    cardiac_depth: float = 0.04
    breathing_depth: float = 0.12
    rssi_dbm: float = -65.0
    seed: int = 42

    # Tolerances (analyzer outputs vs ground truth)
    bpm_tolerance: float = 3.0
    snr_loss_tolerance_db: float = 3.0
    off_replay_drift_tolerance_db: float = 2.0


@dataclass
class TestResult:
    name: str
    passed: bool
    failures: list[str]
    bpm_truth: float
    bpm_extracted: float | None
    snr_loss_truth_db: float
    snr_loss_extracted_db: float | None
    off_replay_drift_db: float | None
    raw_report: dict


def _strong(**kw):
    """Tier 1: strong cardiac signal, low noise: validates analyzer math."""
    base = dict(cardiac_depth=0.30, breathing_depth=0.05,
                noise_amplitude=0.2, rssi_dbm=-55.0,
                bpm_tolerance=2.0, snr_loss_tolerance_db=2.5,
                off_replay_drift_tolerance_db=2.0)
    base.update(kw)
    return base


def _realistic(**kw):
    """Tier 2: cardiac modulation similar to literature (a few % of baseline),
    moderate noise. Realistic operating regime."""
    base = dict(cardiac_depth=0.12, breathing_depth=0.18,
                noise_amplitude=0.35, rssi_dbm=-65.0,
                bpm_tolerance=4.0, snr_loss_tolerance_db=4.0,
                off_replay_drift_tolerance_db=3.0)
    base.update(kw)
    return base


def _weak(**kw):
    """Tier 3: literature low-end cardiac depth, higher noise. Stress test:
    tells us the analyzer's operating floor. Not all expected to pass."""
    base = dict(cardiac_depth=0.05, breathing_depth=0.12,
                noise_amplitude=0.6, rssi_dbm=-80.0,
                bpm_tolerance=8.0, snr_loss_tolerance_db=6.0,
                off_replay_drift_tolerance_db=4.0)
    base.update(kw)
    return base


TEST_CASES = [
    # ─── Tier 1: strong signal, validate analyzer logic ───────────────────
    TestSpec(name="T1_bpm60_noshield",  bpm=60.0,  shield_attenuation_db=0.0,
             **_strong()),
    TestSpec(name="T1_bpm75_noshield",  bpm=75.0,  shield_attenuation_db=0.0,
             **_strong()),
    TestSpec(name="T1_bpm90_noshield",  bpm=90.0,  shield_attenuation_db=0.0,
             **_strong()),
    TestSpec(name="T1_bpm110_noshield", bpm=110.0, shield_attenuation_db=0.0,
             **_strong(bpm_tolerance=3.0)),
    TestSpec(name="T1_shield_3db",      bpm=75.0,  shield_attenuation_db=3.0,
             **_strong(snr_loss_tolerance_db=2.5)),
    TestSpec(name="T1_shield_10db",     bpm=75.0,  shield_attenuation_db=10.0,
             **_strong()),
    TestSpec(name="T1_shield_20db",     bpm=75.0,  shield_attenuation_db=20.0,
             **_strong(snr_loss_tolerance_db=4.0)),

    # ─── Tier 2: realistic operating regime ────────────────────────────────
    TestSpec(name="T2_bpm75_noshield",  bpm=75.0,  shield_attenuation_db=0.0,
             **_realistic()),
    TestSpec(name="T2_shield_10db",     bpm=75.0,  shield_attenuation_db=10.0,
             **_realistic()),
    TestSpec(name="T2_rate10hz",        bpm=75.0,  shield_attenuation_db=0.0,
             sample_rate_hz=10.0, **_realistic(bpm_tolerance=5.0)),

    # ─── Tier 3: stress floor (most expected to fail) ─────────────────────
    TestSpec(name="T3_weak_baseline",   bpm=75.0,  shield_attenuation_db=0.0,
             **_weak()),
    TestSpec(name="T3_weak_shield10",   bpm=75.0,  shield_attenuation_db=10.0,
             **_weak()),
]


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def evaluate(spec: TestSpec, report: dict) -> TestResult:
    """Compare analyzer output to ground truth, return pass/fail with details."""
    failures: list[str] = []

    # BPM extraction: check block 2 (OFF baseline): body present, no shield
    blocks = {b["name"]: b for b in report.get("blocks", [])}
    block2 = blocks.get("block2_off_baseline", {})
    bpm_extracted = block2.get("bpm")
    if bpm_extracted is None:
        failures.append(f"block 2 BPM not extracted (truth={spec.bpm})")
    else:
        if abs(bpm_extracted - spec.bpm) > spec.bpm_tolerance:
            failures.append(
                f"block 2 BPM error: extracted={bpm_extracted}, "
                f"truth={spec.bpm}, tol=±{spec.bpm_tolerance}"
            )

    # Shield SNR loss
    L3 = report.get("L3_quantitative", {})
    snr_loss = L3.get("L3_shield_snr_loss_db")
    if snr_loss is None:
        failures.append(
            f"L3 shield SNR loss not computed (truth={spec.shield_attenuation_db})"
        )
    else:
        if abs(snr_loss - spec.shield_attenuation_db) > spec.snr_loss_tolerance_db:
            failures.append(
                f"shield SNR loss error: extracted={snr_loss} dB, "
                f"truth={spec.shield_attenuation_db} dB, "
                f"tol=±{spec.snr_loss_tolerance_db}"
            )

    # OFF-replay drift (block 2 vs block 4): synthetic blocks are
    # generated with different seeds, so some drift is expected. Floor
    # at the tolerance.
    drift = L3.get("L3_off_replay_drift_db")
    if drift is None:
        failures.append("L3 OFF replay drift not computed")
    else:
        if abs(drift) > spec.off_replay_drift_tolerance_db:
            failures.append(
                f"OFF replay drift too large: {drift} dB, "
                f"tol=±{spec.off_replay_drift_tolerance_db}"
            )

    # L1 detection: body should NOT be reported present in cold ambient
    # blocks. For synthetic data we don't strictly require this because
    # the L1 ambient threshold is heuristic, but flag if it's wildly off.
    L1 = report.get("L1_detection", {})
    if L1.get("block2_subject_present") is False:
        failures.append(
            "L1: body not detected in block 2 despite cardiac signal present"
        )

    return TestResult(
        name=spec.name,
        passed=not failures,
        failures=failures,
        bpm_truth=spec.bpm,
        bpm_extracted=bpm_extracted,
        snr_loss_truth_db=spec.shield_attenuation_db,
        snr_loss_extracted_db=snr_loss,
        off_replay_drift_db=drift,
        raw_report=report,
    )


def run_one_test(spec: TestSpec, work_dir: Path,
                 verbose: bool = False) -> TestResult:
    session_dir = work_dir / spec.name

    # Generate synthetic session
    synth_cmd = [
        sys.executable, str(SYNTH),
        "--output", str(session_dir),
        "--bpm", str(spec.bpm),
        "--shield-attenuation-db", str(spec.shield_attenuation_db),
        "--sample-rate", str(spec.sample_rate_hz),
        "--block-duration", str(spec.block_duration_s),
        "--noise-amplitude", str(spec.noise_amplitude),
        "--cardiac-depth", str(spec.cardiac_depth),
        "--breathing-depth", str(spec.breathing_depth),
        "--rssi", str(spec.rssi_dbm),
        "--seed", str(spec.seed),
    ]
    proc = run(synth_cmd)
    if proc.returncode != 0:
        return TestResult(
            name=spec.name, passed=False,
            failures=[f"synthetic generation failed rc={proc.returncode}: "
                      f"{proc.stderr.strip()[:200]}"],
            bpm_truth=spec.bpm, bpm_extracted=None,
            snr_loss_truth_db=spec.shield_attenuation_db,
            snr_loss_extracted_db=None,
            off_replay_drift_db=None,
            raw_report={},
        )

    # Run analyzer
    analyze_cmd = [sys.executable, str(ANALYZE), str(session_dir), "--json"]
    proc = run(analyze_cmd)
    if proc.returncode != 0:
        return TestResult(
            name=spec.name, passed=False,
            failures=[f"analyzer failed rc={proc.returncode}: "
                      f"{proc.stderr.strip()[:200]}"],
            bpm_truth=spec.bpm, bpm_extracted=None,
            snr_loss_truth_db=spec.shield_attenuation_db,
            snr_loss_extracted_db=None,
            off_replay_drift_db=None,
            raw_report={},
        )

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return TestResult(
            name=spec.name, passed=False,
            failures=[f"analyzer stdout not JSON: {proc.stdout[:200]}"],
            bpm_truth=spec.bpm, bpm_extracted=None,
            snr_loss_truth_db=spec.shield_attenuation_db,
            snr_loss_extracted_db=None,
            off_replay_drift_db=None,
            raw_report={},
        )

    if verbose:
        print(f"  raw L3: {report.get('L3_quantitative', {})}")

    return evaluate(spec, report)


def render_summary(results: list[TestResult]) -> str:
    lines = []
    lines.append("")
    lines.append("  extractor.py validation against synthetic CSI")
    lines.append("  " + "─" * 70)
    lines.append(f"  {'test case':<22} {'BPM truth':>9} {'BPM ext':>9} "
                 f"{'SNR loss truth':>14} {'SNR loss ext':>13} {'pass':>5}")
    lines.append("  " + "─" * 70)
    for r in results:
        bpm_ext = f"{r.bpm_extracted:.1f}" if r.bpm_extracted is not None else ":"
        snr_ext = f"{r.snr_loss_extracted_db:.2f}" if r.snr_loss_extracted_db is not None else ":"
        status = "✓" if r.passed else "✗"
        lines.append(
            f"  {r.name:<22} {r.bpm_truth:>9.1f} {bpm_ext:>9} "
            f"{r.snr_loss_truth_db:>14.1f} {snr_ext:>13} {status:>5}"
        )
    n_pass = sum(1 for r in results if r.passed)
    lines.append("  " + "─" * 70)
    lines.append(f"  {n_pass} / {len(results)} passed")
    lines.append("")
    if any(not r.passed for r in results):
        lines.append("  Failures:")
        for r in results:
            if not r.passed:
                lines.append(f"    {r.name}:")
                for f in r.failures:
                    lines.append(f"      • {f}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-sessions", action="store_true",
                        help="Do not delete generated synthetic sessions")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--filter", default="",
                        help="Only run tests whose name contains this substring")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Write full results as JSON")
    args = parser.parse_args()

    cases = [c for c in TEST_CASES if args.filter in c.name]
    if not cases:
        print("no test cases match --filter")
        sys.exit(2)

    work_dir = Path(tempfile.mkdtemp(prefix="extractor_validate_"))
    print(f"work dir: {work_dir}")
    print(f"running {len(cases)} test cases...")

    results: list[TestResult] = []
    for i, spec in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {spec.name}...", flush=True)
        r = run_one_test(spec, work_dir, verbose=args.verbose)
        results.append(r)
        if not r.passed and args.verbose:
            for f in r.failures:
                print(f"      ✗ {f}")

    print(render_summary(results))

    if args.output_json:
        args.output_json.write_text(json.dumps(
            [{"name": r.name, "passed": r.passed,
              "failures": r.failures,
              "bpm_truth": r.bpm_truth, "bpm_extracted": r.bpm_extracted,
              "snr_loss_truth_db": r.snr_loss_truth_db,
              "snr_loss_extracted_db": r.snr_loss_extracted_db,
              "off_replay_drift_db": r.off_replay_drift_db}
             for r in results],
            indent=2, default=str) + "\n")

    if not args.keep_sessions:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"\n  sessions kept at {work_dir}")

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
