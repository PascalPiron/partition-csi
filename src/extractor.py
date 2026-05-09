#!/usr/bin/env python3
"""
partition-csi: offline cardiac signal extractor from WiFi Channel State
Information.

Reads a session directory (.jsonl per measurement block + session.json
metadata) and computes three metric layers:

  L1: Detection (binary). Cold-ambient blocks vs subject-present blocks.
  L2: Extraction (BPM + prominence + confidence in [0,1]).
  L3: Quantitative SNR in the 0.8-2.0 Hz cardiac band, dB.

The analyzer pipeline:
  1. Hampel outlier rejection per subcarrier.
  2. Mean removal.
  3. HSR (Heartbeat-to-Subcomponent Ratio) subcarrier selection,
     locked to the OFF baseline block to prevent shield-effect masking.
  4. PCA fusion across selected subcarriers (first principal component).
  5. FFT band-power for L3 SNR; FFT peak (parabolic-equivalent
     resolution via 4x zero-padding) for L2 BPM.

Thresholds are literature-anchored (see docs/THRESHOLDS.md):
  cardiac band: 0.8-2.0 Hz (Pulse-Fi 2024-25 floor; conservative on
  upper edge to avoid second-harmonic contamination of the noise band).
  noise band: 2.5-4.0 Hz (above strongest second-harmonic content of
  60-75 BPM cardiac).
  L1 threshold: 1.0 dB above ambient (~3 sigma under realistic conditions).
  L2 confidence: prominence > 3.8 (~5.8 dB above flat-spectrum baseline,
  consistent with WiCG 2024 4-5 dB empirical detection band).

Usage:
    python3 src/extractor.py sessions/<session_id>/
    python3 src/extractor.py sessions/<session_id>/ --json

License: AGPL-3.0-or-later.
"""
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ── Constants + literature anchoring ──
# See docs/THRESHOLDS.md for the citation chain.
# Cardiac band: 0.8 to 2.0 Hz (48 to 120 BPM). Conservative vs. the
# literature consensus of 0.8-2.5 Hz; the narrower band protects the
# SNR estimate from second-harmonic contamination of 60-75 BPM cardiac
# (which falls at 2.0-2.5 Hz). Subjects at rest do not exceed 120 BPM.
# Noise reference: single wide band above cardiac, 2.5 to 4.0 Hz.
# - Below cardiac: respiratory dominates 0.1-0.5 Hz (Liu 2020,
#   Wavelet-decoupling 2024), slow drift dominates <0.1 Hz.
# - Above cardiac: 2.0-2.5 Hz contains second harmonics of 60-75 BPM
#   cardiac. Skipping it preserves the SNR estimate.
CARDIAC_LOW_HZ = 0.8
CARDIAC_HIGH_HZ = 2.0
NOISE_LOW_BAND = None  # disabled
NOISE_HIGH_BAND = (2.5, 4.0)

# Hampel filter (cardiac_extractor.py defaults)
HAMPEL_WINDOW = 5
HAMPEL_THRESHOLD = 3.0

# HSR
HSR_TOP_K = 8
HSR_MIN_RATIO = 0.05


def load_block_amplitudes(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a block's frames and return (timestamps, amplitudes_matrix).

    amplitudes_matrix has shape (n_subcarriers, n_frames).
    Frames whose payload does not parse or does not contain 'amplitudes'
    are dropped silently (counted in the returned dict's drop count, but
    here we just drop and return what we have)."""
    timestamps = []
    amp_rows = []
    n_sc = None

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("_meta"):
                continue
            raw = obj.get("raw")
            if not raw:
                continue
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                continue
            amps = frame.get("amplitudes") or frame.get("amplitude")
            if not amps or not isinstance(amps, list):
                continue
            if n_sc is None:
                n_sc = len(amps)
            if len(amps) != n_sc:
                # Subcarrier count change mid-block: skip frame.
                continue
            timestamps.append(obj.get("t", 0.0))
            amp_rows.append(amps)

    if not amp_rows:
        return np.array([]), np.zeros((0, 0))

    ts = np.array(timestamps, dtype=np.float64)
    amp = np.array(amp_rows, dtype=np.float64).T  # (n_sc, n_frames)
    return ts, amp


def estimate_sample_rate(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    dt = np.diff(timestamps)
    dt = dt[(dt > 0) & (dt < 1.0)]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def hampel(x: np.ndarray, window: int = HAMPEL_WINDOW,
           threshold: float = HAMPEL_THRESHOLD) -> np.ndarray:
    """Vectorized-ish Hampel filter over each subcarrier row independently."""
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    out = x.copy()
    n = x.shape[1]
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        local = x[:, lo:hi]
        med = np.median(local, axis=1)
        mad = 1.4826 * np.median(np.abs(local - med[:, None]), axis=1)
        mad = np.where(mad < 1e-12, 1e-12, mad)
        outlier = np.abs(x[:, i] - med) > threshold * mad
        out[outlier, i] = med[outlier]
    return out[0] if squeeze else out


def select_subcarriers(amp: np.ndarray, fs: float,
                       top_k: int = HSR_TOP_K,
                       min_ratio: float = HSR_MIN_RATIO) -> list:
    """HSR-based subcarrier selection. Returns list of indices."""
    n_sc, n = amp.shape
    if n < 16 or fs <= 0:
        return list(range(min(n_sc, top_k)))

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    cardiac_mask = (freqs >= CARDIAC_LOW_HZ) & (freqs <= CARDIAC_HIGH_HZ)

    hsr = np.zeros(n_sc)
    for k in range(n_sc):
        x = amp[k] - np.mean(amp[k])
        spec = np.abs(np.fft.rfft(x)) ** 2
        total = np.sum(spec[1:])
        if total < 1e-15:
            continue
        hsr[k] = np.sum(spec[cardiac_mask]) / total

    candidates = [(i, h) for i, h in enumerate(hsr) if h >= min_ratio]
    if not candidates:
        candidates = list(enumerate(hsr))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in candidates[:top_k]]


def pca_fuse(amp: np.ndarray) -> np.ndarray:
    """First principal component across rows (subcarriers)."""
    if amp.shape[0] == 1:
        return amp[0] - np.mean(amp[0])
    centered = amp - amp.mean(axis=1, keepdims=True)
    cov = np.cov(centered)
    _, vecs = np.linalg.eigh(cov)
    pc1 = vecs[:, -1]
    return pc1 @ centered


def band_power(signal: np.ndarray, fs: float,
               low: float, high: float) -> float:
    """Power in a frequency band via Welch-ish single-FFT estimate."""
    n = len(signal)
    if n < 16 or fs <= 0:
        return 0.0
    x = signal - np.mean(signal)
    spec = np.abs(np.fft.rfft(x)) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(spec[mask]))


def cardiac_snr_db(signal: np.ndarray, fs: float) -> float:
    """In-band cardiac PSD vs adjacent-band PSD, in dB."""
    if len(signal) < 16 or fs <= 0:
        return float("nan")
    in_band = band_power(signal, fs, CARDIAC_LOW_HZ, CARDIAC_HIGH_HZ)
    noises = []
    if NOISE_LOW_BAND is not None:
        noises.append(band_power(signal, fs, *NOISE_LOW_BAND))
    if NOISE_HIGH_BAND is not None:
        noises.append(band_power(signal, fs, *NOISE_HIGH_BAND))
    if not noises:
        return float("nan")
    noise = sum(noises) / len(noises)
    if in_band <= 0 or noise <= 0:
        return float("-inf")
    return 10.0 * np.log10(in_band / noise)


def estimate_bpm(signal: np.ndarray, fs: float) -> tuple:
    """FFT peak in cardiac band. Returns (bpm, prominence, confidence)."""
    if len(signal) < 32 or fs <= 0:
        return (None, 0.0, 0.0)
    n = len(signal)
    pad = n * 4
    x = signal - np.mean(signal)
    if np.max(np.abs(x)) < 1e-15:
        return (None, 0.0, 0.0)
    spec = np.abs(np.fft.rfft(x, n=pad))
    freqs = np.fft.rfftfreq(pad, d=1.0 / fs)
    mask = (freqs >= CARDIAC_LOW_HZ) & (freqs <= CARDIAC_HIGH_HZ)
    if not np.any(mask):
        return (None, 0.0, 0.0)
    band_spec = spec[mask]
    band_freqs = freqs[mask]
    peak_idx = int(np.argmax(band_spec))
    peak_val = float(band_spec[peak_idx])
    band_mean = float(np.mean(band_spec)) + 1e-15
    prominence = peak_val / band_mean
    bpm = float(band_freqs[peak_idx]) * 60.0
    # Confidence: prominence mapped via cardiac_extractor.py defaults.
    # SNR_floor=2.0 -> 0.0, SNR_ceil=8.0 -> 1.0
    confidence = max(0.0, min(1.0, (prominence - 2.0) / (8.0 - 2.0)))
    return (bpm, prominence, confidence)


@dataclass
class BlockResult:
    name: str
    label: str
    n_frames: int
    duration_s: float
    sample_rate_hz: float
    n_subcarriers: int
    selected_subcarriers: list
    cardiac_snr_db: float
    bpm: float | None
    bpm_prominence: float
    bpm_confidence: float


def analyze_block(jsonl_path: Path, block_label: str,
                  locked_subcarriers: list | None = None) -> BlockResult:
    """Analyze one block.

    If locked_subcarriers is provided, use those subcarriers instead of
    running HSR selection on this block. This is the correct way to
    measure shield effect: select the cardiac-rich subcarriers from the
    OFF baseline once, then measure SNR on the same subcarriers in the
    ON block. Otherwise HSR will pick noise-dominated subcarriers in the
    suppressed block, masking the shield's effect.
    """
    ts, amp = load_block_amplitudes(jsonl_path)
    n_sc, n = amp.shape if amp.ndim == 2 else (0, 0)

    if n < 32:
        return BlockResult(
            name=jsonl_path.stem, label=block_label,
            n_frames=int(n), duration_s=0.0, sample_rate_hz=0.0,
            n_subcarriers=int(n_sc), selected_subcarriers=[],
            cardiac_snr_db=float("nan"), bpm=None,
            bpm_prominence=0.0, bpm_confidence=0.0,
        )

    fs = estimate_sample_rate(ts)
    duration = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0

    # Hampel cleanup per subcarrier
    cleaned = hampel(amp)

    # Detrend
    cleaned = cleaned - cleaned.mean(axis=1, keepdims=True)

    # Subcarrier selection: locked if provided, else HSR on this block.
    if locked_subcarriers is not None:
        # Filter out any locked indices that don't exist for this block
        # (in practice n_sc is constant across a session but be defensive).
        selected = [i for i in locked_subcarriers if i < n_sc]
        if not selected:
            selected = list(range(min(n_sc, 8)))
    else:
        selected = select_subcarriers(cleaned, fs)
    sub = cleaned[selected, :]

    # PCA fusion
    fused = pca_fuse(sub)

    snr_db = cardiac_snr_db(fused, fs)
    bpm, prom, conf = estimate_bpm(fused, fs)

    return BlockResult(
        name=jsonl_path.stem, label=block_label,
        n_frames=int(n), duration_s=round(duration, 2),
        sample_rate_hz=round(fs, 2), n_subcarriers=int(n_sc),
        selected_subcarriers=selected,
        cardiac_snr_db=round(snr_db, 2) if np.isfinite(snr_db) else snr_db,
        bpm=round(bpm, 1) if bpm is not None else None,
        bpm_prominence=round(prom, 2),
        bpm_confidence=round(conf, 3),
    )


def analyze_session(session_dir: Path) -> dict:
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"session.json not found in {session_dir}")
    meta = json.loads(meta_path.read_text())

    block_files = [
        ("block1_cold_ambient",   "Cold ambient PRE"),
        ("block2_off_baseline",   "Shield OFF baseline"),
        ("block3_on_measurement", "Shield ON measurement"),
        ("block4_off_replay",     "Shield OFF replay"),
        ("block5_cold_ambient",   "Cold ambient POST"),
    ]

    # Pass 1: analyze block 2 (OFF baseline) FIRST to lock the subcarrier
    # set. The cardiac-rich subcarriers selected here will be reused for
    # blocks 3, 4, 5, 1. Without this lock, HSR runs independently per
    # block and picks different subcarriers when cardiac is suppressed,
    # masking shield effect.
    locked_subcarriers: list | None = None
    block2_path = session_dir / "block2_off_baseline.jsonl"
    if block2_path.exists():
        block2_result = analyze_block(block2_path, "Shield OFF baseline")
        locked_subcarriers = block2_result.selected_subcarriers

    # Pass 2: analyze every block with locked subcarriers (reusing block 2
    # result rather than recomputing).
    results = []
    for fname, label in block_files:
        path = session_dir / f"{fname}.jsonl"
        if not path.exists():
            continue
        if fname == "block2_off_baseline" and locked_subcarriers is not None:
            results.append(block2_result)
        else:
            results.append(analyze_block(path, label,
                                         locked_subcarriers=locked_subcarriers))

    # Derived metrics
    by_name = {r.name: r for r in results}
    snr_off2 = by_name.get("block2_off_baseline").cardiac_snr_db if "block2_off_baseline" in by_name else None
    snr_on  = by_name.get("block3_on_measurement").cardiac_snr_db if "block3_on_measurement" in by_name else None
    snr_off4 = by_name.get("block4_off_replay").cardiac_snr_db if "block4_off_replay" in by_name else None
    snr_amb1 = by_name.get("block1_cold_ambient").cardiac_snr_db if "block1_cold_ambient" in by_name else None
    snr_amb5 = by_name.get("block5_cold_ambient").cardiac_snr_db if "block5_cold_ambient" in by_name else None

    def safe_sub(a, b):
        if a is None or b is None:
            return None
        if not (np.isfinite(a) and np.isfinite(b)):
            return None
        return round(a - b, 2)

    derived = {
        "L3_shield_snr_loss_db": safe_sub(snr_off2, snr_on),
        "L3_off_replay_drift_db": safe_sub(snr_off2, snr_off4),
        "L3_ambient_drift_db": safe_sub(snr_amb1, snr_amb5),
        "snr_block_summary": {
            "block1_cold_ambient_db":   snr_amb1,
            "block2_off_baseline_db":   snr_off2,
            "block3_on_measurement_db": snr_on,
            "block4_off_replay_db":     snr_off4,
            "block5_cold_ambient_db":   snr_amb5,
        },
    }

    # L1: presence flag: body present iff cardiac SNR exceeds ambient by
    # at least L1_PRESENCE_THRESHOLD_DB. Threshold = approximately 3σ
    # above ambient block-to-block SNR standard deviation under
    # synthetic-realistic conditions (σ_ambient ≈ 0.3 dB; 3σ ≈ 0.9 dB).
    # Re-derived from real-bench data after Phase 0a per pre-reg §A2.
    # See LITERATURE_THRESHOLDS.md §4.
    L1_PRESENCE_THRESHOLD_DB = 1.0

    def present(snr_block, snr_amb):
        if snr_block is None or snr_amb is None:
            return None
        if not (np.isfinite(snr_block) and np.isfinite(snr_amb)):
            return None
        return bool((snr_block - snr_amb) > L1_PRESENCE_THRESHOLD_DB)

    snr_amb_mean = None
    amb_vals = [v for v in (snr_amb1, snr_amb5) if v is not None and np.isfinite(v)]
    if amb_vals:
        snr_amb_mean = float(np.mean(amb_vals))
    l1 = {
        "ambient_snr_db": round(snr_amb_mean, 2) if snr_amb_mean is not None else None,
        "block2_subject_present": present(snr_off2, snr_amb_mean),
        "block3_subject_present": present(snr_on, snr_amb_mean),
        "block4_subject_present": present(snr_off4, snr_amb_mean),
        "shield_invisibility": (present(snr_off2, snr_amb_mean) is True
                                and present(snr_on, snr_amb_mean) is False),
    }

    # L2: BPM extractable per block (confidence > 0.3 = "extractable")
    def extractable(r):
        if r is None:
            return None
        return bool(r.bpm is not None and r.bpm_confidence > 0.3)
    l2 = {
        "block2_off_extractable": extractable(by_name.get("block2_off_baseline")),
        "block3_on_extractable":  extractable(by_name.get("block3_on_measurement")),
        "block4_off_extractable": extractable(by_name.get("block4_off_replay")),
    }

    return {
        "session_id": meta.get("session_id"),
        "spec_version": meta.get("spec_version"),
        "subject": meta.get("subject"),
        "room": meta.get("room"),
        "victim_config": meta.get("victim_config"),
        "shield_config": meta.get("shield_config"),
        "countermeasure_id": meta.get("countermeasure_id"),
        "traffic_pattern": meta.get("traffic_pattern"),
        "blocks": [asdict(r) for r in results],
        "L1_detection": l1,
        "L2_extraction": l2,
        "L3_quantitative": derived,
    }


def render_report(report: dict) -> str:
    lines = []
    lines.append("")
    lines.append(f"  partition-csi: Session Report")
    lines.append(f"  Session: {report['session_id']}  spec v{report['spec_version']}")
    lines.append(f"  Subject={report['subject']}  Room={report['room']}  "
                 f"Shield={report['shield_config']}  Countermeasure={report['countermeasure_id']}")
    lines.append("  " + "─" * 70)
    lines.append("")
    lines.append("  Per-block:")
    lines.append("  " + "─" * 70)
    lines.append(f"    {'block':<26} {'frames':>6} {'fs(Hz)':>7} "
                 f"{'SNR(dB)':>8} {'BPM':>6} {'conf':>6}")
    for b in report["blocks"]:
        bpm = b["bpm"] if b["bpm"] is not None else ":"
        snr = b["cardiac_snr_db"]
        snr_s = f"{snr:.2f}" if isinstance(snr, (int, float)) and np.isfinite(snr) else ":"
        lines.append(f"    {b['label']:<26} {b['n_frames']:>6} {b['sample_rate_hz']:>7.2f} "
                     f"{snr_s:>8} {bpm!s:>6} {b['bpm_confidence']:>6.3f}")
    lines.append("")
    L1 = report["L1_detection"]
    L2 = report["L2_extraction"]
    L3 = report["L3_quantitative"]
    lines.append("  L1: Detection:")
    lines.append(f"    ambient SNR floor:        {L1['ambient_snr_db']} dB")
    lines.append(f"    body present, block 2:    {L1['block2_subject_present']}")
    lines.append(f"    body present, block 3:    {L1['block3_subject_present']}   <- shield ON")
    lines.append(f"    body present, block 4:    {L1['block4_subject_present']}")
    lines.append(f"    SHIELD INVISIBILITY:      {L1['shield_invisibility']}")
    lines.append("")
    lines.append("  L2: Extraction (BPM extractable, conf > 0.3):")
    lines.append(f"    block 2 (OFF):            {L2['block2_off_extractable']}")
    lines.append(f"    block 3 (ON):             {L2['block3_on_extractable']}   <- shield ON")
    lines.append(f"    block 4 (OFF replay):     {L2['block4_off_extractable']}")
    lines.append("")
    lines.append("  L3: Quantitative degradation:")
    lines.append(f"    SHIELD SNR LOSS:          {L3['L3_shield_snr_loss_db']} dB   "
                 f"(higher = better; spec acceptance ≥10 dB)")
    lines.append(f"    OFF replay drift:         {L3['L3_off_replay_drift_db']} dB   "
                 f"(should be near 0; flags within-session drift)")
    lines.append(f"    Ambient drift:            {L3['L3_ambient_drift_db']} dB   "
                 f"(should be near 0; flags pre/post environmental change)")
    lines.append("")
    lines.append("  " + "─" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path,
                        help="Path to a session directory (contains session.json)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON report instead of human-readable text")
    args = parser.parse_args()

    if not args.session_dir.is_dir():
        print(f"error: not a directory: {args.session_dir}", file=sys.stderr)
        sys.exit(1)

    report = analyze_session(args.session_dir)

    # Save the report alongside the session. If the session lives under
    # a parent's `sessions/` directory (canonical session layout), use a
    # peer `reports/` directory. Otherwise drop the report inside the
    # session directory: useful for ad-hoc /tmp/ sessions in tests.
    if args.session_dir.parent.name == "sessions":
        reports_dir = args.session_dir.parent.parent / "reports"
    else:
        reports_dir = args.session_dir
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        reports_dir = args.session_dir
    report_path = reports_dir / f"{args.session_dir.name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_report(report))
        print(f"  full report saved to: {report_path}")


if __name__ == "__main__":
    main()
