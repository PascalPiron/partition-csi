#!/usr/bin/env python3
"""
External dataset validation against the partition-csi extractor primitives.

Addresses Self-critique that motivated this generator: "the entire validation
chain is synthetic."

This script validates the extractor primitives (Hampel cleanup,
HSR subcarrier selection, PCA fusion, cardiac-band SNR, FFT-based BPM
extraction) against a public CSI dataset with cardiac ground truth in
the filename.

The dataset used is from the Gi-z/CSI-Data repository, which mirrors
heart-rate-labeled CSI captures collected with the linux-80211n-csitool
on Intel IWL5300 hardware. Files are labeled NNbpm.dat (e.g.,
75bpm.dat) where the label is the ground-truth heart rate in BPM.

Usage:
    pip install --user --break-system-packages csiread
    mkdir -p /tmp/csi_public_dataset
    cd /tmp/csi_public_dataset
    for bpm in 66 71 73 75 76 84 88 90 92; do
        curl -sL -o "${bpm}bpm.dat" \\
            "https://raw.githubusercontent.com/Gi-z/CSI-Data/main/\\
Internal/intel/Heart%20Rate/${bpm}bpm.dat"
    done

    python3 analysis/external_dataset_validate.py \\
        --dataset-dir /tmp/csi_public_dataset \\
        --output reports/EXTERNAL_DATASET_VALIDATION.md

The script:
1. Loads each .dat file via the csiread library.
2. Reduces the (n_packets, n_subcarriers, n_rx, n_tx) tensor to a
   per-packet, per-subcarrier amplitude matrix by averaging over RX/TX
   antennas (a defensible reduction; the extractor is single-antenna).
3. Estimates the sample rate from the packet timestamps.
4. Runs the same cleanup + extraction primitives the extractor uses.
5. Compares the extracted BPM to the ground truth in the filename.

Output: a Markdown table of (filename, ground-truth BPM, extracted BPM,
absolute error, sample rate, packet count, in-band SNR). Also reports
mean absolute error across all files.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

# Reuse extractor primitives so this validation tests the SAME math
# the extractor uses on real captures.
import extractor  # noqa: E402


def estimate_sample_rate_from_timestamps(timestamps_us: np.ndarray) -> float:
    """Estimate sample rate from microsecond timestamps."""
    if len(timestamps_us) < 2:
        return 0.0
    # Handle uint32 wraparound by converting to int64 and unwrapping
    ts = timestamps_us.astype(np.int64)
    dt_us = np.diff(ts)
    # Wraparound at 2^32 microseconds (~71 minutes); detect and correct
    wrap = 2 ** 32
    dt_us = np.where(dt_us < 0, dt_us + wrap, dt_us)
    dt_us = dt_us[(dt_us > 0) & (dt_us < 1_000_000)]  # keep [1us, 1s]
    if len(dt_us) == 0:
        return 0.0
    return float(1_000_000 / np.median(dt_us))


def load_intel_dat(dat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an Intel IWL5300 .dat file. Returns (amp, ts_seconds, rssi_dbm).

    amp shape: (n_subcarriers, n_packets): same convention as
    extractor.load_block_amplitudes.
    """
    import csiread
    data = csiread.Intel(str(dat_path), nrxnum=3, ntxnum=2, pl_size=10)
    data.read()
    n = data.count
    if n < 32:
        return np.zeros((30, 0)), np.array([]), np.array([])
    csi = data.get_scaled_csi()
    # csi shape varies by tool version: try (n, 30, n_rx, n_tx)
    if csi.ndim != 4:
        raise ValueError(f"unexpected CSI shape {csi.shape}")
    n_pkts, n_sc, n_rx, n_tx = csi.shape
    # Reduce RX/TX dimensions: take the first RX-TX pair (most common
    # use; alternative is mean across antennas which is defensible too).
    # Single-antenna config matches the extractor's ESP32 setup.
    amp = np.abs(csi[:, :, 0, 0]).T  # (n_sc, n_pkts)
    fs = estimate_sample_rate_from_timestamps(data.timestamp_low)
    if fs > 0:
        ts = np.arange(n_pkts) / fs
    else:
        ts = np.arange(n_pkts) * 0.033  # fallback 30 Hz
    # RSSI from raw rssi_a (dBm-scale, signed conversion needed)
    rssi_a = np.array(data.rssi_a, dtype=np.float64)
    rssi_dbm = -(rssi_a + 30)  # IWL5300 raw → dBm offset; not exact but indicative
    return amp, ts, rssi_dbm


def analyze_external_capture(amp: np.ndarray, ts: np.ndarray) -> dict:
    """Run extractor primitives on imported (amp, timestamps)."""
    n_sc, n_pkts = amp.shape
    if n_pkts < 32:
        return {"error": f"only {n_pkts} packets, need ≥32"}
    fs = extractor.estimate_sample_rate(ts) if n_pkts >= 2 else 0.0
    if fs <= 0 or n_sc < 8:
        return {"error": f"degenerate: fs={fs}, n_sc={n_sc}"}

    # Apply the same pipeline as analyze_block (without the 5-block
    # structure since the external capture is one continuous segment).
    cleaned = extractor.hampel(amp)
    cleaned = cleaned - cleaned.mean(axis=1, keepdims=True)
    selected = extractor.select_subcarriers(cleaned, fs)
    sub = cleaned[selected, :]
    fused = extractor.pca_fuse(sub)
    snr_db = extractor.cardiac_snr_db(fused, fs)
    bpm, prom, conf = extractor.estimate_bpm(fused, fs)

    return {
        "n_packets": int(n_pkts),
        "n_subcarriers": int(n_sc),
        "sample_rate_hz": round(fs, 2),
        "duration_s": round(float(ts[-1] - ts[0]), 1) if n_pkts >= 2 else 0.0,
        "selected_subcarriers": selected,
        "cardiac_snr_db": (None if not np.isfinite(snr_db)
                           else round(snr_db, 2)),
        "bpm_extracted": (None if bpm is None else round(bpm, 1)),
        "bpm_prominence": round(prom, 2),
        "bpm_confidence": round(conf, 3),
    }


def parse_bpm_from_filename(filename: str) -> int | None:
    """Filename '75bpm.dat' -> 75. None if no match."""
    m = re.match(r"(\d+)\s*bpm", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="Directory of NNbpm.dat files")
    ap.add_argument("--output", type=Path,
                    help="Markdown report output path. If omitted, prints to stdout.")
    args = ap.parse_args()

    files = sorted(args.dataset_dir.glob("*bpm.dat"))
    if not files:
        print(f"no NNbpm.dat files in {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        gt_bpm = parse_bpm_from_filename(f.name)
        if gt_bpm is None:
            continue
        try:
            amp, ts, rssi = load_intel_dat(f)
        except Exception as e:
            rows.append({"file": f.name, "gt_bpm": gt_bpm, "error": str(e)})
            continue
        if amp.size == 0:
            rows.append({"file": f.name, "gt_bpm": gt_bpm,
                         "error": "no packets parsed"})
            continue
        result = analyze_external_capture(amp, ts)
        result["file"] = f.name
        result["gt_bpm"] = gt_bpm
        result["mean_rssi_dbm"] = (round(float(np.mean(rssi)), 1)
                                    if rssi.size else None)
        if "bpm_extracted" in result and result["bpm_extracted"] is not None:
            result["bpm_err"] = round(abs(result["bpm_extracted"] - gt_bpm), 1)
        rows.append(result)

    # Render Markdown
    out = []
    out.append("# External Dataset Validation: Results\n")
    out.append("Validates extractor primitives against the heart-rate-")
    out.append("labeled subset of the [Gi-z/CSI-Data](https://github.com/Gi-z/CSI-Data)")
    out.append("repository (Intel IWL5300 captures via linux-80211n-csitool).")
    out.append("Filename `NNbpm.dat` carries the ground-truth heart rate.\n")
    out.append("| File | GT BPM | Extracted BPM | Err | fs (Hz) | Pkts | Dur (s) | SNR (dB) | conf | RSSI |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    valid_errs = []
    for r in rows:
        if "error" in r:
            out.append(f"| {r['file']} | {r['gt_bpm']} | ERROR | "
                       f": |: |: |: |: |: | {r.get('error', '')} |")
            continue
        bpm_ext = r.get("bpm_extracted")
        err = r.get("bpm_err", ":")
        if isinstance(err, (int, float)):
            valid_errs.append(err)
        out.append(f"| {r['file']} | {r['gt_bpm']} | {bpm_ext} | {err} | "
                   f"{r.get('sample_rate_hz')} | {r.get('n_packets')} | "
                   f"{r.get('duration_s')} | {r.get('cardiac_snr_db')} | "
                   f"{r.get('bpm_confidence')} | {r.get('mean_rssi_dbm')} |")
    out.append("")
    if valid_errs:
        out.append(f"\n**Mean absolute BPM error across {len(valid_errs)} files: "
                   f"{np.mean(valid_errs):.1f} BPM**")
        out.append(f"**Median absolute BPM error: {np.median(valid_errs):.1f} BPM**")
        out.append(f"**Max absolute BPM error: {np.max(valid_errs):.1f} BPM**")
        in_band = sum(1 for e in valid_errs if e <= 5)
        out.append(f"\n{in_band}/{len(valid_errs)} files within 5 BPM of "
                   f"ground truth ({100*in_band/len(valid_errs):.0f}%).")

    out.append("\n## Methodology notes\n")
    out.append("- The dataset .dat files are short (under 5 seconds in some cases)")
    out.append("  and were captured at varying sample rates. Short captures and low")
    out.append("  rates fundamentally limit BPM extraction precision.")
    out.append("- Reduction from (subcarriers × RX × TX) to (subcarriers,) is via")
    out.append("  selecting the first RX-TX pair, matching the extractor's single-antenna")
    out.append("  ESP32 configuration. Alternative reductions (mean across antennas,")
    out.append("  best-SNR antenna) would give different numbers but are not closer")
    out.append("  to the extractor's actual capture path.")
    out.append("- This validation tests the extractor primitives (Hampel, HSR,")
    out.append("  PCA fusion, FFT band power, BPM peak) against real CSI captured by")
    out.append("  different hardware (Intel IWL5300, not ESP32) in a different")
    out.append("  environment. Convergence on ground-truth BPM is evidence that the")
    out.append("  primitives are not over-fitted to synthetic CSI.")
    out.append("- This is NOT a test of the extractor's full session protocol (5-block")
    out.append("  structure, NCs, cross-room, cross-victim). Those need real bench")
    out.append("  sessions.\n")

    text = "\n".join(out)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
