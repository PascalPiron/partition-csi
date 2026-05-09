#!/usr/bin/env python3
"""
Unit tests for the partition-csi extractor's mathematical primitives.

Function-level regression tests. Catches subtle bugs in Hampel, HSR,
PCA, SNR, BPM extraction that the synthetic-end-to-end suite might
miss. Run after any change to src/extractor.py.

Usage
=====
    python3 tests/test_extractor_units.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from extractor import (  # noqa: E402
    hampel, pca_fuse, band_power, cardiac_snr_db, estimate_bpm,
    select_subcarriers, estimate_sample_rate,
    CARDIAC_LOW_HZ, CARDIAC_HIGH_HZ, NOISE_HIGH_BAND,
)


PASSED = 0
FAILED = 0


def expect(cond: bool, label: str, ctx: str = ""):
    global PASSED, FAILED
    if cond:
        print(f"  ✓ {label}")
        PASSED += 1
    else:
        print(f"  ✗ {label}    {ctx}")
        FAILED += 1


def section(title: str):
    print(f"\n[{title}]")


# ─── hampel ─────────────────────────────────────────────────────────

def test_hampel():
    section("hampel filter")

    # 1D: clean signal: Hampel should not change it
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.normal(0, 0.05, 200)
    out = hampel(clean.copy())
    expect(np.allclose(out, clean, atol=0.1),
           "clean 1D signal preserved within 0.1")

    # 1D: spike injection: should be removed
    spiked = clean.copy()
    spiked[100] = 100  # massive spike
    out = hampel(spiked)
    expect(abs(out[100] - clean[100]) < 0.5,
           f"single spike removed (was 100, became {out[100]:.2f})")

    # 2D: row independence
    matrix = np.zeros((4, 100))
    for k in range(4):
        matrix[k] = np.sin(np.linspace(0, 2 * np.pi * (k + 1), 100))
    matrix[1, 50] = 100  # spike on row 1 only
    out = hampel(matrix)
    expect(abs(out[1, 50]) < 5,
           f"row-1 spike removed (was 100, became {out[1, 50]:.2f})")
    expect(np.allclose(out[0], matrix[0], atol=0.1),
           "row 0 (spike-free) untouched")
    expect(np.allclose(out[2], matrix[2], atol=0.1),
           "row 2 (spike-free) untouched")


# ─── PCA fusion ────────────────────────────────────────────────────

def test_pca_fuse():
    section("PCA fusion")

    # Single row: should return mean-centered version
    row = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = pca_fuse(row)
    expect(out.ndim == 1 and len(out) == 5,
           f"single-row returns 1D length-5 (got shape {out.shape})")

    # Two correlated rows: PC1 should align with the dominant trend
    n = 200
    t = np.linspace(0, 4 * np.pi, n)
    sig = np.sin(t)
    rng = np.random.default_rng(1)
    rows = np.array([
        sig + rng.normal(0, 0.1, n),
        sig + rng.normal(0, 0.1, n),
    ])
    fused = pca_fuse(rows)
    # Correlation with the underlying signal should be high (in absolute value)
    corr = abs(np.corrcoef(fused, sig)[0, 1])
    expect(corr > 0.95,
           f"PC1 of two correlated rows tracks underlying signal (|corr|={corr:.3f})")


# ─── band_power ────────────────────────────────────────────────────

def test_band_power():
    section("band_power")

    fs = 20.0
    n = 1200  # 60 seconds
    t = np.arange(n) / fs

    # Pure 1.0 Hz tone
    tone = np.sin(2 * np.pi * 1.0 * t)
    p_in = band_power(tone, fs, 0.8, 1.2)
    p_out = band_power(tone, fs, 2.5, 4.0)
    expect(p_in > p_out * 100,
           f"1 Hz tone: in-band power >> out-of-band ({p_in:.4f} vs {p_out:.6f})")

    # Pure 3 Hz tone
    tone3 = np.sin(2 * np.pi * 3.0 * t)
    p_card = band_power(tone3, fs, 0.8, 2.0)
    p_noise = band_power(tone3, fs, 2.5, 4.0)
    expect(p_noise > p_card * 100,
           f"3 Hz tone: noise band >> cardiac band ({p_noise:.4f} vs {p_card:.6f})")

    # Empty / degenerate
    expect(band_power(np.array([1, 2, 3]), fs, 0.5, 1.0) == 0.0,
           "too-short signal returns 0")
    expect(band_power(tone, 0.0, 0.5, 1.0) == 0.0,
           "zero sample rate returns 0")


# ─── cardiac_snr_db ────────────────────────────────────────────────

def test_cardiac_snr_db():
    section("cardiac_snr_db")

    fs = 20.0
    n = 1200
    t = np.arange(n) / fs

    # Pure cardiac at 1.25 Hz (75 BPM): SNR should be very high
    pure_card = np.sin(2 * np.pi * 1.25 * t)
    snr = cardiac_snr_db(pure_card, fs)
    expect(snr > 30,
           f"pure cardiac at 75 BPM: SNR > 30 dB (got {snr:.1f})")

    # Pure noise: SNR should be near 0
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, n)
    snr = cardiac_snr_db(noise, fs)
    expect(abs(snr) < 5,
           f"white noise: SNR near 0 (got {snr:.1f} dB)")

    # Mix: 75 BPM cardiac + noise: SNR positive but moderate
    mixed = 0.3 * pure_card + 1.0 * noise
    snr = cardiac_snr_db(mixed, fs)
    expect(0 < snr < 20,
           f"cardiac + noise mix: 0 < SNR < 20 (got {snr:.1f} dB)")


# ─── estimate_bpm ──────────────────────────────────────────────────

def test_estimate_bpm():
    section("estimate_bpm")

    fs = 20.0
    n = 1200
    t = np.arange(n) / fs

    for true_bpm in (50, 60, 75, 90, 110, 119):
        f_card = true_bpm / 60.0
        if f_card < CARDIAC_LOW_HZ or f_card > CARDIAC_HIGH_HZ:
            continue
        sig = np.sin(2 * np.pi * f_card * t)
        bpm, prom, conf = estimate_bpm(sig, fs)
        expect(bpm is not None and abs(bpm - true_bpm) <= 2,
               f"BPM={true_bpm}: extracted {bpm} (within ±2)")
        expect(conf > 0.9,
               f"BPM={true_bpm}: confidence high ({conf:.2f})")

    # Out-of-band: 30 BPM (0.5 Hz, below cardiac).
    # FFT leakage from a strong low-freq tone produces SOME spurious peak
    # in the cardiac band. The expected confidence is "lower than a real
    # cardiac signal" rather than "absolutely below 0.3": a 0.5 Hz tone
    # at our 60-second window leaks enough that conf can reach ~0.5.
    # Documented limitation: L2 extractable threshold (conf > 0.3) may
    # fire on strong out-of-band tones; relying on it alone is unsafe.
    # Use the joint L1+L2+L3 criteria instead.
    sig = np.sin(2 * np.pi * 0.5 * t)
    bpm_oob, prom_oob, conf_oob = estimate_bpm(sig, fs)
    bpm_real, prom_real, conf_real = estimate_bpm(
        np.sin(2 * np.pi * 1.25 * t), fs)
    expect(conf_real - conf_oob > 0.2,
           f"in-band confidence ({conf_real:.2f}) exceeds out-of-band "
           f"({conf_oob:.2f}) by ≥0.2")

    # Empty input
    bpm, _, _ = estimate_bpm(np.array([1, 2, 3]), fs)
    expect(bpm is None, "too-short signal returns BPM=None")


# ─── select_subcarriers ────────────────────────────────────────────

def test_select_subcarriers():
    section("select_subcarriers (HSR)")

    fs = 20.0
    n = 1200
    n_sc = 64
    rng = np.random.default_rng(0)

    # All subcarriers white noise
    matrix = rng.normal(0, 1.0, (n_sc, n))
    selected = select_subcarriers(matrix, fs, top_k=8)
    expect(len(selected) == 8,
           f"top_k=8 returns 8 indices (got {len(selected)})")

    # Subcarriers 30-37 have strong cardiac, others noise
    t = np.arange(n) / fs
    cardiac = np.sin(2 * np.pi * 1.25 * t)
    matrix = rng.normal(0, 1.0, (n_sc, n))
    for k in range(30, 38):
        matrix[k] += 5 * cardiac  # strong cardiac component
    selected = select_subcarriers(matrix, fs, top_k=8)
    overlap = set(selected) & set(range(30, 38))
    expect(len(overlap) >= 6,
           f"HSR picks ≥6 of 8 cardiac-rich subcarriers (got {len(overlap)})")


# ─── estimate_sample_rate ──────────────────────────────────────────

def test_estimate_sample_rate():
    section("estimate_sample_rate")

    # Regular 20 Hz timestamps
    ts = np.arange(0, 60, 0.05)
    fs = estimate_sample_rate(ts)
    expect(abs(fs - 20.0) < 0.1,
           f"regular 20 Hz sequence: estimated fs={fs:.2f}")

    # Empty
    expect(estimate_sample_rate(np.array([])) == 0.0,
           "empty input returns 0")
    expect(estimate_sample_rate(np.array([1.0])) == 0.0,
           "single sample returns 0")


# ─── Run all ───────────────────────────────────────────────────────

def main():
    test_hampel()
    test_pca_fuse()
    test_band_power()
    test_cardiac_snr_db()
    test_estimate_bpm()
    test_select_subcarriers()
    test_estimate_sample_rate()

    print(f"\n  {PASSED} passed, {FAILED} failed")
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
