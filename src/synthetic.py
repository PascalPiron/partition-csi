#!/usr/bin/env python3
"""
Synthetic CSI generator for partition-csi validation.

Produces session directories that look exactly like real captures (same
.jsonl format, same session.json) but with controlled ground-truth
signals embedded. The point is to verify the extractor recovers what we
put in. Without this, every session report is built on math we have
not stress-tested.

Signal model
============
Real CSI from a body-in-the-room scenario contains:

  amp[k, t] = amp_baseline[k]
              * (1 + chest_resp[k] * sin(2π * f_resp * t)
                   + chest_card[k] * sin(2π * f_card * t))
              + noise[k, t]

where:
  - f_card is the cardiac frequency (BPM/60)
  - f_resp is the respiratory frequency (~0.2 Hz)
  - chest_card[k] is per-subcarrier cardiac modulation depth (small,
    concentrated in a few subcarriers near the body's strongest
    multipath)
  - chest_resp[k] is per-subcarrier respiratory modulation depth (larger
    than cardiac, broader subcarrier spread)
  - noise[k, t] is broadband Gaussian, RSSI-dependent variance

This generator implements that model with parameters controllable from
the command line, so we can sweep cardiac signal strength and verify
the analyzer's L3 SNR loss measurement responds linearly.

For shield-attenuation modeling, set --shield-attenuation-db. The
generator multiplies chest_card by 10^(-att/20) (amplitude scaling),
which is what a shield would do to the receivable cardiac
modulation in the propagation path.

Usage
=====
    python3 src/synthetic.py \\
        --output sessions/synthetic_test1 \\
        --bpm 75 --shield-attenuation-db 0

License: AGPL-3.0-or-later.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np


# ── Realism artifact toggles ──
# Real CSI from a body-in-the-room scenario contains slow drift, motion
# artifacts, multipath fading, packet-level discontinuities, and noise
# that is NOT independent across subcarriers. The base generator emits a
# clean approximation (sinusoid + Gaussian + small random walk). These
# helpers add controlled artifact overlays so the analyzer can be
# stress-tested against more-real-than-clean synthetic input.
#
# Self-critique that motivated this generator: "synthetic generator uses pure
# sinusoids + white Gaussian noise + a small random walk; real CSI has
# 1/f drift, large non-Gaussian motion artifacts, packet-level
# discontinuities, multipath fading on multiple timescales." This
# narrows that gap.
#
# Each helper is independently toggleable via --realism={low,med,high}.
# Backward compatibility: realism=low produces the original generator
# behavior so the 12-test validate_extractor suite continues to pass.

def one_over_f_noise(n_frames: int, rng: np.random.Generator,
                     amplitude: float) -> np.ndarray:
    """Generate 1/f (pink) noise via FFT-based filtering.

    Real CSI baseline drift is dominated by 1/f-like processes (slow
    environmental change, antenna heating, AGC adjustment). Pure-white
    drift in the original generator is unrealistically stationary.
    """
    if n_frames < 4 or amplitude <= 0:
        return np.zeros(n_frames)
    # Generate white noise, FFT, filter by 1/sqrt(f), inverse FFT
    white = rng.standard_normal(n_frames)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_frames)
    freqs[0] = 1.0  # avoid div-by-zero on DC
    filt = 1.0 / np.sqrt(freqs)
    filt[0] = 0.0  # no DC
    pink = np.fft.irfft(spec * filt, n=n_frames)
    # Normalize to unit standard deviation, then scale
    pink = pink / (np.std(pink) + 1e-15)
    return amplitude * pink


def motion_artifact_bursts(n_frames: int, sample_rate_hz: float,
                           rng: np.random.Generator,
                           burst_rate_hz: float,
                           burst_amplitude: float) -> np.ndarray:
    """Sparse impulsive bursts (motion of a body part, door, etc.).

    Real captures include occasional large transients from non-cardiac
    motion. Modeled as a Poisson process of exponentially-decaying
    impulses with random sign.
    """
    if n_frames < 4 or burst_rate_hz <= 0 or burst_amplitude <= 0:
        return np.zeros(n_frames)
    duration = n_frames / sample_rate_hz
    n_bursts = max(1, rng.poisson(burst_rate_hz * duration))
    out = np.zeros(n_frames)
    decay_samples = max(2, int(0.4 * sample_rate_hz))  # 0.4 s decay
    decay_kernel = np.exp(-np.arange(decay_samples) / (decay_samples * 0.3))
    for _ in range(n_bursts):
        start = rng.integers(0, n_frames)
        sign = rng.choice([-1, 1])
        amp = burst_amplitude * (0.5 + rng.exponential(0.7))
        end = min(n_frames, start + decay_samples)
        out[start:end] += sign * amp * decay_kernel[:end - start]
    return out


def multipath_fading(n_frames: int, sample_rate_hz: float,
                     rng: np.random.Generator,
                     slow_amp: float, slow_period_s: float,
                     fast_amp: float, fast_period_s: float) -> np.ndarray:
    """Multiplicative envelope mimicking multipath fading.

    Returns a per-frame multiplicative envelope (around 1.0) capturing
    slow drift (geometry change, person shifting) and fast drift
    (small antenna movement, air currents).
    """
    if n_frames < 4:
        return np.ones(n_frames)
    t = np.arange(n_frames) / sample_rate_hz
    slow = slow_amp * np.sin(2 * np.pi * t / slow_period_s + rng.uniform(0, 2 * np.pi))
    fast = fast_amp * np.sin(2 * np.pi * t / fast_period_s + rng.uniform(0, 2 * np.pi))
    # Add small jitter to phase
    fast = fast + 0.5 * fast_amp * rng.standard_normal(n_frames) * fast_amp
    return 1.0 + slow + fast


def correlated_subcarrier_noise(n_subcarriers: int, n_frames: int,
                                rng: np.random.Generator,
                                amplitude: float,
                                correlation_length: int = 4) -> np.ndarray:
    """Generate per-frame, per-subcarrier noise with subcarrier correlation.

    Real CSI noise is NOT independent across subcarriers: adjacent
    subcarriers share much of their multipath response and the receiver
    front-end couples them. Modeled as a moving-average filter applied
    to white noise across the subcarrier axis.
    """
    if n_subcarriers < 2 or n_frames < 1 or amplitude <= 0:
        return np.zeros((n_subcarriers, n_frames))
    white = rng.standard_normal((n_subcarriers, n_frames))
    # Moving average across subcarrier axis to introduce correlation
    if correlation_length > 1:
        kernel = np.ones(correlation_length) / correlation_length
        smoothed = np.zeros_like(white)
        for j in range(n_frames):
            smoothed[:, j] = np.convolve(white[:, j], kernel, mode="same")
        # Normalize (moving avg reduces variance)
        smoothed = smoothed * np.sqrt(correlation_length)
        return amplitude * smoothed
    return amplitude * white


def packet_drops(frames: list[dict], rng: np.random.Generator,
                 drop_rate: float, burst_drop_rate: float = 0.0,
                 max_burst: int = 5) -> list[dict]:
    """Introduce realistic packet-level discontinuities.

    Real captures lose frames to weak RSSI, channel contention, and
    USB/UDP buffer overflow. Two patterns:
    - Random-drop: each frame has independent prob `drop_rate` of
      being dropped.
    - Burst-drop: occasional bursts of consecutive drops (channel
      handover, AGC event), prob `burst_drop_rate` per frame, length
      uniform [1, max_burst].
    """
    if not frames:
        return frames
    keep = [True] * len(frames)
    if drop_rate > 0:
        random_drops = rng.random(len(frames)) < drop_rate
        for i, drop in enumerate(random_drops):
            if drop:
                keep[i] = False
    if burst_drop_rate > 0:
        # Random starting points for bursts
        burst_starts = rng.random(len(frames)) < burst_drop_rate
        for i, b in enumerate(burst_starts):
            if not b:
                continue
            burst_len = int(rng.integers(1, max_burst + 1))
            for j in range(i, min(len(frames), i + burst_len)):
                keep[j] = False
    return [f for f, k in zip(frames, keep) if k]


def generate_block(
    n_frames: int,
    n_subcarriers: int,
    sample_rate_hz: float,
    bpm: float,
    breathing_bpm: float,
    cardiac_modulation_depth: float,
    breathing_modulation_depth: float,
    noise_amplitude: float,
    body_present: bool,
    shield_attenuation_db: float,
    rssi_dbm: float,
    seed: int = 0,
    realism: str = "low",
) -> tuple[list[dict], dict]:
    """
    Generate one block of CSI frames.

    Returns (frames, ground_truth_metadata).

    realism: 'low' = clean signal + Gaussian noise + small white drift
             (backward compatible with the original generator).
             'med' = adds 1/f drift, packet drops, RSSI fluctuation.
             'high' = also adds multipath fading (slow + fast),
             motion artifacts, and subcarrier-correlated noise.
    """
    rng = np.random.default_rng(seed)
    f_card = bpm / 60.0
    f_resp = breathing_bpm / 60.0

    t = np.arange(n_frames) / sample_rate_hz

    # Baseline amplitude profile across subcarriers: bell shape, like a
    # typical multipath response. DC and edge subcarriers near zero
    # (matches the firmware's observed pattern).
    sc = np.arange(n_subcarriers)
    bell = np.exp(-((sc - n_subcarriers / 2) ** 2) / (n_subcarriers ** 2 / 8.0))
    bell[:6] = 0.0
    bell[-5:] = 0.0
    baseline = 8.0 * bell  # arbitrary units roughly matching observed amplitudes

    # Per-subcarrier modulation depth profiles. Cardiac modulation is
    # concentrated in a narrow subcarrier band (HSR-selectable); breathing
    # is broader.
    card_profile = np.exp(-((sc - n_subcarriers * 0.45) ** 2) / 24.0) * cardiac_modulation_depth
    resp_profile = np.exp(-((sc - n_subcarriers * 0.52) ** 2) / 80.0) * breathing_modulation_depth

    # Apply shield attenuation to cardiac modulation only (the shield
    # disrupts cardiac-extractable structure but not respiratory or
    # baseline multipath).
    if shield_attenuation_db > 0:
        card_profile *= 10 ** (-shield_attenuation_db / 20.0)

    # If no body present, set both modulations to zero. The block
    # captures only ambient noise and baseline multipath.
    if not body_present:
        card_profile *= 0.0
        resp_profile *= 0.0

    # Realism-dependent multiplicative envelope (multipath fading)
    if realism == "high":
        envelope = multipath_fading(
            n_frames, sample_rate_hz, rng,
            slow_amp=0.05, slow_period_s=30.0,    # ±5% over 30 s
            fast_amp=0.02, fast_period_s=2.5,     # ±2% over 2.5 s
        )
    else:
        envelope = np.ones(n_frames)

    # Realism-dependent additive impulsive bursts (motion)
    if realism == "high" and body_present:
        bursts = motion_artifact_bursts(
            n_frames, sample_rate_hz, rng,
            burst_rate_hz=0.05,                   # avg 1 burst per 20 s
            burst_amplitude=0.6 * noise_amplitude,
        )
    else:
        bursts = np.zeros(n_frames)

    # Realism-dependent correlated noise across subcarriers
    if realism in ("med", "high"):
        corr_noise = correlated_subcarrier_noise(
            n_subcarriers, n_frames, rng,
            amplitude=0.3 * noise_amplitude,
            correlation_length=4,
        )
    else:
        corr_noise = np.zeros((n_subcarriers, n_frames))

    # Build amplitude matrix (n_subcarriers, n_frames)
    amps = np.zeros((n_subcarriers, n_frames))
    card_signal = np.sin(2 * np.pi * f_card * t)
    resp_signal = np.sin(2 * np.pi * f_resp * t)
    for k in range(n_subcarriers):
        modulation = (1.0
                      + card_profile[k] * card_signal
                      + resp_profile[k] * resp_signal)
        # Per-subcarrier baseline drift. Low realism: small white random
        # walk (original behavior). Med/high: 1/f pink drift (realistic).
        if realism in ("med", "high"):
            drift = one_over_f_noise(
                n_frames, rng,
                amplitude=0.15 * baseline[k] if baseline[k] > 0 else 0.05,
            )
        else:
            drift = rng.normal(0, 0.02, n_frames).cumsum() * 0.1
        amps[k] = baseline[k] * modulation * envelope + drift
        # Per-subcarrier independent Gaussian noise (always present;
        # variance scales weakly with baseline).
        sigma = noise_amplitude * (0.3 + 0.7 * (baseline[k] / max(baseline.max(), 1)))
        amps[k] += rng.normal(0, sigma, n_frames)
        # Add correlated-noise component (med/high only)
        amps[k] += corr_noise[k]
        # Add motion-artifact bursts (high only, body_present only)
        amps[k] += bursts * (baseline[k] / max(baseline.max(), 1))

    # Clamp negatives to zero (real amplitudes are |H(k)| ≥ 0)
    amps = np.maximum(amps, 0.0)

    # Zero DC and edge subcarriers like real CSI firmware does (the
    # firmware leaves these as 0.00 in its output: the first 6 and last 5
    # subcarriers in HT20 are DC + guard subcarriers).
    amps[:6, :] = 0.0
    amps[-5:, :] = 0.0

    # Per-frame RSSI with realism-dependent fluctuation
    if realism in ("med", "high"):
        # Slow RSSI drift: ±5 dB over the block (channel state changes)
        rssi_drift = 5.0 * np.sin(2 * np.pi * t / max(n_frames / sample_rate_hz, 1.0)
                                  + rng.uniform(0, 2 * np.pi))
    else:
        rssi_drift = np.zeros(n_frames)

    # Build per-frame JSON payloads
    base_ts = time.time()
    frames = []
    for j in range(n_frames):
        amplitudes = [round(float(amps[k, j]), 2) for k in range(n_subcarriers)]
        rssi_inst = rssi_dbm + rssi_drift[j] + rng.normal(0, 1.0)
        payload = {
            "amplitudes": amplitudes,
            "rssi": int(round(rssi_inst)),
            "timestamp_us": int(j * (1e6 / sample_rate_hz)),
            "seq": j,
        }
        recv_ts = base_ts + j / sample_rate_hz
        frames.append({"t": recv_ts, "raw": json.dumps(payload)})

    # Realism-dependent packet drops
    n_pre_drop = len(frames)
    if realism == "med":
        frames = packet_drops(frames, rng, drop_rate=0.02, burst_drop_rate=0.001)
    elif realism == "high":
        frames = packet_drops(frames, rng, drop_rate=0.05, burst_drop_rate=0.003)
    n_dropped = n_pre_drop - len(frames)

    truth = {
        "block_n_frames": len(frames),
        "block_sample_rate_hz": sample_rate_hz,
        "block_bpm_truth": bpm if body_present else None,
        "block_body_present": body_present,
        "block_shield_attenuation_db_truth": shield_attenuation_db,
        "block_rssi_target_dbm": rssi_dbm,
        "block_realism": realism,
        "block_packets_dropped": n_dropped,
    }
    return frames, truth


def write_block(path: Path, frames: list[dict], duration_target_s: float):
    """Write frames to .jsonl with the meta line a recorder typically writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({
            "_meta": True,
            "recorder_version": "1.0",
            "started_at": frames[0]["t"] if frames else time.time(),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "duration_target_s": duration_target_s,
            "udp_bind": "synthetic",
            "synthetic": True,
        }) + "\n")
        for frame in frames:
            f.write(json.dumps(frame) + "\n")


def make_session(
    output_dir: Path,
    bpm: float,
    shield_attenuation_db: float,
    breathing_bpm: float,
    sample_rate_hz: float,
    duration_per_block_s: float,
    n_subcarriers: int,
    cardiac_modulation_depth: float,
    breathing_modulation_depth: float,
    noise_amplitude: float,
    rssi_dbm: float,
    seed: int,
    subject: str = "SYNTHETIC",
    room: str = "synthetic",
    victim_config: str = "synthetic",
    countermeasure_id: str | None = None,
    realism: str = "low",
) -> dict:
    """Generate a complete five-block session."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration_per_block_s * sample_rate_hz)

    blocks_meta = []
    truths = []

    block_specs = [
        # (filename, label, body_present, shield_att_db_for_block)
        ("block1_cold_ambient",   "Cold ambient PRE",      False, 0.0),
        ("block2_off_baseline",   "Shield OFF baseline",   True,  0.0),
        ("block3_on_measurement", "Shield ON measurement", True,  shield_attenuation_db),
        ("block4_off_replay",     "Shield OFF replay",     True,  0.0),
        ("block5_cold_ambient",   "Cold ambient POST",     False, 0.0),
    ]

    for i, (filename, label, body, att) in enumerate(block_specs):
        frames, truth = generate_block(
            n_frames=n_frames,
            n_subcarriers=n_subcarriers,
            sample_rate_hz=sample_rate_hz,
            bpm=bpm,
            breathing_bpm=breathing_bpm,
            cardiac_modulation_depth=cardiac_modulation_depth,
            breathing_modulation_depth=breathing_modulation_depth,
            noise_amplitude=noise_amplitude,
            body_present=body,
            shield_attenuation_db=att,
            rssi_dbm=rssi_dbm,
            seed=seed * 7 + i,
            realism=realism,
        )
        write_block(output_dir / f"{filename}.jsonl",
                    frames, duration_per_block_s)
        truth["block_label"] = label
        truth["block_filename"] = filename
        truths.append(truth)
        blocks_meta.append({
            "name": filename,
            "label": label,
            "started_at": frames[0]["t"] if frames else time.time(),
            "ended_at": frames[-1]["t"] if frames else time.time(),
            "stats": {
                "frames_received": len(frames),
                "rate_hz": sample_rate_hz,
                "synthetic": True,
            },
        })

    started_at = blocks_meta[0]["started_at"]
    ended_at = blocks_meta[-1]["ended_at"]

    if countermeasure_id is None:
        countermeasure_id = ("none" if shield_attenuation_db == 0
                             else f"SYN_{shield_attenuation_db:.0f}dB")
    session_meta = {
        "spec_version": "1.1",
        "session_id": output_dir.name,
        "subject": subject,
        "room": room,
        "victim_config": victim_config,
        "shield_config": "off" if shield_attenuation_db == 0 else f"synthetic_{shield_attenuation_db:.0f}dB",
        "countermeasure_id": countermeasure_id,
        "traffic_pattern": "synthetic",
        "block_duration_s": duration_per_block_s,
        "udp_port": 0,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_total_s": round(ended_at - started_at, 1),
        "aborted": False,
        "notes": (f"synthetic session: bpm={bpm}, "
                  f"shield_att={shield_attenuation_db}dB, seed={seed}"),
        "pre_flight_scan": {"synthetic": True},
        "blocks": blocks_meta,
        "blinded": False,
        "ground_truth": {
            "bpm": bpm,
            "shield_attenuation_db": shield_attenuation_db,
            "breathing_bpm": breathing_bpm,
            "sample_rate_hz": sample_rate_hz,
            "n_subcarriers": n_subcarriers,
            "cardiac_modulation_depth": cardiac_modulation_depth,
            "breathing_modulation_depth": breathing_modulation_depth,
            "noise_amplitude": noise_amplitude,
            "rssi_dbm": rssi_dbm,
            "realism": realism,
            "per_block": truths,
        },
    }

    (output_dir / "session.json").write_text(
        json.dumps(session_meta, indent=2) + "\n")
    return session_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output session directory")
    parser.add_argument("--bpm", type=float, default=75.0,
                        help="Cardiac frequency in BPM (default 75)")
    parser.add_argument("--shield-attenuation-db", type=float, default=0.0,
                        help="Shield attenuation applied in block 3 (default 0)")
    parser.add_argument("--breathing-bpm", type=float, default=12.0,
                        help="Respiratory rate in BPM (default 12)")
    parser.add_argument("--sample-rate", type=float, default=20.0,
                        help="CSI sample rate Hz (default 20)")
    parser.add_argument("--block-duration", type=float, default=60.0,
                        help="Per-block duration s (default 60)")
    parser.add_argument("--n-subcarriers", type=int, default=64,
                        help="Subcarriers per frame (default 64)")
    parser.add_argument("--cardiac-depth", type=float, default=0.04,
                        help="Cardiac modulation depth as fraction (default 0.04 = ±4%%)")
    parser.add_argument("--breathing-depth", type=float, default=0.12,
                        help="Breathing modulation depth as fraction (default 0.12)")
    parser.add_argument("--noise-amplitude", type=float, default=0.6,
                        help="Per-subcarrier noise amplitude (default 0.6)")
    parser.add_argument("--rssi", type=float, default=-65.0,
                        help="Mean RSSI dBm (default -65)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject", default="SYNTHETIC")
    parser.add_argument("--room", default="synthetic")
    parser.add_argument("--victim-config", default="synthetic")
    parser.add_argument("--countermeasure-id", default=None,
                        help="If unset: 'none' for 0 dB, 'SYN_<N>dB' otherwise")
    parser.add_argument("--realism", choices=["low", "med", "high"],
                        default="low",
                        help="Artifact realism level. low (default) = clean "
                             "signal + Gaussian noise + small white drift "
                             "(matches the 12-test validate_extractor suite). "
                             "med = adds 1/f drift, packet drops, RSSI "
                             "fluctuation. high = also adds multipath fading, "
                             "motion artifact bursts, subcarrier-correlated "
                             "noise. Use med/high to stress-test the analyzer "
                             "against more-real-than-clean conditions.")
    args = parser.parse_args()

    meta = make_session(
        output_dir=args.output,
        bpm=args.bpm,
        shield_attenuation_db=args.shield_attenuation_db,
        breathing_bpm=args.breathing_bpm,
        sample_rate_hz=args.sample_rate,
        duration_per_block_s=args.block_duration,
        n_subcarriers=args.n_subcarriers,
        cardiac_modulation_depth=args.cardiac_depth,
        breathing_modulation_depth=args.breathing_depth,
        noise_amplitude=args.noise_amplitude,
        rssi_dbm=args.rssi,
        seed=args.seed,
        subject=args.subject,
        room=args.room,
        victim_config=args.victim_config,
        countermeasure_id=args.countermeasure_id,
        realism=args.realism,
    )
    print(f"synthetic session written to {args.output}")
    print(f"  bpm={args.bpm}, shield_att={args.shield_attenuation_db}dB, "
          f"sample_rate={args.sample_rate}Hz, "
          f"frames/block={int(args.block_duration * args.sample_rate)}, "
          f"realism={args.realism}")


if __name__ == "__main__":
    main()
