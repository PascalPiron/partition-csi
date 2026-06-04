# partition-csi

Cardiac signal extraction from WiFi Channel State Information (CSI),
using only commodity off-the-shelf 802.11 hardware.

This is the analyzer backbone of [PARTITION](https://pascalpiron.com),
an artwork that listens through walls. A WiFi sensor reads the cardiac
and respiratory rhythm of an unsuspecting body, and a pen plotter
inscribes it in ink at the wavelength of the radio that found it
(12.5 cm). The drawing is the only place the trace is held.

The code in this repository is the offline analysis pipeline: given a
captured CSI stream, extract heart rate. It is the same pipeline the
artwork uses, separated from the artwork's hardware-specific runtime so
others can study it, validate it against their own data, or extend it.

## What's here

```
src/
  extractor.py       Pipeline: Hampel + HSR + PCA + FFT, in ~500 lines.
  synthetic.py       Synthetic-CSI generator with ground-truth BPM,
                     used by tests/. Models multipath, breathing,
                     cardiac, shield-attenuation, RSSI-coupled noise.
tests/
  test_extractor_units.py   33 function-level unit tests.
  validate_extractor.py     12 end-to-end ground-truth tests (synthetic
                            CSI in, BPM/SNR out, must match within
                            tolerance).
examples/
  intel_iwl5300_dat_example.py   Validate against public IWL5300
                                 captures (Gi-z/CSI-Data heart-rate
                                 dataset). Demonstrates that the
                                 confidence gate correctly refuses to
                                 commit BPM on insufficient inputs.
docs/
  THRESHOLDS.md      Literature anchoring for L1/L2/L3 thresholds.
  CSI_FORMAT.md      The .jsonl input format the extractor expects.
```

## Pipeline

1. **Hampel outlier rejection** per subcarrier (window 5, 3-sigma).
2. **Mean removal**.
3. **HSR (Heartbeat-to-Subcomponent Ratio) subcarrier selection**: pick
   the top K subcarriers whose in-band cardiac power is at least 5% of
   their total spectral power. Lock the selection to the OFF baseline
   block of the session, apply that same set to all blocks. This
   prevents the analyzer from reselecting noise-dominated subcarriers
   when cardiac is suppressed (which would mask any countermeasure
   effect and produce false negatives).
4. **PCA fusion** across selected subcarriers (first principal
   component).
5. **FFT band-power** for L3 SNR: ratio of in-band (0.8-2.0 Hz cardiac)
   to noise band (2.5-4.0 Hz, above strongest second-harmonic content
   of resting cardiac).
6. **FFT peak with 4x zero-padding** for L2 BPM. Confidence in [0,1]
   from peak prominence relative to band mean.

## Thresholds (literature-anchored)

| Layer | Threshold | Rationale |
|---|---|---|
| L1 detection | block SNR > ambient + 1.0 dB | ~3-sigma above ambient block-to-block variance under realistic conditions. |
| L2 extraction | confidence > 0.3 (prominence > 3.8) | ~5.8 dB above flat-spectrum baseline. Consistent with WiCG (Trans Sensor Networks 2024) 4-5 dB empirical detection band, and with the spectroscopy rule that a peak is detectable when its amplitude exceeds 3-5x the local noise. |
| L3 quantitative | mean ≥ 10 dB with 95% CI lower bound ≥ 6 dB | Tse-Viswanath jamming-effectiveness threshold: 10 dB SNR reduction is the canonical "order of magnitude" criterion; 6 dB is the "halving of signal power" threshold considered the minimum perceivable degradation. |
| Cardiac band | 0.8-2.0 Hz (48-120 BPM) | Conservative vs literature 0.8-2.5 Hz. The narrower band protects the SNR estimate from second-harmonic contamination of 60-75 BPM cardiac, which falls at 2.0-2.5 Hz. Subjects at rest in laboratory conditions do not exceed 120 BPM. |
| Sample rate floor | fs >= 7.4 Hz | Pulse-Fi (UCSC, 2024-25) E-Health configuration is the documented lower bound for WiFi-CSI cardiac extraction. |

Full citation chain in `docs/THRESHOLDS.md`.

## Known limitations

**Block stationarity assumption.** The extractor takes a single FFT
over the entire block to find the cardiac peak. This implicitly
assumes heart rate is approximately stationary across the block.
For brief captures (tens of seconds to ~2 minutes) on a subject at
rest, this holds well enough. For longer captures, natural heart
rate variability spreads spectral energy across multiple bins and
can lower SNR rather than raise it. Empirically observed in two
captures of the same seated subject: a 90-second block yielded
70.2 BPM at 2.55 dB SNR; a 10-minute block of the same subject in
the same position yielded 67.0 BPM at 1.16 dB SNR. **Longer is not
monotonically better with this pipeline. Keep blocks short.**

A sliding-window FFT with median peak across windows (per PulseFi
and WiCG) would handle HRV. It is not implemented here. The
single-block FFT is kept because it makes the entire spectral path
auditable as one transform; a windowed estimator is a future
addition that would change the math primitives.

**Single-AP geometry.** The pipeline assumes one transmitter and
one receiver, both stationary. Multi-AP fusion and receiver mobility
are out of scope.

**No respiratory-band reporting.** Respiration is a separate signal
in 0.1-0.5 Hz that the pipeline filters out by construction. If you
want respiratory rate, that's a different extractor; the math here
is cardiac-only.

## Quick start

```bash
# Verify the math on synthetic data with known ground truth
python3 tests/test_extractor_units.py    # 33 unit tests
python3 tests/validate_extractor.py      # 12 end-to-end tests

# Generate a synthetic session
python3 src/synthetic.py \
    --output /tmp/syn1 \
    --bpm 75 --shield-attenuation-db 0

# Run the extractor on it
python3 src/extractor.py /tmp/syn1
```

## Validation against external public data

```bash
pip install --user csiread
mkdir -p /tmp/csi_public_dataset
cd /tmp/csi_public_dataset
for bpm in 66 71 73 75 76 84 88 90 92; do
    curl -sL -o "${bpm}bpm.dat" \
        "https://raw.githubusercontent.com/Gi-z/CSI-Data/main/Internal/intel/Heart%20Rate/${bpm}bpm.dat"
done

cd ~/path/to/partition-csi
python3 examples/intel_iwl5300_dat_example.py \
    --dataset-dir /tmp/csi_public_dataset \
    --output /tmp/external_validation.md
```

Expected outcome: 0 of 7 captures pass the L2 confidence gate. This is
correct behavior. The captures are short (under 17 seconds), captured
on different hardware (Intel IWL5300), and the L2 gate is doing exactly
what it should: refusing to commit a BPM extraction it can't trust.

## Hardware

The artwork's runtime uses:
- 2x ESP32-S3 boards (one access point, one CSI receiver) running custom
  firmware in a companion repository (not yet public).
- The CSI receiver UART output is bridged to UDP / .jsonl by the
  serial-to-UDP daemon in that companion repository.
- This `partition-csi` repository consumes the .jsonl session
  directories that result.

For replication on different hardware (Intel IWL5300, Atheros, BCM43455
via nexmon, ESP-CSI-Tool), see `docs/CSI_FORMAT.md`.

## Why publish this

PARTITION is an artwork about wireless biometric surveillance. The
counter to surveillance is not secrecy of methods. It is general
literacy of the methods, so that surveillance loses asymmetric
advantage. The pipeline here is intentionally simple and well-cited so
that researchers, journalists, artists, and concerned citizens can run
it, audit it, and understand what their walls already know about them.

The defensive countermeasure (Shield) is a separate, open-source
project.

## Authorship and AI assistance

This code was generated and polished with **Claude Opus 4.7** (Anthropic, 2026)
working under the specification, architectural direction, and review of
Pascal Piron.

The mathematical pipeline (Hampel + HSR + PCA + FFT band-power) is
standard signal-processing methodology with citations in
`docs/THRESHOLDS.md`. The Python implementation was drafted by the
model and corrected through iterative testing against the synthetic
ground truth in `src/synthetic.py`. The 33 unit tests and 12
ground-truth tests are part of this disclosure: they exist so that any
AI-introduced bug has a chance of being caught before the code is
trusted. Run them yourself before relying on the output.

Architectural decisions, threshold choices, license selection,
repository structure, and the artwork context are Pascal's. The model
wrote, refactored, and tested the Python.

If you find a bug, please open an issue. The fact that the code came
through a model is not an excuse for the bug; it is a reason for
extra vigilance from both author and user.

## Citing this work

```bibtex
@software{piron_partition_csi_2026,
  author = {Piron, Pascal},
  title  = {{partition-csi}: WiFi-CSI cardiac extraction},
  year   = {2026},
  url    = {https://github.com/PascalPiron/partition-csi}
}
```

Companion artworks: PARTITION (the work proper), Public Record (a
parallel investigation of the Luxembourg Business Register's
data-staleness architecture).

## Citations

- Pulse-Fi (UCSC iNRG, 2024-25). arXiv 2510.24744.
- WiCG. ACM Trans. Sensor Networks (2024). doi:10.1145/3748330
- Liu et al. WiFi-based Real-time Breathing and Heart Rate
  Monitoring during Sleep. arXiv 1908.05108 (2020).
- Adib et al. Smart Homes that Monitor Breathing and Heart Rate.
  ACM CHI 2015 (Vital-Radio: FMCW radar, not commodity WiFi CSI;
  cited as vital-sign-from-RF prior art). https://witrack.csail.mit.edu/vitalradio/
- Wang et al. FullBreathe. ACM IMWUT 2(3) (2018).
  doi:10.1145/3264958
- Tse & Viswanath. Fundamentals of Wireless Communication.
  Cambridge University Press (2005).

## License

AGPL-3.0-or-later. See `LICENSE`.

In plain language: anyone can use, modify, or redistribute this code
for any purpose. If you modify it and run a service or product based on
your modified version, you must publish your modifications under the
same license. This protects the code from corporate appropriation
without forbidding any legitimate use.

## Author

Pascal Piron, Luxembourg. https://pascalpiron.com
