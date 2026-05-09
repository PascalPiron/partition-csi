# Threshold Derivations

The extractor's L1, L2, and L3 thresholds are anchored in published
WiFi-CSI cardiac-extraction literature. This document records the
chain of reasoning so the choices are auditable.

## 1. Literature consensus on the cardiac frequency band

| Paper | Cardiac band (Hz) | Equivalent BPM |
|---|---|---|
| Pulse-Fi (UCSC, 2024-25) | 0.8 - 2.17 | 48 - 130 |
| Non-Contact HR Monitoring (MDPI Sensors 2024) | 0.8 - 2.5 | 48 - 150 |
| WiCG (Trans Sensor Networks 2024) | 0.8 - 2.5 | 48 - 150 |
| Liu et al. (arXiv 1908.05108, 2020) | 1.0 - 2.0 | 60 - 120 |
| Wavelet decoupling 2024 | 0.8 - 2.5 | 48 - 150 |
| **partition-csi extractor** | **0.8 - 2.0** | **48 - 120** |

The extractor uses 0.8-2.0 Hz, conservative vs. the literature
consensus of 0.8-2.5 Hz. Rationale: the canonical use case is a
subject at rest (HR not exceeding 120 BPM under the experimental
protocol), and widening the band would absorb the second harmonic
of 60-75 BPM cardiac (which falls at 2.0-2.5 Hz), contaminating
the SNR estimate.

The noise reference band is 2.5-4.0 Hz: above the second harmonics
of typical resting cardiac, below the 5+ Hz region where motion
artifacts and ambient drift dominate.

## 2. Sample-rate floor

Pulse-Fi's E-Health configuration captures cardiac at 7.4 Hz
(WiFi pings every 136 ms) and reports MAE 0.17 BPM at a 30-second
window. This is the lowest documented rate at which WiFi-CSI cardiac
extraction is shown to work; the extractor uses **fs >= 7.4 Hz** as
the inclusion floor for sessions to be considered analyzable.

Captures below 7.4 Hz can still be passed to the extractor, but the
output L2 confidence will reflect the under-sampled regime and is
likely to fall below the extractable threshold.

## 3. L1 detection threshold

**Value: a block is "subject present" if its cardiac-band SNR exceeds
the cold-ambient mean by at least 1.0 dB.**

Derivation: the cold-ambient blocks measure the noise-floor SNR of the
hardware. Their within-session standard deviation provides an empirical
estimate of ambient block-to-block SNR variance. A threshold of
3-sigma is the standard "this is not noise" criterion in spectroscopy
and signal processing. Under realistic synthetic conditions, the
ambient SNR standard deviation is approximately 0.3 dB block-to-block;
3-sigma is approximately 0.9 dB, which round-trips to the chosen
1.0 dB.

For a specific deployment, this should be re-derived as 3-sigma from
the empirical real-bench sigma_ambient measured across the cold-ambient
blocks of the first few sessions.

## 4. L2 cardiac extraction confidence threshold

**Value: a block's cardiac is "extractable" if confidence > 0.3, where
confidence = clip((prominence - 2.0) / 6.0, 0, 1) and prominence =
peak / band_mean.**

Translation: confidence > 0.3 is equivalent to peak / band_mean > 3.8,
i.e., the FFT peak in the cardiac band is at least 3.8x the average
band amplitude.

Derivation:

- **Spectroscopy rule of thumb (Harris, "Quantitative Chemical
  Analysis", 9th ed.):** a peak is detectable when peak amplitude is
  at least 3x the local noise standard deviation. For a chi-squared
  power spectrum (FFT magnitude squared distribution), the noise
  standard deviation equals the noise mean, so peak/band_mean > 3
  corresponds approximately to peak SNR > 3.
- **Welch's method standard practice:** spectral peak detection
  typically uses 3-5x prominence-over-mean as the detection threshold.
- **WiCG (Trans Sensor Networks 2024)** defines its cardiac SNR as
  the energy in a 0.2 Hz bin centered on the true HR divided by the
  total energy in the 0.8-2.5 Hz band. A 0.2 Hz bin within a 1.7 Hz
  band represents 0.118 of the band uniformly; a peak that is 3x
  uniform corresponds to 35% of total energy in the bin, i.e.,
  approximately 4.7 dB above flat-spectrum baseline. WiCG's
  detection-success criterion is implicit (they report HR error,
  not pass/fail) but this 4-5 dB-above-baseline range is consistent
  with the extractor's prominence > 3.8 (~5.8 dB above baseline)
  threshold.

**Important caveat:** the L2 confidence threshold is necessary but
not sufficient for cardiac detection. Out-of-band tones can leak
into the cardiac band via FFT spectral leakage and trigger
conf > 0.3. The joint L1 + L2 + L3 acceptance is what makes the
extraction trustworthy.

## 5. L3 quantitative SNR loss threshold

**Value: in countermeasure-evaluation contexts (e.g. shield-OFF vs
shield-ON), L3 SNR loss mean >= 10 dB with bootstrap 95% CI lower
bound >= 6 dB constitutes a successful countermeasure.**

Derivation:

- **General signal-suppression literature (Tse & Viswanath,
  "Fundamentals of Wireless Communication", 2005):** a 10 dB SNR
  reduction is the canonical "order of magnitude" threshold for
  rendering a signal undetectable to a coherent receiver. A 6 dB
  reduction is the "halving of signal power" threshold and is
  considered the minimum perceivable degradation in audio/signal
  engineering.
- **WiFi jamming literature** (Pelechrinis et al. 2011 survey):
  jammer effectiveness is typically reported as packet-delivery-ratio
  reduction or PHY-layer SNR margin loss; >=10 dB SNR reduction is
  the standard "jamming successful" criterion.

These thresholds apply to comparative evaluation of countermeasure
hardware. The extractor itself does not depend on them; they are
documented here for users who run the comparative-evaluation flow.

## Citations

- Pulse-Fi: A Low-Cost System for Accurate Heart Rate Monitoring
  Using Wi-Fi Channel State Information. UCSC iNRG group. arXiv
  2510.24744 (2024). https://arxiv.org/html/2510.24744v1
- Non-Contact Heart Rate Monitoring Method Based on Wi-Fi CSI Signal.
  MDPI Sensors 24(7):2111 (2024). PMC11013971.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11013971/
- WiCG: Heartbeat Sensing Using COTS WiFi Devices with Common
  Antenna. ACM Trans. Sensor Networks (2024).
  https://dl.acm.org/doi/10.1145/3748330
- Liu et al., WiFi-based Real-time Breathing and Heart Rate
  Monitoring during Sleep. arXiv 1908.05108 (2020).
  https://arxiv.org/abs/1908.05108
- A survey on vital signs monitoring based on Wi-Fi CSI data. PMC
  9375645 (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9375645/
- Adib et al., Smart Homes that Monitor Breathing and Heart Rate.
  ACM CHI 2015. https://witrack.csail.mit.edu/vitalradio/
- Wang et al., FullBreathe: Full Human Respiration Detection
  Exploiting Complementarity of CSI Phase and Amplitude of WiFi
  Signals. ACM IMWUT 2(3) (2018). doi:10.1145/3264958
- Tse & Viswanath, Fundamentals of Wireless Communication. Cambridge
  University Press, 2005. ISBN 978-0521845274.
- Pelechrinis et al., Denial-of-Service Attacks in Wireless Networks:
  The Case of Jammers. IEEE Communications Surveys & Tutorials 13(2),
  2011. doi:10.1109/SURV.2011.041110.00022.
- Harris, Quantitative Chemical Analysis, 9th ed. W.H. Freeman, 2015.
  ISBN 978-1464135385.
