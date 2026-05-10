# CSI input format

The extractor consumes a session directory with the following layout:

```
sessions/<session_id>/
  session.json                 metadata (subject, room, sample rate, etc.)
  block1_cold_ambient.jsonl    per-block CSI captures
  block2_off_baseline.jsonl
  block3_on_measurement.jsonl
  block4_off_replay.jsonl
  block5_cold_ambient.jsonl
```

Block files are optional except `block2_off_baseline.jsonl`, which is
the canonical reference for HSR subcarrier locking.

## Block .jsonl format

Each line is a JSON object. The first line is metadata; subsequent
lines are CSI frames.

### Metadata line (line 1)

```json
{"_meta": true,
 "recorder_version": "1.0",
 "started_at": 1714876543.21,
 "started_at_iso": "2026-05-09T10:55:43+0200",
 "duration_target_s": 60.0,
 "udp_bind": "127.0.0.1:5005",
 "synthetic": false}
```

The `_meta` field is the marker. The extractor recognizes the line and
skips it during frame parsing.

### Frame lines (line 2 onward)

```json
{"t": 1714876543.234, "raw": "{\"amplitudes\": [...], \"rssi\": -65, \"seq\": 17}"}
```

Each frame has:
- `t`: receive timestamp, Unix-epoch seconds (float).
- `raw`: a JSON-encoded string of the actual CSI payload.

The double-encoding (`raw` is itself JSON-stringified) is a pragmatic
choice: the recorder layer doesn't have to parse the firmware's CSI
payload format, it just timestamps and stores the raw line.

### Inner CSI payload

Inside `raw`, the parsed object has:

```json
{"amplitudes": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.21, 9.45, 10.12, ...],
 "rssi": -65,
 "seq": 17,
 "timestamp_us": 850000}
```

- `amplitudes`: list of float, one per subcarrier. Length must be
  consistent across all frames in a block. Typical values: 64 (HT20
  ESP32), 30 (Intel IWL5300), 114 (BCM43455 nexmon). The first 6 and
  last 5 entries are typically zero (DC + guard subcarriers in HT20).
- `rssi`: integer, dBm. Used by the rig's quality-checker; not used
  directly by the extractor.
- `seq`: integer, 802.11 sequence number. Used to detect packet drops.
- `timestamp_us`: integer, microsecond timestamp. Used for sample-rate
  estimation if `t` resolution is insufficient. Optional.

The `amplitude` (singular) field is also accepted for backward
compatibility.

## session.json

```json
{
  "spec_version": "1.1",
  "session_id": "2026-05-09_1100_ab12",
  "subject": "A",
  "room": "canonical",
  "victim_config": "2x_S3_v1",
  "shield_config": "off",
  "countermeasure_id": "none",
  "traffic_pattern": "default",
  "block_duration_s": 60,
  "udp_port": 5005,
  "started_at": 1714876543.0,
  "ended_at": 1714876843.0,
  "duration_total_s": 300.0,
  "aborted": false,
  "notes": "OFF baseline #1",
  "blocks": [
    {"name": "block1_cold_ambient",   "label": "Cold ambient PRE",      "started_at": 1714876543.0, "ended_at": 1714876603.0, "stats": {"frames_received": 1200}},
    ...
  ]
}
```

The `blocks` list is the per-block timing reference. The `name` field
is what maps to the `<name>.jsonl` file in the session directory.

## Producing the format from your own hardware

### From an ESP32 with esp-csi-tool

The simplest path: write a small UDP listener that:
1. Receives the ESP32's CSI UDP packets.
2. Parses them into `{amplitudes: [...], rssi, seq}` JSON.
3. Writes one frame per line to `block<N>.jsonl` with a timestamp wrapper.

A reference recorder is in the [partition-sensing](TBD) companion
repository.

### From an Intel IWL5300 .dat file (linux-80211n-csitool)

See `examples/intel_iwl5300_dat_example.py`. The example uses the
`csiread` Python library to parse the binary `.dat` format and run the
extractor against the resulting amplitude matrix. It does NOT produce
session directory output (the captures are too short for the rig's
5-block protocol); it runs the extractor's primitives directly.

### From a BCM43455 (Pi 4) with nexmon CSI

Format conversion: nexmon CSI emits 256 raw FFT bins per packet (114
useful subcarriers after pilot/DC removal) as complex int16 I/Q. Take
the magnitude per subcarrier, dump the 114 useful bins as the
`amplitudes` array, wrap with timestamp + RSSI + seq from the nexmon
frame header. A reference converter is in the partition-sensing
companion.

## Sample rate considerations

The extractor estimates fs from the median of consecutive frame
timestamps. This works robustly for streams above 7.4 Hz; below that,
the inclusion floor in `docs/THRESHOLDS.md` rejects the analysis.

For an ESP32-S3 receiver with a beacon-only transmitter, expected fs
is ~10 Hz (beacon interval 100 ms). With null-data flooding from the
AP at 20-50 Hz, expected fs rises to 20-50 Hz. With a CBR data flood,
fs can reach the chip's airtime ceiling (typically 100+ Hz). Higher fs
gives richer harmonics and better BPM resolution; lower fs (closer to
the 7.4 Hz floor) still works for fundamental BPM but loses harmonic
information.

## Common pitfalls

### Batched per-frame timestamps (the most common recorder bug)

When a recorder writes one `time.time()` per `serial.read()` (or per
UDP recv) instead of per CSI frame, multiple frames returned in the
same syscall share the same outer `t`. Consecutive deltas collapse to
zero, the extractor's median-delta estimator divides by something
tiny, and `fs` blows up to thousands of Hz. Downstream FFT bins are
then miscomputed and BPM extraction fails or produces nonsense.

The right fix is at the recorder layer: stamp each frame from the
firmware's monotonic `timestamp_us` field, anchored to the first
frame's wall-clock `t`. See `examples/esp32_s3_serial_recorder.py`
for the canonical pattern.

The extractor also has a defensive fallback: if more than 50% of
consecutive outer-`t` deltas are below 1 ms (the marker of a batched
recorder), `load_block_amplitudes` reconstructs per-frame timestamps
from the inner `timestamp_us` field automatically. Recorders that
already do the right thing pass through unchanged. Recorders that
forget get auto-corrected on the read side.

This fallback only fires when the inner payload includes
`timestamp_us`. If your firmware doesn't emit that field, get the
recorder right: don't rely on `time.time()` per `serial.read()`.

### Subcarrier count drift mid-block

If the firmware switches between HT20 (64 subcarriers) and HT40 (128)
mid-capture, frames with a different `len(amplitudes)` than the first
parsed frame are silently dropped. This is intentional: mixing them
would corrupt the per-subcarrier Hampel filter. Pick one bandwidth at
the firmware level and stick with it.

### RSSI ignored by the extractor

The `rssi` field is recorded but not used by the extraction pipeline.
It's there for the rig's quality-checker and for human inspection of
the capture environment. Don't expect tuning RSSI to change BPM.
