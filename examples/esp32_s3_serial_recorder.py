#!/usr/bin/env python3
"""
Reference recorder: ESP32-S3 partition-sensing rig over serial.

Captures one block of CSI from the STA-CSI ESP32-S3 board's UART output
into the .jsonl format the partition-csi extractor consumes (see
docs/CSI_FORMAT.md).

This is the recorder side that pairs with the extractor in this repo.
The artwork's full-stack runtime adds presence detection, multi-block
session orchestration, and a UDP/WebSocket bridge; here we keep the
single-board, single-block path for clarity.

Hardware assumed:
- Two ESP32-S3 dev boards on a USB hub. One runs the AP firmware and
  broadcasts beacons at ~20 Hz; the other runs the STA-CSI firmware
  and emits one CSI:{...} line per received frame on its UART.
- Both firmwares from the partition-sensing companion repository
  (referenced from this repo's README).
- Subject seated still in front of the sensor pair, heart roughly at
  sensor height.

Usage:

    python3 examples/esp32_s3_serial_recorder.py \\
        --port /dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_<sn>-if00-port0 \\
        --duration 90 \\
        --output /tmp/cardiac_session/block2_off_baseline.jsonl

Then run the extractor:

    python3 src/extractor.py /tmp/cardiac_session

A minimal session.json is written alongside the block file so the
extractor recognizes the directory.

Why per-frame `timestamp_us` matters:

    The naive recorder writes one `time.time()` per `serial.read()`
    boundary. Because pyserial returns chunks of multiple buffered
    frames in a single read, several consecutive frames end up sharing
    the same Python timestamp, and the extractor's sample-rate
    estimator collapses to nonsense (we observed 31 kHz in a real
    capture that should have been 23 Hz). The fix used here: read the
    firmware's monotonic `timestamp_us` field per frame, anchor the
    first frame's wall-clock to its outer `t`, and offset each
    subsequent frame's `t` by the per-frame microsecond delta. The
    extractor also has a defensive fallback that does this
    reconstruction itself if the recorder forgot, but doing it here is
    the right primary location.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import serial


CSI_LINE_RE = re.compile(rb"CSI:(\{.*\})")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True,
                    help="Serial device of the STA-CSI ESP32-S3.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=float, default=90.0,
                    help="Capture duration in seconds.")
    ap.add_argument("--output", required=True,
                    help="Output .jsonl path. The extractor expects names "
                         "like blockN_<label>.jsonl inside a session dir.")
    ap.add_argument("--label", default="off_baseline",
                    help="Block label for session.json.")
    ap.add_argument("--subject", default="anonymous")
    ap.add_argument("--room", default="default")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    meta = {
        "_meta": True,
        "recorder_version": "esp32-s3-serial/1.0",
        "started_at": started,
        "started_at_iso": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "duration_target_s": args.duration,
        "udp_bind": "serial-direct",
        "synthetic": False,
    }

    first_t = None
    first_us = None
    frames_written = 0

    with serial.Serial(args.port, args.baud, timeout=1.0) as ser, \
         out.open("w") as fh:
        fh.write(json.dumps(meta) + "\n")

        buf = b""
        deadline = started + args.duration
        last_progress = started

        while time.time() < deadline:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                m = CSI_LINE_RE.search(line)
                if not m:
                    continue
                payload_bytes = m.group(1)
                try:
                    payload = json.loads(payload_bytes)
                except json.JSONDecodeError:
                    continue

                # Per-frame wall-clock reconstruction from monotonic
                # firmware microseconds.
                us = payload.get("timestamp_us")
                if us is None:
                    # Firmware lacks the field; fall back to read()-boundary
                    # time. The extractor's defensive fallback won't fire
                    # because there's no inner timestamp to use either.
                    t = time.time()
                else:
                    if first_t is None:
                        first_t = time.time()
                        first_us = us
                    t = first_t + (us - first_us) / 1e6

                fh.write(json.dumps(
                    {"t": t, "raw": payload_bytes.decode("utf-8")}) + "\n")
                frames_written += 1

            now = time.time()
            if now - last_progress >= 5.0:
                elapsed = now - started
                print(f"  {frames_written} frames captured, "
                      f"t={elapsed:.1f}s / {args.duration:.0f}s",
                      file=sys.stderr, flush=True)
                last_progress = now

    ended = time.time()
    duration = ended - started

    # Write a minimal session.json next to the block file so the
    # extractor recognizes the directory as a session.
    session_path = out.parent / "session.json"
    if not session_path.exists():
        sess = {
            "spec_version": "1.1",
            "session_id": time.strftime("%Y%m%d_%H%M%S",
                                        time.localtime(started)),
            "subject": args.subject,
            "room": args.room,
            "victim_config": "2x_S3_v1",
            "shield_config": "off",
            "countermeasure_id": "none",
            "traffic_pattern": "default",
            "block_duration_s": args.duration,
            "udp_port": 0,
            "started_at": started,
            "ended_at": ended,
            "duration_total_s": duration,
            "aborted": False,
            "notes": f"Recorded with esp32_s3_serial_recorder.py from {args.port}.",
            "blocks": [{
                "name": out.stem,
                "label": args.label,
                "started_at": started,
                "ended_at": ended,
                "stats": {"frames_received": frames_written},
            }],
        }
        session_path.write_text(json.dumps(sess, indent=2))
        print(f"Wrote {session_path}", file=sys.stderr)

    print(f"DONE: {frames_written} frames in {duration:.1f}s "
          f"({frames_written/duration:.1f} Hz) -> {out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
