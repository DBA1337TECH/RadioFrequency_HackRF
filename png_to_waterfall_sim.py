#!/usr/bin/env python3
"""
image_to_sdr_modes.py — Encode a PNG into CS8 IQ and simulate SDR waterfall.

Modes:
  --mode normal : map image into positive freqs only (0 → +BW/2).
  --mode full   : map entire image into ±BW/2 (appears doubled).
  --mode mirror : symmetric reflection across DC.

Features:
 1. Full-height image mapping (correct).
 2. Mode selection for how the PNG maps into spectrum.
 3. Simulation mode with --sim-time.
 4. Prints quarter, half, full timing commands for easy reruns.

Dependencies: numpy, Pillow, matplotlib
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# -----------------------------
# Helpers
# -----------------------------
def load_png(path, target_rows):
    """Load and resize PNG to target rows, grayscale, flip vertically (top = high freq)."""
    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.resize((w, target_rows), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.flipud(arr)  # top → high frequency
    return arr  # shape: (rows, cols)

def encode_cs8(arr, samp_rate=2e6, fft_size=1024, col_time=0.01,
               db_range=30.0, mode="normal"):
    """
    Encode image array into CS8 IQ.
    Each column = one time slice. Each row = one FFT bin (frequency).
    """
    rows, cols = arr.shape
    mags = 10 ** ((arr*0.0 - (1.0 - arr)*db_range) / 20.0)
    rng = np.random.default_rng(123)
    blocks = []
    col_samps = int(col_time * samp_rate)

    for c in range(cols):
        spec = np.zeros(fft_size, dtype=np.complex64)

        if mode == "normal":
            # Fit image rows only into upper half (positive freqs)
            pos_len = min(rows, fft_size // 2)
            row_bins = np.linspace(fft_size-1, 1, rows, dtype=int)
            phases = rng.uniform(0, 2*np.pi, pos_len)
            spec[row_bins] = mags[:pos_len, c] * np.exp(1j*phases)

        elif mode == "full":
            half = rows // 2
            # Top half of image → positive freqs
            pos_bins = np.linspace(fft_size//2, fft_size-1, half, dtype=int)
            phases = rng.uniform(0, 2*np.pi, half)
            spec[pos_bins] = mags[:half, c] * np.exp(1j*phases)

            # Bottom half of image → negative freqs
            neg_bins = np.linspace(1, fft_size//2-1, rows-half, dtype=int)
            phases = rng.uniform(0, 2*np.pi, rows-half)
            spec[neg_bins] = mags[half:, c] * np.exp(1j*phases)

        elif mode == "mirror":
            # Map into positive freqs then mirror into negative
            pos_len = min(rows, fft_size // 2 - 1)
            row_bins = np.linspace(fft_size-1, 1, rows, dtype=int)
            phases = rng.uniform(0, 2*np.pi, pos_len)
            spec[row_bins] = mags[:pos_len, c] * np.exp(1j*phases)
            # Mirror into negative frequencies
            spec[-pos_len:] = np.conj(spec[row_bins])[::-1]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # IFFT → time-domain block
        block = np.fft.ifft(spec).astype(np.complex64)
        reps = int(np.ceil(col_samps / fft_size))
        block = np.tile(block, reps)[:col_samps]
        blocks.append(block)

    iq = np.concatenate(blocks)

    # Normalize to int8
    peak = np.max(np.abs(iq))
    if peak < 1e-9:
        peak = 1.0
    scale = 0.9 * 127.0 / peak
    i8 = np.clip(np.round(iq.real * scale), -127, 127).astype(np.int8)
    q8 = np.clip(np.round(iq.imag * scale), -127, 127).astype(np.int8)

    interleaved = np.empty(i8.size*2, dtype=np.int8)
    interleaved[0::2] = i8
    interleaved[1::2] = q8
    return interleaved

def read_cs8(path):
    """Read CS8 IQ file back into complex float array."""
    raw = np.fromfile(path, dtype=np.int8)
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return (i/127.0) + 1j*(q/127.0)

def simulate_waterfall(iq, samp_rate, center_freq, fft_size=1024,
                       overlap=0.5, drange=60, out_png="preview.png"):
    """Simulate SDR waterfall from IQ samples."""
    hop = int(fft_size*(1-overlap))
    if hop <= 0: hop = 1
    frames = np.lib.stride_tricks.sliding_window_view(iq, fft_size)[::hop]
    S = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    db = 20*np.log10(np.abs(S)+1e-12)
    vmax = np.max(db)
    vmin = vmax - drange

    # Frequency axis in MHz
    freqs = np.linspace(center_freq - samp_rate/2,
                        center_freq + samp_rate/2, fft_size, endpoint=False)/1e6

    plt.figure(figsize=(10,5))
    plt.imshow(db.T[::-1,:], aspect="auto", cmap="inferno",
               extent=[0, len(iq)/samp_rate, freqs[0], freqs[-1]],
               vmin=vmin, vmax=vmax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"Simulated Waterfall @ {center_freq/1e6:.3f} MHz (BW {samp_rate/1e6:.2f} MHz)")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved waterfall preview → {out_png}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png", help="Input PNG")
    ap.add_argument("--out-iq", default="out.cs8", help="Output CS8 file")
    ap.add_argument("--out-waterfall", default="preview.png", help="Waterfall preview PNG")
    ap.add_argument("-s", "--samp-rate", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--fft-size", type=int, default=1024, help="FFT size / image height")
    ap.add_argument("--col-time", type=float, default=0.01, help="Seconds per image column")
    ap.add_argument("--center-freq", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--sim-time", type=float, default=None, help="Simulation duration limit (s)")
    ap.add_argument("--mode", choices=["normal","full","mirror"], default="normal",
                    help="Spectrum mapping mode: normal | full | mirror")
    args = ap.parse_args()

    # Step 1: Load image
    arr = load_png(args.png, target_rows=args.fft_size)
    rows, cols = arr.shape
    total_time = cols * args.col_time

    # Print suggested runs
    print("\n--- Suggested run commands ---")
    for frac, label in [(0.25, "Quarter"), (0.5, "Half"), (1.0, "Full")]:
        sim_time = total_time * frac
        cmd = (f"python3 {os.path.basename(__file__)} {args.png} "
               f"--out-iq {args.out_iq} --out-waterfall {args.out_waterfall} "
               f"-s {args.samp_rate} --fft-size {args.fft_size} "
               f"--col-time {args.col_time} --center-freq {args.center_freq} "
               f"--mode {args.mode} --sim-time {sim_time:.2f}")
        print(f"{label} image (~{sim_time:.2f}s):\n  {cmd}\n")

    # Step 2: Encode to CS8
    cs8 = encode_cs8(arr, samp_rate=args.samp_rate, fft_size=args.fft_size,
                     col_time=args.col_time, mode=args.mode)

    with open(args.out_iq, "wb") as f:
        f.write(cs8.tobytes())
    print(f"Saved CS8 IQ file → {args.out_iq}")

    # Step 3: Simulate waterfall
    iq = read_cs8(args.out_iq)

    if args.sim_time is not None:
        nmax = int(args.sim_time * args.samp_rate)
        iq = iq[:nmax]
        print(f"Simulating only first {args.sim_time:.2f}s of signal...")

    simulate_waterfall(iq, args.samp_rate, args.center_freq,
                       fft_size=args.fft_size, out_png=args.out_waterfall)

if __name__ == "__main__":
    main()
