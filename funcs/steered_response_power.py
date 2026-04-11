import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile


def speech_weight(freqs):
    """
    Smooth speech-shaped weighting curve applied inside SRP.
    Rises around 400 Hz, falls around 3500 Hz.
    Downweights low freqs (tiny phase diffs) and high freqs (spatial aliasing).
    """
    f  = np.asarray(freqs, dtype=np.float64)
    lo = 1.0 / (1.0 + np.exp(-(f - 400)  / 80))
    hi = 1.0 / (1.0 + np.exp( (f - 3500) / 500))
    return (lo * hi).astype(np.float32)


def diagnose_array(audio, fs, eps):
    print("  Channel RMS levels:")
    labels = ["LF", "LR", "RF", "RR"]
    rms = [np.sqrt(np.mean(audio[:, c] ** 2)) for c in range(4)]
    for lab, r in zip(labels, rms):
        bar = "█" * int(r / max(rms) * 20)
        print(f"    {lab}: {r:.4f}  {bar}")
    lfe = rms[0]**2 + rms[1]**2
    rfe = rms[2]**2 + rms[3]**2
    ffe = rms[0]**2 + rms[2]**2
    bre = rms[1]**2 + rms[3]**2
    print(f"  L/R : {10*np.log10((lfe+eps)/(rfe+eps)):+.1f} dB  "
          f"({'left' if lfe > rfe else 'right'} dominant)")
    print(f"  F/B : {10*np.log10((ffe+eps)/(bre+eps)):+.1f} dB  "
          f"({'front' if ffe > bre else 'back'} dominant)")


def srp_spectrum_for_band(audio, fs, mic_array, freq_band, nfft, smooth_sigma, eps):
    """SRP-PHAT for one frequency band with speech-shaped weighting."""
    azimuths = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    w     = speech_weight(freqs)
    w[(freqs < freq_band[0]) | (freqs > freq_band[1])] = 0.0

    X = np.array([
        pra.transform.stft.analysis(audio[:, m], nfft, nfft // 2).T * w[:, None]
        for m in range(audio.shape[1])
    ])

    doa = pra.doa.SRP(mic_array, fs, nfft=nfft, azimuth=azimuths, num_src=6)
    doa.locate_sources(X, freq_range=freq_band)

    spec = doa.grid.values.copy()
    spec = (spec - spec.min()) / (spec.max() - spec.min() + eps)
    spec_smooth = gaussian_filter1d(np.concatenate([spec]*3),
                                    sigma=smooth_sigma)[360:720]
    return spec_smooth, np.degrees(azimuths)


def find_speaker_angles(audio, fs, mic_array, freq_bands, min_dist_deg,
                        band_consensus, nfft, smooth_sigma, eps,
                        abs_threshold=0.3):
    """
    Run SRP independently per band, vote across bands, return speaker angles.
    Circular padding ensures peaks at 0°/360° are never missed.
    """
    min_dist = int(min_dist_deg / (360 / 360))
    pad      = min_dist

    band_peaks   = []
    band_spectra = []

    for band in freq_bands:
        spec, angles_deg = srp_spectrum_for_band(
            audio, fs, mic_array, band, nfft, smooth_sigma, eps
        )
        # Circular padding so 0° peak is never split across the boundary
        spec_pad  = np.concatenate([spec[-pad:], spec, spec[:pad]])
        peaks_pad, _ = find_peaks(spec_pad, distance=min_dist,
                                  prominence=0.05, height=abs_threshold)
        peaks_idx = peaks_pad - pad
        peaks_idx = peaks_idx[(peaks_idx >= 0) & (peaks_idx < len(spec))]
        band_peaks.append(peaks_idx)
        band_spectra.append((spec, angles_deg, band))

    # Accumulate votes + power across bands
    vote_map  = np.zeros(360)
    power_map = np.zeros(360)
    for b_idx, peaks_idx in enumerate(band_peaks):
        spec, _, _ = band_spectra[b_idx]
        for pidx in peaks_idx:
            lo = max(0, pidx - min_dist // 2)
            hi = min(360, pidx + min_dist // 2)
            vote_map[lo:hi]  += 1
            power_map[lo:hi] += spec[pidx]

    # Smooth vote map circularly
    vote_smooth = gaussian_filter1d(
        np.concatenate([vote_map]*3), sigma=2.0
    )[360:720]

    # Find peaks in vote map — circularly padded
    vote_pad      = np.concatenate([vote_smooth[-pad:], vote_smooth, vote_smooth[:pad]])
    vote_peaks_pad, _ = find_peaks(vote_pad, distance=min_dist,
                                   height=band_consensus - 0.5)
    vote_peaks_idx = vote_peaks_pad - pad
    vote_peaks_idx = vote_peaks_idx[
        (vote_peaks_idx >= 0) & (vote_peaks_idx < len(vote_smooth))
    ]

    if len(vote_peaks_idx) == 0:
        print(f"  No peaks at consensus={band_consensus}, relaxing to {band_consensus-1}")
        vote_peaks_pad, _ = find_peaks(vote_pad, distance=min_dist,
                                       height=band_consensus - 1.5)
        vote_peaks_idx = vote_peaks_pad - pad
        vote_peaks_idx = vote_peaks_idx[
            (vote_peaks_idx >= 0) & (vote_peaks_idx < len(vote_smooth))
        ]

    vote_peaks_idx = sorted(vote_peaks_idx,
                            key=lambda i: vote_smooth[i], reverse=True)

    angles_deg = np.linspace(0, 360, 360, endpoint=False)
    speaker_angles = [
        (float(angles_deg[i]), float(vote_smooth[i]),
         float(power_map[i] / (vote_map[i] + eps)))
        for i in vote_peaks_idx
    ]
    speaker_angles = sorted(speaker_angles, key=lambda x: x[0])
    return speaker_angles, band_spectra, vote_smooth, angles_deg


def run_srp_fullband(audio, fs, mic_array, n_speakers, nfft, smooth_sigma, eps,
                     freq_range=[300, 7000]):
    """Full-band SRP with speech weighting — for visualisation only."""
    azimuths   = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    angles_deg = np.degrees(azimuths)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    w     = speech_weight(freqs)
    w[(freqs < freq_range[0]) | (freqs > freq_range[1])] = 0.0

    X = np.array([
        pra.transform.stft.analysis(audio[:, m], nfft, nfft // 2).T * w[:, None]
        for m in range(audio.shape[1])
    ])

    doa = pra.doa.SRP(mic_array, fs, nfft=nfft,
                      azimuth=azimuths, num_src=n_speakers)
    doa.locate_sources(X, freq_range=freq_range)

    spec = doa.grid.values.copy()
    spec = (spec - spec.min()) / (spec.max() - spec.min() + eps)
    spec_smooth = gaussian_filter1d(
        np.concatenate([spec]*3), sigma=smooth_sigma
    )[360:720]
    return angles_deg, spec_smooth


def plot_results(band_spectra, vote_smooth, angles_deg,
                 speaker_angles, fullband_smooth, band_consensus, eps):
    n_bands = len(band_spectra)
    fig, axes = plt.subplots(2, n_bands + 1,
                             figsize=(4 * (n_bands + 1), 7))

    # Top row: per-band SRP spectra
    for col, (spec, ang, band) in enumerate(band_spectra):
        ax = axes[0, col]
        ax.plot(ang, spec, color='steelblue', lw=1.5)
        ax.fill_between(ang, spec, alpha=0.15, color='steelblue')
        ax.set_title(f"{band[0]}–{band[1]} Hz", fontsize=9)
        ax.set_xticks(range(0, 361, 90))
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("SRP power")

    # Top-right: vote map
    ax = axes[0, n_bands]
    ax.plot(angles_deg, vote_smooth, color='darkorange', lw=2)
    ax.fill_between(angles_deg, vote_smooth, alpha=0.2, color='darkorange')
    ax.axhline(band_consensus - 0.5, color='crimson', lw=1.5,
               linestyle='--', label=f'Consensus={band_consensus}')
    for ang, votes, _ in speaker_angles:
        ax.axvline(ang, color='crimson', lw=1.5, alpha=0.7)
        ax.text(ang + 2, votes + 0.1, f"{ang:.0f}°",
                color='crimson', fontsize=8)
    ax.set_title("Cross-band vote map", fontsize=9)
    ax.set_xticks(range(0, 361, 90))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom: full-band SRP (reference) + vote-map speaker markers
    ax_full = fig.add_subplot(2, 1, 2)
    for sp in axes[1]:
        sp.set_visible(False)
    ax_full.set_position([0.08, 0.05, 0.88, 0.38])
    ax_full.plot(angles_deg, fullband_smooth, color='steelblue', lw=2,
                 label='Full-band SRP (reference)')
    ax_full.fill_between(angles_deg, fullband_smooth, alpha=0.15, color='steelblue')
    ax_full.plot(angles_deg, vote_smooth / (vote_smooth.max() + eps),
                 color='darkorange', lw=1.5, linestyle='--',
                 label='Vote map (normalised)')
    for ang, votes, _ in speaker_angles:
        ax_full.axvline(ang, color='crimson', lw=2)
        ax_full.text(ang + 2, 0.92, f"{ang:.0f}°",
                     color='crimson', fontsize=9, fontweight='bold',
                     transform=ax_full.get_xaxis_transform())
    ax_full.set_title(f"Full-band SRP + vote-map markers — {len(speaker_angles)} speaker(s)")
    ax_full.set_xlabel("Azimuth (°)")
    ax_full.set_ylabel("SRP power (normalised)")
    ax_full.set_xticks(range(0, 361, 30))
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)

    plt.savefig("results/srp_crossband.png", dpi=150, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    # ── Load audio ────────────────────────────────────────────────────────
    fs, data = wavfile.read('recordings/mixture.wav')  # (samples, 4) → [LF, LR, RF, RR]
    data = data.astype(np.float32) / 32768.0

    # ── Microphone array geometry (metres) ────────────────────────────────
    d_lr = 0.085   # lateral spacing  → alias freq ≈ c/(2d) ≈ 2 kHz
    d_fb = 0.0075  # front-back offset → breaks left/right symmetry
    mic_array = np.array([
        [-d_lr, -d_lr,  d_lr,  d_lr],
        [ d_fb, -d_fb,  d_fb, -d_fb],
    ])

    # ── Constants ─────────────────────────────────────────────────────────
    eps     = 1e-8   # numerical stability floor
    C_SOUND = 343.0  # speed of sound (m/s)

    # ── Tuning hyperparameters ────────────────────────────────────────────
    NFFT           = 2048  # FFT size - 7.8 Hz/bin at 16 kHz
    SMOOTH_SIGMA   = 5     # Gaussian smoothing width (degrees)
    MIN_DIST_DEG   = 30    # minimum angular separation between speakers
    BAND_CONSENSUS = 2     # how many bands must agree to confirm a speaker

    # Bands shifted to cover speech energy more evenly.
    # Alias frequency for 85 mm spacing = c/(2d) = ca. 2 kHz, so we split more
    # finely below that and use one wider band above where phase is less reliable.
    FREQ_BANDS = [
        [300,   800],   # fundamental frequencies + low formants
        [800,  1800],   # core speech, most reliable phase
        [1800, 3500],   # upper formants, still mostly below alias freq
        [3500, 7000],   # higher frequencies, down-weighted by speech curve
    ]

    # ── Array diagnostics ─────────────────────────────────────────────────
    print("=== Array diagnostics ===")
    diagnose_array(data, fs, eps)

    # ── Cross-band SRP vote map ───────────────────────────────────────────
    print("\n=== Cross-band SRP speaker localisation ===")
    speaker_angles, band_spectra, vote_smooth, angles_deg = find_speaker_angles(
        data, fs, mic_array,
        freq_bands=FREQ_BANDS,
        min_dist_deg=MIN_DIST_DEG,
        band_consensus=BAND_CONSENSUS,
        nfft=NFFT,
        smooth_sigma=SMOOTH_SIGMA,
        eps=eps,
    )

    print(f"\nDetected {len(speaker_angles)} speaker(s):")
    for ang, votes, pwr in speaker_angles:
        print(f"  {ang:>6.1f}°  votes={votes:.1f}/{len(FREQ_BANDS)}  "
              f"mean_power={pwr:.3f}")

    # ── Full-band SRP (visual reference only) ─────────────────────────────
    print("\n=== Full-band SRP (visual reference) ===")
    _, fullband_smooth = run_srp_fullband(
        data, fs, mic_array, len(speaker_angles),
        nfft=NFFT, smooth_sigma=SMOOTH_SIGMA, eps=eps,
    )

    plot_results(
        band_spectra, vote_smooth, angles_deg,
        speaker_angles, fullband_smooth,
        band_consensus=BAND_CONSENSUS, eps=eps,
    )

    print("\nDone.")