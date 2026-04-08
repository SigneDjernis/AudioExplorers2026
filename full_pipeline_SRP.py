import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import scipy.signal as signal
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile

# ── Load audio ────────────────────────────────────────────────────────────────
fs, data = wavfile.read('Recordings/mixture.wav')  # (samples, 4) → [LF, LR, RF, RR]
data = data.astype(np.float32) / 32768.0

# ── Microphone array geometry (metres) ───────────────────────────────────────
d_lr = 0.085
d_fb = 0.0075
mic_array = np.array([
    [-d_lr, -d_lr,  d_lr,  d_lr],
    [ d_fb, -d_fb,  d_fb, -d_fb],
])

eps = 1e-8

# ── Tuning ────────────────────────────────────────────────────────────────────
NFFT           = 2048
SMOOTH_SIGMA   = 5
MIN_DIST_DEG   = 30
BEAMFORM_TOL   = 22

# Sub-bands used for cross-band speaker counting
# Real speakers appear consistently across bands; sidelobes don't
FREQ_BANDS = [
    [400,  1200],
    [1200, 2500],
    [2500, 4000],
    [4000, 8000],
]
# A candidate angle must appear in at least this many bands to count as a speaker
BAND_CONSENSUS = 2


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_array(audio, fs):
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Cross-band SRP speaker counting
# ══════════════════════════════════════════════════════════════════════════════

def srp_spectrum_for_band(audio, fs, mic_array, freq_band,
                          nfft=NFFT, smooth_sigma=SMOOTH_SIGMA):
    """Run SRP-PHAT for a single frequency band. Returns smoothed spectrum."""
    azimuths = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    X = np.array([
        pra.transform.stft.analysis(audio[:, m], nfft, nfft // 2).T
        for m in range(audio.shape[1])
    ])
    # num_src=6 — we want the full spectrum, not limited peaks
    doa = pra.doa.SRP(mic_array, fs, nfft=nfft, azimuth=azimuths, num_src=6)
    doa.locate_sources(X, freq_range=freq_band)

    spec = doa.grid.values.copy()
    spec = (spec - spec.min()) / (spec.max() - spec.min() + eps)
    spec_smooth = gaussian_filter1d(np.concatenate([spec]*3),
                                    sigma=smooth_sigma)[360:720]
    return spec_smooth, np.degrees(azimuths)


def count_and_locate_speakers(audio, fs, mic_array,
                               freq_bands=FREQ_BANDS,
                               min_dist_deg=MIN_DIST_DEG,
                               band_consensus=BAND_CONSENSUS,
                               abs_threshold=0.3):
    """
    Run SRP independently in each frequency sub-band and find peaks in each.
    A direction is counted as a real speaker only if it appears as a peak
    in at least `band_consensus` out of len(freq_bands) bands.

    Real speakers are broadband — they show up in multiple bands.
    Sidelobes and room reflections are narrowband — they appear in only 1-2 bands.

    Returns:
        consensus_angles : list of (angle_deg, vote_count, mean_power)
        band_spectra     : list of (spectrum, angles_deg) per band  [for plotting]
    """
    min_dist = int(min_dist_deg / (360 / 360))

    # Collect peak angles per band
    band_peaks  = []
    band_spectra = []

    for band in freq_bands:
        spec, angles_deg = srp_spectrum_for_band(audio, fs, mic_array, band)
        peaks_idx, props = find_peaks(spec, distance=min_dist, prominence=0.05,
                                      height=abs_threshold)
        band_peaks.append(peaks_idx)
        band_spectra.append((spec, angles_deg, band))

    # Vote: for each degree, count how many bands have a peak within ±min_dist/2
    vote_map   = np.zeros(360)
    power_map  = np.zeros(360)

    for b_idx, peaks_idx in enumerate(band_peaks):
        spec, angles_deg, _ = band_spectra[b_idx]
        for pidx in peaks_idx:
            lo = max(0, pidx - min_dist // 2)
            hi = min(360, pidx + min_dist // 2)
            vote_map[lo:hi]  += 1
            power_map[lo:hi] += spec[pidx]

    # Smooth the vote map to merge nearby votes
    vote_smooth = gaussian_filter1d(
        np.concatenate([vote_map]*3), sigma=2.0
    )[360:720]

    # Find peaks in the vote map that exceed consensus threshold
    vote_peaks_idx, _ = find_peaks(vote_smooth, distance=min_dist,
                                   height=band_consensus - 0.5)

    if len(vote_peaks_idx) == 0:
        # Relax consensus if nothing found
        print(f"  No peaks at consensus={band_consensus}, "
              f"relaxing to {band_consensus - 1}")
        vote_peaks_idx, _ = find_peaks(vote_smooth, distance=min_dist,
                                       height=band_consensus - 1.5)

    # Sort by vote count (most consistent = most likely real speaker)
    vote_peaks_idx = sorted(vote_peaks_idx,
                            key=lambda i: vote_smooth[i], reverse=True)

    angles_deg = np.linspace(0, 360, 360, endpoint=False)
    consensus_angles = [
        (float(angles_deg[i]), float(vote_smooth[i]),
         float(power_map[i] / (vote_map[i] + eps)))
        for i in vote_peaks_idx
    ]
    consensus_angles = sorted(consensus_angles, key=lambda x: x[0])
    return consensus_angles, band_spectra, vote_smooth, angles_deg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Full-band SRP for final localisation (using known count)
# ══════════════════════════════════════════════════════════════════════════════

def run_srp_fullband(audio, fs, mic_array, n_speakers,
                     nfft=NFFT, smooth_sigma=SMOOTH_SIGMA,
                     min_dist_deg=MIN_DIST_DEG,
                     freq_range=[400, 8000]):
    """
    Full-band SRP with the speaker count confirmed by cross-band voting.
    Picks exactly n_speakers peaks.
    """
    azimuths   = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    angles_deg = np.degrees(azimuths)

    X = np.array([
        pra.transform.stft.analysis(audio[:, m], nfft, nfft // 2).T
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

    min_dist  = int(min_dist_deg / (360 / len(spec_smooth)))
    peaks_idx, props = find_peaks(spec_smooth, distance=min_dist, prominence=0.0)

    if len(peaks_idx) < n_speakers:
        # Fallback: greedy suppression
        spec_tmp  = spec_smooth.copy()
        peaks_idx = []
        for _ in range(n_speakers):
            idx = int(np.argmax(spec_tmp))
            peaks_idx.append(idx)
            lo = max(0, idx - min_dist)
            hi = min(len(spec_tmp), idx + min_dist)
            spec_tmp[lo:hi] = -np.inf
    else:
        order     = np.argsort(props["prominences"])[::-1]
        peaks_idx = peaks_idx[order[:n_speakers]]

    peaks = sorted(
        [(angles_deg[i], float(spec_smooth[i])) for i in peaks_idx],
        key=lambda x: x[0]
    )
    return peaks, spec, spec_smooth, angles_deg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MVDR beamformer
# ══════════════════════════════════════════════════════════════════════════════

def beamform_at_angle(audio, fs, target_angle_deg,
                      tolerance_deg=BEAMFORM_TOL, nperseg=1024):
    """Soft directional mask + covariance MVDR. Returns mono signal."""
    audio = audio.astype(np.float32)
    _, _, Zxx = signal.stft(audio.T, fs=fs, nperseg=nperseg)
    C, F, T   = Zxx.shape
    mag       = np.abs(Zxx)

    front_energy = mag[0] + mag[2]
    back_energy  = mag[1] + mag[3]
    left_energy  = mag[0] + mag[1]
    right_energy = mag[2] + mag[3]

    # Direction scores
    fb_score = np.log((front_energy + eps) / (back_energy + eps))
    lr_score = np.log((left_energy + eps) / (right_energy + eps))
    angle_est = np.arctan2(lr_score, fb_score)
    angle_est = np.mod(angle_est, 2 * np.pi)

    target_rad  = np.deg2rad(target_angle_deg)
    tolerance   = np.deg2rad(tolerance_deg)
    angle_diff  = np.angle(np.exp(1j * (angle_est - target_rad)))
    target_mask = np.exp(-0.5 * (angle_diff / tolerance)**2).astype(np.float32)
    noise_mask  = 1.0 - target_mask

    enhanced = np.zeros((F, T), dtype=np.complex64)
    for f in range(F):
        Xf     = Zxx[:, f, :]
        Rs     = np.zeros((C, C), dtype=np.complex64)
        Rn     = np.zeros((C, C), dtype=np.complex64)
        ms_sum = np.sum(target_mask[f]) + eps
        mn_sum = np.sum(noise_mask[f])  + eps
        for t in range(T):
            x   = Xf[:, t:t+1]
            Rs += target_mask[f, t] * (x @ x.conj().T)
            Rn += noise_mask[f, t]  * (x @ x.conj().T)
        Rs /= ms_sum
        Rn /= mn_sum
        Rn += 1e-3 * np.eye(C, dtype=np.complex64)
        Rs += 1e-6 * np.eye(C, dtype=np.complex64)
        vals, vecs = eigh(Rs)
        d        = vecs[:, -1]
        d        = d / (d[0] + eps)
        Rn_inv_d = np.linalg.solve(Rn, d)
        w        = Rn_inv_d / (d.conj().T @ Rn_inv_d + eps)
        enhanced[f, :] = np.conj(w).T @ Xf

    _, y = signal.istft(enhanced, fs=fs, nperseg=nperseg)
    y    = np.real(y)
    y   /= np.max(np.abs(y)) + eps
    return y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Second-pass SRP refinement
# ══════════════════════════════════════════════════════════════════════════════

def isolate_angle_multichannel(audio, fs, target_angle_deg,
                                tolerance_deg=30, nperseg=1024):
    """Soft directional mask → masked 4-channel audio (phase intact)."""
    audio = audio.astype(np.float32)
    _, _, Zxx = signal.stft(audio.T, fs=fs, nperseg=nperseg)
    mag       = np.abs(Zxx)

    front_energy = mag[0] + mag[2]
    back_energy  = mag[1] + mag[3]
    left_energy  = mag[0] + mag[1]
    right_energy = mag[2] + mag[3]

    # Direction scores
    fb_score = np.log((front_energy + eps) / (back_energy + eps))
    lr_score = np.log((left_energy + eps) / (right_energy + eps))
    angle_est = np.arctan2(lr_score, fb_score) - np.deg2rad(90)
    angle_est = np.mod(angle_est, 2 * np.pi)

    angle_diff   = np.angle(np.exp(1j*(angle_est - np.deg2rad(target_angle_deg))))
    target_mask  = np.exp(-0.5*(angle_diff/np.deg2rad(tolerance_deg))**2).astype(np.float32)
    Zxx_filtered = Zxx * target_mask[None, :, :]

    _, filtered  = signal.istft(Zxx_filtered, fs=fs, nperseg=nperseg)
    filtered     = np.real(filtered).T
    filtered    /= np.max(np.abs(filtered)) + eps
    return filtered


def refine_angle(masked_audio, fs, mic_array,
                 rough_angle, search_width=30, nfft=NFFT):
    """Second-pass SRP in ±search_width window. Returns refined angle."""
    angles_all = np.linspace(0, 360, 360, endpoint=False)
    a_min  = (rough_angle - search_width) % 360
    a_max  = (rough_angle + search_width) % 360
    window = ((angles_all >= a_min) & (angles_all <= a_max)
              if a_min < a_max else
              (angles_all >= a_min) | (angles_all <= a_max))

    X = np.array([
        pra.transform.stft.analysis(masked_audio[:, m], nfft, nfft // 2).T
        for m in range(masked_audio.shape[1])
    ])
    doa = pra.doa.SRP(mic_array, fs, nfft=nfft,
                      azimuth=np.deg2rad(angles_all[window]), num_src=1)
    doa.locate_sources(X, freq_range=[400, 8000])
    return float(angles_all[window][np.argmax(doa.grid.values)])


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(band_spectra, vote_smooth, angles_deg,
                 final_peaks, consensus_angles, fullband_smooth):
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
    ax.axhline(BAND_CONSENSUS - 0.5, color='crimson', lw=1.5,
               linestyle='--', label=f'Consensus={BAND_CONSENSUS}')
    for ang, votes, _ in consensus_angles:
        ax.axvline(ang, color='crimson', lw=1.5, alpha=0.7)
        ax.text(ang + 2, votes + 0.1, f"{ang:.0f}°",
                color='crimson', fontsize=8)
    ax.set_title("Cross-band vote map", fontsize=9)
    ax.set_xticks(range(0, 361, 90))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom row: full-band SRP with final peaks (spans full width)
    ax_full = fig.add_subplot(2, 1, 2)
    for sp in axes[1]:
        sp.set_visible(False)
    ax_full.set_position([0.08, 0.05, 0.88, 0.38])
    ax_full.plot(angles_deg, fullband_smooth, color='steelblue', lw=2,
                 label='Full-band SRP (smoothed)')
    ax_full.fill_between(angles_deg, fullband_smooth, alpha=0.15, color='steelblue')
    for ang, pwr in final_peaks:
        ax_full.axvline(ang, color='crimson', lw=2)
        ax_full.text(ang + 2, 0.92, f"{ang:.0f}°",
                     color='crimson', fontsize=9, fontweight='bold',
                     transform=ax_full.get_xaxis_transform())
    ax_full.set_title(f"Full-band SRP — final {len(final_peaks)} speaker(s)")
    ax_full.set_xlabel("Azimuth (°)")
    ax_full.set_ylabel("SRP power (normalised)")
    ax_full.set_xticks(range(0, 361, 30))
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)

    plt.savefig("srp_crossband.png", dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════ 

if __name__ == "__main__":

    print("=== Array diagnostics ===")
    diagnose_array(data, fs)

    # ── Step 1: cross-band speaker counting + rough localisation ──────────
    print("\n=== Step 1: Cross-band SRP speaker counting ===")
    consensus_angles, band_spectra, vote_smooth, angles_deg = \
        count_and_locate_speakers(data, fs, mic_array)

    n_speakers = len(consensus_angles)
    print(f"\nCross-band consensus: {n_speakers} speaker(s)")
    for ang, votes, pwr in consensus_angles:
        print(f"  {ang:>6.1f}°  votes={votes:.1f}/{len(FREQ_BANDS)}  "
              f"mean_power={pwr:.3f}")

    # ── Step 2: full-band SRP with confirmed count ─────────────────────────
    print(f"\n=== Step 2: Full-band SRP (n={n_speakers}) ===")
    final_peaks, spec_raw, spec_smooth, angles_deg_fb = run_srp_fullband(
        data, fs, mic_array, n_speakers
    )
    print("Final peak angles:")
    for ang, pwr in final_peaks:
        print(f"  {ang:>6.1f}°  (power: {pwr:.3f})")

    plot_results(band_spectra, vote_smooth, angles_deg,
                 final_peaks, consensus_angles, spec_smooth)

    # ── Step 3: refine + beamform ──────────────────────────────────────────
    print("\n=== Step 3: Refine + beamform ===")
    for rough_angle, pwr in final_peaks:
        print(f"\nSpeaker at ~{rough_angle:.0f}°")
        masked      = isolate_angle_multichannel(data, fs, rough_angle)
        refined     = refine_angle(masked, fs, mic_array, rough_angle)
        shift       = abs(refined - rough_angle)
        final_angle = refined if shift > 3 else rough_angle
        if shift > 3:
            print(f"  {rough_angle:.1f}° → refined {refined:.1f}° (re-beamforming)")
        else:
            print(f"  {rough_angle:.1f}° confirmed")

        y = beamform_at_angle(data, fs, final_angle)
        sf.write(f"speaker_{final_angle:.0f}deg.wav", y, fs)
        print(f"  Saved: speaker_{final_angle:.0f}deg.wav")

    print("\nDone.")