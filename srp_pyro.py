import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# ── Load audio ────────────────────────────────────────────────────────────────
# Change to "mixture.wav" when ready to run on the real scene
data, fs = sf.read("Recordings/mixture.wav")
# data shape: (samples, 4) → [left_front, left_rear, right_front, right_rear]

# ── Microphone array geometry (metres) ───────────────────────────────────────
# pyroomacoustics expects shape (n_dimensions, n_mics)
# We work in 2D: x = left/right, y = front/back
d_lr = 0.15 / 2   # half inter-aural distance
d_fb = 0.01 / 2   # half front-rear spacing per ear

#         left_front   left_rear    right_front  right_rear
mic_array = np.array([
    [-d_lr,  -d_lr,    d_lr,    d_lr],   # x (left/right)
    [ d_fb,  -d_fb,   d_fb,   -d_fb],   # y (front/back)
])  # shape: (2, 4)

# ── Candidate azimuth angles ──────────────────────────────────────────────────
# pyroomacoustics uses radians, azimuth from positive x-axis
# We scan full 360° at 1° resolution
azimuths = np.linspace(0, 2 * np.pi, 360, endpoint=False)

# ── Run SRP-PHAT ──────────────────────────────────────────────────────────────
nfft = 512

# Compute STFT for each channel → pyroomacoustics expects (n_mics, nfft//2+1, n_frames)
X = np.array([
    pra.transform.stft.analysis(data[:, m], nfft, nfft // 2).T
    for m in range(data.shape[1])
])  # (n_mics, nfft//2+1, n_frames)

doa = pra.doa.SRP(mic_array, fs, nfft=nfft, azimuth=azimuths, num_src=4)
doa.locate_sources(X, freq_range=[300, 3400])

# Spatial spectrum (power at each candidate angle)
spectrum = doa.grid.values
angles_deg = np.degrees(azimuths)

# Normalise to [0, 1]
spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-10)

# ── Adaptive peak finding ─────────────────────────────────────────────────────
def find_adaptive_peaks(power, angles, min_distance=20, threshold_ratio=0.4):
    found = []
    power_copy = power.copy()
    threshold  = threshold_ratio * power_copy.max()
    while True:
        idx = np.argmax(power_copy)
        if power_copy[idx] < threshold:
            break
        found.append((angles[idx], power_copy[idx]))
        for i, a in enumerate(angles):
            diff = abs(a - angles[idx])
            if min(diff, 360 - diff) < min_distance:
                power_copy[i] = -np.inf
    return sorted(found, key=lambda x: x[0])

peaks = find_adaptive_peaks(spectrum, angles_deg, min_distance=20, threshold_ratio=0.4)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(angles_deg, spectrum, color='steelblue')
ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("SRP-PHAT Power (normalised)")
ax.set_title("pyroomacoustics SRP-PHAT – DoA Estimation")
ax.set_xticks(range(0, 361, 45))
for ang, pwr in peaks:
    ax.axvline(ang, color='green', linewidth=1.5, alpha=0.9)
    ax.text(ang + 2, 0.95, f"{ang:.0f}°", color='green', fontsize=9,
            fontweight='bold', transform=ax.get_xaxis_transform())
plt.tight_layout()
plt.savefig("srp_pyroomacoustics.png", dpi=150)
plt.show()

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\nDetected {len(peaks)} talker(s):")
for ang, pwr in peaks:
    print(f"  {ang:>6.1f}°  (power: {pwr:.3f})")