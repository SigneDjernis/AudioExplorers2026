import numpy as np
import scipy.signal as signal

def isolate_angle(audio, fs, target_angle_deg, tolerance_deg=25, nperseg=1024):
    """
    Apply a soft directional mask but preserve all channels.

    audio: shape (samples, channels)
    returns:
        filtered_audio: shape (samples, channels)
        target_mask: shape (freqs, frames)
        Zxx_filtered: shape (channels, freqs, frames)
    """
    eps = 1e-8
    audio = audio.astype(np.float32)

    # STFT: (channels, freqs, frames)
    _, _, Zxx = signal.stft(audio.T, fs=fs, nperseg=nperseg)
    mag = np.abs(Zxx)

    # Directional energies
    front_energy = mag[0] + mag[1]
    back_energy  = mag[2] + mag[3]
    left_energy  = mag[0] + mag[2]
    right_energy = mag[1] + mag[3]

    # Direction scores
    fb_score = np.log((front_energy + eps) / (back_energy + eps))
    lr_score = np.log((left_energy + eps) / (right_energy + eps))

    angle_est = np.arctan2(lr_score, fb_score)
    angle_est = angle_est - np.deg2rad(90)
    angle_est = np.mod(angle_est, 2 * np.pi)

    target_angle = np.deg2rad(target_angle_deg)
    tolerance = np.deg2rad(tolerance_deg)

    angle_diff = np.angle(np.exp(1j * (angle_est - target_angle)))

    # Soft mask
    target_mask = np.exp(-0.5 * (angle_diff / tolerance) ** 2).astype(np.float32)

    # Apply same mask to all channels, preserving multichannel structure
    Zxx_filtered = Zxx * target_mask[None, :, :]

    # ISTFT back to multichannel audio
    _, filtered_audio = signal.istft(Zxx_filtered, fs=fs, nperseg=nperseg)
    filtered_audio = np.real(filtered_audio).T   # back to (samples, channels)

    mx = np.max(np.abs(filtered_audio)) + eps
    filtered_audio = filtered_audio / mx

    return filtered_audio, target_mask, Zxx_filtered



def isolate_angle_and_denoise(audio, fs, target_angle_deg, tolerance_deg=25, nperseg=1024):
    """
    Soft directional masking + covariance estimation + MVDR-like beamforming.

    audio: shape (samples, channels)
    returns mono enhanced signal
    """

    eps = 1e-8
    audio = audio.astype(np.float32)

    # STFT: Zxx shape (channels, freqs, frames)
    _, _, Zxx = signal.stft(audio.T, fs=fs, nperseg=nperseg)
    C, F, T = Zxx.shape
    mag = np.abs(Zxx)

    # Same directional energies as your original code
    front_energy = mag[0] + mag[1]
    back_energy  = mag[2] + mag[3]
    left_energy  = mag[0] + mag[2]
    right_energy = mag[1] + mag[3]

    # Direction scores
    fb_score = np.log((front_energy + eps) / (back_energy + eps))
    lr_score = np.log((left_energy + eps) / (right_energy + eps))

    angle_est = np.arctan2(lr_score, fb_score)
    angle_est = angle_est - np.deg2rad(90)
    angle_est = np.mod(angle_est, 2 * np.pi)

    target_angle = np.deg2rad(target_angle_deg)
    tolerance = np.deg2rad(tolerance_deg)

    angle_diff = np.angle(np.exp(1j * (angle_est - target_angle)))

    # Soft target mask instead of hard mask
    target_mask = np.exp(-0.5 * (angle_diff / tolerance) ** 2).astype(np.float32)
    noise_mask = 1.0 - target_mask

    enhanced = np.zeros((F, T), dtype=np.complex64)

    ref_mic = 0

    for f in range(F):
        Xf = Zxx[:, f, :]   # shape (C, T)

        # Spatial covariance matrices
        Rs = np.zeros((C, C), dtype=np.complex64)
        Rn = np.zeros((C, C), dtype=np.complex64)

        ms_sum = np.sum(target_mask[f]) + eps
        mn_sum = np.sum(noise_mask[f]) + eps

        for t in range(T):
            x = Xf[:, t:t+1]   # shape (C,1)
            Rs += target_mask[f, t] * (x @ x.conj().T)
            Rn += noise_mask[f, t] * (x @ x.conj().T)

        Rs /= ms_sum
        Rn /= mn_sum

        # Regularization
        Rn += 1e-3 * np.eye(C, dtype=np.complex64)
        Rs += 1e-6 * np.eye(C, dtype=np.complex64)

        # Principal eigenvector of speech covariance as steering vector estimate
        vals, vecs = eigh(Rs)
        d = vecs[:, -1]
        d = d / (d[ref_mic] + eps)

        # MVDR weights
        Rn_inv_d = np.linalg.solve(Rn, d)
        w = Rn_inv_d / (d.conj().T @ Rn_inv_d + eps)

        # Beamform
        enhanced[f, :] = np.conj(w).T @ Xf

    _, y = signal.istft(enhanced, fs=fs, nperseg=nperseg)
    y = np.real(y)
    y /= np.max(np.abs(y) + eps)

    return y