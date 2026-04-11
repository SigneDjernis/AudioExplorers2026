import numpy as np
import scipy.signal as signal
from scipy.linalg import eigh
from scipy.signal import resample_poly
import soundfile as sf
from scipy.io import wavfile

def estimate_directional_masks(Zxx, target_angle_deg, tolerance_deg=25, eps=1e-8):
    """
    Helper function to estimate soft target/noise masks from multichannel STFT using simple
    directional energy heuristics.

    Parameters
    ----------
    Zxx : np.ndarray
        Complex STFT with shape (channels, freqs, frames).
    target_angle_deg : float
        Desired look direction in degrees.
    tolerance_deg : float, optional
        Angular spread of the soft target mask in degrees.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    target_mask : np.ndarray
        Soft target mask of shape (freqs, frames).
    noise_mask : np.ndarray
        Complementary soft noise mask of shape (freqs, frames).
    angle_est : np.ndarray
        Estimated angle per time-frequency bin, shape (freqs, frames).
    """
    mag = np.abs(Zxx)

    # Assumes 4 channels arranged as in your original heuristic
    front_energy = mag[0] + mag[2]
    back_energy = mag[1] + mag[3]
    left_energy = mag[0] + mag[1]
    right_energy = mag[2] + mag[3]

    fb_score = np.log((front_energy + eps) / (back_energy + eps))
    lr_score = np.log((left_energy + eps) / (right_energy + eps))

    angle_est = np.arctan2(lr_score, fb_score)
    angle_est = np.mod(angle_est, 2 * np.pi)

    target_angle = np.deg2rad(target_angle_deg)
    tolerance = np.deg2rad(tolerance_deg)

    # Wrapped angular difference in [-pi, pi]
    angle_diff = np.angle(np.exp(1j * (angle_est - target_angle)))

    target_mask = np.exp(-0.5 * (angle_diff / tolerance) ** 2).astype(np.float32)
    noise_mask = 1.0 - target_mask

    return target_mask, noise_mask, angle_est


def mvdr_beamform(Zxx, target_mask, noise_mask, ref_mic=0, eps=1e-8):
    """
    Helper function to apply MVDR beamforming using target/noise masks to estimate spatial
    covariance matrices.

    Parameters
    ----------
    Zxx : np.ndarray
        Complex STFT with shape (channels, freqs, frames).
    target_mask : np.ndarray
        Target mask with shape (freqs, frames).
    noise_mask : np.ndarray
        Noise mask with shape (freqs, frames).
    ref_mic : int, optional
        Reference microphone index used to normalize the steering vector.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    enhanced : np.ndarray
        Beamformed complex STFT with shape (freqs, frames).
    """
    C, F, T = Zxx.shape
    enhanced = np.zeros((F, T), dtype=np.complex64)

    for f in range(F):
        Xf = Zxx[:, f, :]  # shape (C, T)

        Rs = np.zeros((C, C), dtype=np.complex64)
        Rn = np.zeros((C, C), dtype=np.complex64)

        ms_sum = np.sum(target_mask[f]) + eps
        mn_sum = np.sum(noise_mask[f]) + eps

        for t in range(T):
            x = Xf[:, t:t + 1]  # shape (C, 1)
            Rs += target_mask[f, t] * (x @ x.conj().T)
            Rn += noise_mask[f, t] * (x @ x.conj().T)

        Rs /= ms_sum
        Rn /= mn_sum

        # Regularization
        Rs += 1e-6 * np.eye(C, dtype=np.complex64)
        Rn += 1e-3 * np.eye(C, dtype=np.complex64)

        _, vecs = eigh(Rs)
        d = vecs[:, -1]
        d = d / (d[ref_mic] + eps)

        Rn_inv_d = np.linalg.solve(Rn, d)
        w = Rn_inv_d / (d.conj().T @ Rn_inv_d + eps)

        enhanced[f, :] = np.conj(w).T @ Xf

    return enhanced


def enhance_source(audio, fs, target_angle_deg, tolerance_deg=25, nperseg=1024):
    """
    Full pipeline:
    1. STFT
    2. Directional soft mask estimation
    3. MVDR beamforming
    4. iSTFT reconstruction

    Parameters
    ----------
    audio : np.ndarray
        Input signal with shape (samples, channels).
    fs : int
        Sample rate.
    target_angle_deg : float
        Desired look direction in degrees.
    tolerance_deg : float, optional
        Angular spread of the soft target mask in degrees.
    nperseg : int, optional
        STFT window length.

    Returns
    -------
    y : np.ndarray
        Mono enhanced time-domain signal.
    """
    eps = 1e-8
    audio = audio.astype(np.float32)

    _, _, Zxx = signal.stft(audio.T, fs=fs, nperseg=nperseg)

    target_mask, noise_mask, _ = estimate_directional_masks(
        Zxx,
        target_angle_deg=target_angle_deg,
        tolerance_deg=tolerance_deg,
        eps=eps,
    )

    enhanced = mvdr_beamform(
        Zxx,
        target_mask=target_mask,
        noise_mask=noise_mask,
        ref_mic=0,
        eps=eps,
    )

    _, y = signal.istft(enhanced, fs=fs)
    y = np.real(y)

    peak = np.max(np.abs(y))
    if peak > eps:
        y /= peak

    return y

if __name__ == "__main__":
    angles = [7,87,137,184,230,271] # Angles found from the SRP analysis
    fs, data = wavfile.read('recordings/mixture.wav')
    data = data.astype(np.float32) / 32768.0

    for angle in angles:
        print(f"Isolating angle {angle}°...")
        enhanced = enhance_source(
            data,
            fs,
            target_angle_deg=angle,
            tolerance_deg=25,
            nperseg=1024
        )

        enhanced_16k = resample_poly(enhanced, up=16000, down=fs).astype(np.float32)
        result = model.transcribe(enhanced_16k, fp16=False)
        print(result["text"])

        sf.write(f"results/speakers/speaker_{angle:.0f}deg.wav", enhanced, fs)