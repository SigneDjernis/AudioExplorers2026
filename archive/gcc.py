import numpy as np

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    Estimate delay between sig and refsig using GCC-PHAT.

    Parameters
    ----------
    sig : ndarray
        First signal frame
    refsig : ndarray
        Reference signal frame
    fs : int or float
        Sampling frequency
    max_tau : float or None
        Maximum absolute delay in seconds
    interp : int
        Interpolation factor for finer delay resolution

    Returns
    -------
    tau : float
        Estimated delay in seconds
    cc : ndarray
        GCC-PHAT correlation curve
    tau_axis : ndarray
        Lag axis in seconds
    """
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-10

    cc = np.fft.irfft(R, n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    abs_cc = np.abs(cc)

    shift = np.argmax(abs_cc) - max_shift
    tau = shift / float(interp * fs)
    tau_axis = np.arange(-max_shift, max_shift + 1) / float(interp * fs)

    return tau, cc, tau_axis


def frame_signal(x, frame_len, hop_len):
    """
    Split 1D signal into overlapping frames.
    """
    frames = []
    starts = []

    for start in range(0, len(x) - frame_len + 1, hop_len):
        frames.append(x[start:start + frame_len])
        starts.append(start)

    return np.array(frames), np.array(starts)


def frame_energy(frames):
    """
    Compute mean squared energy per frame.
    """
    return np.mean(frames**2, axis=1)


def gcc_phat_with_conf(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    GCC-PHAT with a simple confidence score.
    Confidence = peak(abs(cc)) / mean(abs(cc))
    """
    tau, cc, tau_axis = gcc_phat(sig, refsig, fs=fs, max_tau=max_tau, interp=interp)
    abs_cc = np.abs(cc)
    conf = np.max(abs_cc) / (np.mean(abs_cc) + 1e-10)
    return tau, conf, cc, tau_axis


def step3_doa(signal_norm, sr, frame_ms=32, hop_ms=10,
              max_tau_lr=0.0010, max_tau_fr=0.00035,
              energy_percentile=40, conf_percentile=70,
              lateral_threshold_us=10.0):
    """
    Step 3: Direction-of-arrival analysis using GCC-PHAT.

    Parameters
    ----------
    signal_norm : ndarray, shape (N, 4)
        Normalized 4-channel signal with channel order:
        [left front, left rear, right front, right rear]
    sr : int
        Sample rate
    frame_ms : float
        Frame length in milliseconds
    hop_ms : float
        Hop length in milliseconds
    max_tau_lr : float
        Max allowed ear-to-ear delay in seconds
    max_tau_fr : float
        Max allowed front-rear delay in seconds
    energy_percentile : float
        Percentile threshold for frame energy mask
    conf_percentile : float
        Percentile threshold for confidence mask
    lateral_threshold_us : float
        Threshold in microseconds for labeling lateral_pos / lateral_neg

    Returns
    -------
    results : dict
        Dictionary containing delays, masks, frame times, and direction labels
    """
    assert signal_norm.ndim == 2 and signal_norm.shape[1] == 4, \
        "Expected signal_norm with shape (N, 4)"

    # Channel order from case:
    # [left front, left rear, right front, right rear]
    LF = signal_norm[:, 0]
    LR = signal_norm[:, 1]
    RF = signal_norm[:, 2]
    RR = signal_norm[:, 3]

    frame_len = int(frame_ms / 1000 * sr)
    hop_len = int(hop_ms / 1000 * sr)

    LF_frames, starts = frame_signal(LF, frame_len, hop_len)
    LR_frames, _ = frame_signal(LR, frame_len, hop_len)
    RF_frames, _ = frame_signal(RF, frame_len, hop_len)
    RR_frames, _ = frame_signal(RR, frame_len, hop_len)

    times = starts / sr

    energy = (
        frame_energy(LF_frames) +
        frame_energy(LR_frames) +
        frame_energy(RF_frames) +
        frame_energy(RR_frames)
    ) / 4.0
    energy_db = 10 * np.log10(energy + 1e-12)

    tau_lf_rf, conf_lf_rf = [], []
    tau_lr_rr, conf_lr_rr = [], []
    tau_lf_lr, conf_lf_lr = [], []
    tau_rf_rr, conf_rf_rr = [], []

    for i in range(len(times)):
        t1, c1, _, _ = gcc_phat_with_conf(LF_frames[i], RF_frames[i], fs=sr, max_tau=max_tau_lr)
        t2, c2, _, _ = gcc_phat_with_conf(LR_frames[i], RR_frames[i], fs=sr, max_tau=max_tau_lr)
        t3, c3, _, _ = gcc_phat_with_conf(LF_frames[i], LR_frames[i], fs=sr, max_tau=max_tau_fr)
        t4, c4, _, _ = gcc_phat_with_conf(RF_frames[i], RR_frames[i], fs=sr, max_tau=max_tau_fr)

        tau_lf_rf.append(t1)
        conf_lf_rf.append(c1)

        tau_lr_rr.append(t2)
        conf_lr_rr.append(c2)

        tau_lf_lr.append(t3)
        conf_lf_lr.append(c3)

        tau_rf_rr.append(t4)
        conf_rf_rr.append(c4)

    tau_lf_rf = np.array(tau_lf_rf)
    conf_lf_rf = np.array(conf_lf_rf)

    tau_lr_rr = np.array(tau_lr_rr)
    conf_lr_rr = np.array(conf_lr_rr)

    tau_lf_lr = np.array(tau_lf_lr)
    conf_lf_lr = np.array(conf_lf_lr)

    tau_rf_rr = np.array(tau_rf_rr)
    conf_rf_rr = np.array(conf_rf_rr)

    # Combined confidence masks
    conf_lr = 0.5 * (conf_lf_rf + conf_lr_rr)
    conf_fr = 0.5 * (conf_lf_lr + conf_rf_rr)

    energy_thr = np.percentile(energy_db, energy_percentile)
    conf_lr_thr = np.percentile(conf_lr, conf_percentile)
    conf_fr_thr = np.percentile(conf_fr, conf_percentile)

    mask_lr = (energy_db > energy_thr) & (conf_lr > conf_lr_thr)
    mask_fr = (energy_db > energy_thr) & (conf_fr > conf_fr_thr)

    # Combined left-right cue
    tau_lr_combined = 0.5 * (tau_lf_rf + tau_lr_rr)

    # Simple direction labels from combined LR cue
    tau_us = tau_lr_combined * 1e6
    direction = np.full(len(times), "center", dtype=object)
    direction[mask_lr & (tau_us > lateral_threshold_us)] = "lateral_pos"
    direction[mask_lr & (tau_us < -lateral_threshold_us)] = "lateral_neg"

    return {
        "frame_len": frame_len,
        "hop_len": hop_len,
        "times": times,
        "energy_db": energy_db,

        "tau_lf_rf": tau_lf_rf,
        "tau_lr_rr": tau_lr_rr,
        "tau_lf_lr": tau_lf_lr,
        "tau_rf_rr": tau_rf_rr,

        "conf_lf_rf": conf_lf_rf,
        "conf_lr_rr": conf_lr_rr,
        "conf_lf_lr": conf_lf_lr,
        "conf_rf_rr": conf_rf_rr,

        "conf_lr": conf_lr,
        "conf_fr": conf_fr,

        "mask_lr": mask_lr,
        "mask_fr": mask_fr,

        "tau_lr_combined": tau_lr_combined,
        "direction": direction,
    }