import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import whisper

fs, data = wavfile.read("Recordings/example_mixture.wav")
data = data.astype(np.float32) / 32768.0


f, t, Zxx = signal.stft(data.T, fs, nperseg=1024)
fb_map = np.abs(Zxx[0] + Zxx[1]) / (np.abs(Zxx[3] + Zxx[2]) + 1e-10)

rear_mask = fb_map < 0.7
Zxx_rear_only = Zxx * rear_mask

_, clean_rear = signal.istft(Zxx_rear_only, fs)

# Make sure Whisper gets mono float32 audio
clean_rear = np.asarray(clean_rear, dtype=np.float32)

# If stereo/multichannel, collapse to mono
if clean_rear.ndim > 1:
    clean_rear = clean_rear.mean(axis=0)

model = whisper.load_model("base")
result = model.transcribe(clean_rear, fp16=False)
print(result["text"])