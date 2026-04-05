import soundfile as sf

data, samplerate = sf.read("Recordings/mixture.wav")

# Same structure: (samples, channels)
left_front  = data[:, 0]
left_rear   = data[:, 1]
right_front = data[:, 2]
right_rear  = data[:, 3]

sf.write("Recordings/left_front.wav", left_front, samplerate)
sf.write("Recordings/left_rear.wav", left_rear, samplerate)
sf.write("Recordings/right_front.wav", right_front, samplerate)
sf.write("Recordings/right_rear.wav", right_rear, samplerate)