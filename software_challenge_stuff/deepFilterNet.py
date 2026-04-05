from df.enhance import enhance, init_df, load_audio, save_audio
import torch
import librosa

audio_path = "OticonCase2024\data\channels\channel_1.wav"

model, df_state, _ = init_df()  # Load default model

noisy_audio, _ = librosa.load(audio_path, sr=48000)
noisy_audio = torch.tensor(noisy_audio).unsqueeze(0) 
#noisy_audio, _ = load_audio(audio_path, sr=df_state.sr())

enhanced_audio = enhance(model, df_state, noisy_audio)
save_audio("enhanced.wav", enhanced_audio, df_state.sr())
