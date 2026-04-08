from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

#model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')
#print(model)
#save weights for the encoder, decoder and masknet
#model.save_to_state_dict("sepformer-whamr-enhancement.pth")

# for custom file, change path
channel_no = 1
#est_sources = model.separate_file(path=f'D:/marie/OneDrive/Documents/Uni/8semester/OticonChallenge/OticonCase2024/data/channels/channel_{channel_no}.wav') 
#est_sources = model.separate_file(path='D:/marie/OneDrive/Documents/Uni/8semester/OticonChallenge/MS-SNSD/NoisySpeech_training/noisy1_SNRdb_5.685029228780394_11.51584493657106_13.935722729141197_19.45053852066357_clnsp1.wav')
#est_sources = model.separate_file(path='speechbrain/sepformer-wsj02mix/test_mixture.wav') 
est_sources = model.separate_file(path=f'noisy_mixture_ch{1}.wav')
torchaudio.save(f"ehanced_sepformer_ch{1}.wav", est_sources[:, :, 0].detach().cpu(), 8000)

#torchaudio.save(f"enhanced_whamr_channel_{channel_no}.wav", est_sources[:, :, 0].detach().cpu(), 8000)
#print(f"Enhanced audio saved as 'enhanced_whamr_channel_{channel_no}.wav'")