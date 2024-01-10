import torchaudio
import torchaudio.transforms as T
import torch
import utils
import os

from repcodec.whisper_feature_reader import WhisperFeatureReader
from repcodec.RepCodec import RepCodec

device = "cuda" if torch.cuda.is_available() else "cpu"
repcodec_chkpt_path = "/home/pranj/dev/repcodec_200k.pkl"

whisper_root, whisper_name, layer = "/home/pranj/dev/temp/whisper_model",  "medium", 24
reader = WhisperFeatureReader(whisper_root, whisper_name, layer, device=device)

repcodec_model = utils.get_repcodec_model(repcodec_chkpt_path, device).to(device)
print("Repcodec model loaded")

def process_one(in_dir, filename):
    wav, sr = torchaudio.load(filename)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)
    whisper_feat_path = filename.replace(in_dir, in_dir+"_whisperfeat").replace('.mp3','.pt').replace('.flac','.pt')
    filename = filename.replace(in_dir, in_dir+"_processed").replace('.mp3','.wav').replace('.flac','.wav')
    wav24k_path = filename
    wav16k = T.Resample(sr, 16000)(wav)
    wav24k = T.Resample(sr, 24000)(wav)
    if not os.path.exists(os.path.dirname(wav24k_path)):
        os.makedirs(os.path.dirname(wav24k_path))
    if not os.path.exists(os.path.dirname(whisper_feat_path)):
        os.makedirs(os.path.dirname(whisper_feat_path))
    torchaudio.save(wav24k_path, wav24k, 24000)
    cvec_path = filename + ".cvec.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav16k = wav16k.to(device)

    if not os.path.exists(cvec_path):
        whisper_feats = reader.get_feats_tensor(wav16k[0], rate=16000).unsqueeze(0).transpose(1, 2)
        torch.save(whisper_feats.cpu(), whisper_feat_path)
        with torch.no_grad():
            x = repcodec_model.encoder(whisper_feats)
            z = repcodec_model.projector(x)
        torch.save(z.cpu(), cvec_path)

    spec_path = filename.replace(".wav", ".mel.pt")
    spec_process = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        center=True,
        power=1,
    )
    spec = spec_process(wav24k)# 1 100 T
    spec = torch.log(torch.clip(spec, min=1e-7))
    torch.save(spec, spec_path)

