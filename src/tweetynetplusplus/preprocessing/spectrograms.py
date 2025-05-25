import torch
import torchaudio
import librosa
import numpy as np
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB, Resample
from tweetynetplusplus.utils.func import resize_tensor
from tweetynetplusplus.config import settings


def generate_spectrogram_tensor(
    wav_path: str,
    sample_rate: int = settings.spectrogram.sample_rate,
    n_fft: int = settings.spectrogram.n_fft,
    hop_length: int = settings.spectrogram.hop_length,
    target_bins: int = settings.spectrogram.target_height,
) -> torch.Tensor:
    """
    Gera um tensor com 3 canais (mel, linear, cqt) a partir de um arquivo de áudio.

    Args:
        wav_path (str): Caminho do arquivo WAV
        sample_rate (int): Taxa de amostragem desejada
        n_fft (int): Tamanho da janela FFT
        hop_length (int): Salto entre janelas FFT
        target_bins (int): Altura (frequência) para interpolar os espectrogramas

    Returns:
        torch.Tensor: Tensor (3 x target_bins x T) com espectrogramas normalizados
    """
    # Carrega o áudio
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        waveform = Resample(sr, sample_rate)(waveform)

    # Converte para mono se for estéreo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # --- Mel Spectrogram ---
    mel_spec = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=target_bins
    )(waveform)
    mel_spec = AmplitudeToDB()(mel_spec)

    # --- Linear Spectrogram ---
    lin_spec = Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(waveform)
    lin_spec = AmplitudeToDB()(lin_spec)

    # --- CQT com Librosa ---
    waveform_np = waveform.squeeze().numpy()
    cqt = librosa.cqt(waveform_np, sr=sample_rate, hop_length=hop_length)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_tensor = torch.tensor(cqt_db, dtype=torch.float32).unsqueeze(0)

    # Recorte temporal comum
    min_time = min(mel_spec.shape[-1], lin_spec.shape[-1], cqt_tensor.shape[-1])
    mel_spec = mel_spec[..., :min_time]
    lin_spec = lin_spec[..., :min_time]
    cqt_tensor = cqt_tensor[..., :min_time]

    # Interpolação para altura padronizada
    mel_spec = resize_tensor(mel_spec, target_bins)
    lin_spec = resize_tensor(lin_spec, target_bins)
    cqt_tensor = resize_tensor(cqt_tensor, target_bins)

    # Empilha como 3 canais

    stacked = torch.cat([mel_spec, lin_spec, cqt_tensor], dim=0)

    return stacked
