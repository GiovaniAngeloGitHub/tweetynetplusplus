import pytest
import os
from glob import glob
from tweetynetplusplus.preprocessing.spectrograms import generate_spectrogram_tensor
from tweetynetplusplus.config import settings

@pytest.mark.parametrize("bird_id", settings.data.bird_ids)
def test_generate_spectrograms_for_all_audio(bird_id):
    audio_dir = f"data/raw/{bird_id}/{bird_id}_songs"
    assert os.path.exists(audio_dir), f"Diretório {audio_dir} não encontrado"

    audio_files = sorted(glob(os.path.join(audio_dir, "*.wav")))
    assert len(audio_files) > 0, f"Nenhum arquivo .wav encontrado em {audio_dir}"

    # Limitar a 3 arquivos para teste rápido
    for path in audio_files[:3]:
        spec = generate_spectrogram_tensor(path)
        assert spec.shape[0] == 3, f"O espectrograma deve ter 3 canais (mel, lin, cqt): {path}"
        assert spec.shape[1] > 0 and spec.shape[2] > 0, f"Espectrograma vazio para {path}"
