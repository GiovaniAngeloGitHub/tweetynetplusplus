import os
from glob import glob
import torch
from tqdm import tqdm
from tweetynetplusplus.preprocessing.spectrograms import generate_spectrogram_tensor
from tweetynetplusplus.config import settings


def preprocess_bird(bird_id: str):
    input_dir = os.path.join(settings.data.raw_root, bird_id, f"{bird_id}_songs")
    output_dir = os.path.join(settings.data.processed_root, bird_id)
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(glob(os.path.join(input_dir, "*.wav")))

    print(f"\n📦 Processando {len(audio_files)} arquivos para {bird_id}...")

    for audio_path in tqdm(audio_files, desc=f"Processando {bird_id}"):
        try:
            tensor = generate_spectrogram_tensor(audio_path)
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            save_path = os.path.join(output_dir, f"{filename}.pt")
            torch.save(tensor, save_path)
        except Exception as e:
            print(f"❌ Erro ao processar {audio_path}: {e}")

def main():
    for bird_id in settings.data.bird_ids:
        preprocess_bird(bird_id)

if __name__ == "__main__":
    main()
