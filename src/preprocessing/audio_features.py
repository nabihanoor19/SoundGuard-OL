
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def preprocess_esc50(config):
    data_path = Path(config["data"]["data_dir"])
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    sample_rate = config["preprocessing"]["sample_rate"]
    n_mels = config["preprocessing"]["n_mels"]
    n_fft = config["preprocessing"]["n_fft"]
    hop_length = config["preprocessing"]["hop_length"]

    metadata = pd.read_csv(data_path / "meta" / "esc50.csv")

    features = []
    labels = []
    folds = []
    filenames = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        audio_path = data_path / "audio" / row["filename"]

        audio, _ = librosa.load(
            audio_path,
            sr=sample_rate,
            mono=True
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        features.append(log_mel.astype(np.float32))
        labels.append(row["target"])
        folds.append(row["fold"])
        filenames.append(row["filename"])

    features = np.stack(features)
    labels = np.array(labels)
    folds = np.array(folds)

    np.save(output_path / "features.npy", features)
    np.save(output_path / "labels.npy", labels)
    np.save(output_path / "folds.npy", folds)

    pd.DataFrame({
        "filename": filenames,
        "target": labels,
        "fold": folds
    }).to_csv(output_path / "metadata.csv", index=False)

    print("\nPREPROCESSING COMPLETED")
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Folds shape:", folds.shape)


if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    preprocess_esc50(config)
