
from pathlib import Path

import librosa
import pandas as pd


def validate_esc50(data_dir: str):
    """
    Validate the ESC-50 audio dataset and its metadata.
    """

    data_path = Path(data_dir)
    audio_dir = data_path / "audio"
    metadata_path = data_path / "meta" / "esc50.csv"

    # 1. Check required paths
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    # 2. Load metadata
    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "filename", "fold", "target", "category"
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # 3. Check missing values
    if metadata[list(required_columns)].isnull().any().any():
        raise ValueError("Missing values found in required columns")

    # 4. Check audio file count
    audio_files = list(audio_dir.glob("*.wav"))

    if len(audio_files) != len(metadata):
        raise ValueError(
            f"Audio count mismatch: {len(audio_files)} audio files "
            f"and {len(metadata)} metadata records"
        )

    # 5. Check missing audio files
    missing_files = [
        filename for filename in metadata["filename"]
        if not (audio_dir / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing audio files: {missing_files[:5]}"
        )

    # 6. Check labels and folds
    if metadata["category"].nunique() != 50:
        raise ValueError("Expected exactly 50 sound categories")

    if not metadata["fold"].between(1, 5).all():
        raise ValueError("Invalid fold values")

    # 7. Check audio properties on every file
    invalid_audio = []

    for filename in metadata["filename"]:
        file_path = audio_dir / filename

        try:
            audio, sample_rate = librosa.load(
                file_path, sr=None, mono=True
            )

            duration = len(audio) / sample_rate

            if sample_rate != 44100 or abs(duration - 5.0) > 0.1:
                invalid_audio.append(filename)

        except Exception:
            invalid_audio.append(filename)

    if invalid_audio:
        raise ValueError(
            f"Invalid audio files: {invalid_audio[:5]}"
        )

    # 8. Report results
    print("ESC-50 VALIDATION SUCCESSFUL")
    print(f"Total audio files: {len(audio_files)}")
    print(f"Total classes: {metadata['category'].nunique()}")
    print(f"Total folds: {metadata['fold'].nunique()}")
    print("Sample rate: 44100 Hz")
    print("Duration: Approximately 5 seconds")


if __name__ == "__main__":
    validate_esc50("data/raw/ESC-50")
