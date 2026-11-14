
from pathlib import Path

import json
import librosa
import pandas as pd


def validate_esc50(data_dir: str, report_path: str):
    data_path = Path(data_dir)
    audio_dir = data_path / "audio"
    metadata_path = data_path / "meta" / "esc50.csv"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = {"filename", "fold", "target", "category"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if metadata[list(required_columns)].isnull().any().any():
        raise ValueError("Missing values found in required columns")

    audio_files = list(audio_dir.glob("*.wav"))

    if len(audio_files) != len(metadata):
        raise ValueError("Audio count does not match metadata records")

    missing_files = [
        name for name in metadata["filename"]
        if not (audio_dir / name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files[:5]}")

    if metadata["category"].nunique() != 50:
        raise ValueError("Expected exactly 50 categories")

    if not metadata["fold"].between(1, 5).all():
        raise ValueError("Invalid fold values")

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
        raise ValueError(f"Invalid audio files: {invalid_audio[:5]}")

    report = {
        "status": "success",
        "total_audio_files": len(audio_files),
        "total_classes": int(metadata["category"].nunique()),
        "total_folds": int(metadata["fold"].nunique()),
        "sample_rate": 44100,
        "duration_seconds": 5.0
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=4))

    print("ESC-50 VALIDATION SUCCESSFUL")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    validate_esc50(
        "data/raw/ESC-50",
        "reports/validation_report.json"
    )
