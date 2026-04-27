import subprocess
import sys
from pathlib import Path

from config import AUDIO_DIR


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(filter(None, cleaned.split("_"))) or "scene"


def _powershell_escape(value: str) -> str:
    return value.replace("'", "''")


def _gtts_audio(output_path: Path, sentence: str) -> str | None:
    try:
        from gtts import gTTS
    except ImportError:
        return None

    try:
        gTTS(sentence, lang="en").save(str(output_path.with_suffix('.mp3')))
        return str(output_path.with_suffix('.mp3')) if output_path.with_suffix('.mp3').exists() else None
    except Exception:
        return None


def _system_speech_audio(output_path: Path, sentence: str) -> str | None:
    if sys.platform != "win32":
        return None

    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$path = '{_powershell_escape(str(output_path))}'; "
        "$synth.SetOutputToWaveFile($path); "
        f"$synth.Speak('{_powershell_escape(sentence)}'); "
        "$synth.Dispose();"
    )

    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
        )
        return str(output_path) if output_path.exists() else None
    except Exception:
        return None


def generate_narration_audio(scene_index: int, sentence: str) -> str | None:
    base_path = AUDIO_DIR / f"scene_{scene_index:02d}_{slugify(sentence[:40])}"

    for candidate in (base_path.with_suffix('.mp3'), base_path.with_suffix('.wav')):
        if candidate.exists():
            return str(candidate)

    gtts_result = _gtts_audio(base_path, sentence)
    if gtts_result:
        return gtts_result

    system_speech_result = _system_speech_audio(base_path.with_suffix('.wav'), sentence)
    if system_speech_result:
        return system_speech_result

    return None
