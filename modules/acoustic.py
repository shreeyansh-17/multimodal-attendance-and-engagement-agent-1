"""
Module 2: Acoustic Interaction & Noise Dynamics (Tab 2 - Epic 3)

Audio stream processing with spectral analysis and Voice Activity Detection.
Classifies audio windows into: Silence, Instructor Lecture,
Student Discussion, and High Ambient Distraction.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Optional, Tuple
from scipy.signal import butter, filtfilt

from utils.core import AudioWindow, AudioLabel


class VoiceActivityDetector:
    """Energy + Zero-Crossing Rate based Voice Activity Detection."""

    def __init__(
        self,
        energy_threshold: float = 0.01,
        zcr_threshold: float = 0.1,
    ):
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold

    def detect(self, audio_chunk: np.ndarray, sr: int) -> bool:
        """Return True if voice activity is detected."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio_chunk)))) / 2.0

        return rms > self.energy_threshold and zcr < self.zcr_threshold


class AudioClassifier:
    """Classify audio windows into pedagogical categories."""

    # Thresholds derived from classroom acoustic studies
    SILENCE_RMS = 0.005
    HIGH_ENERGY_RMS = 0.15
    LECTURE_CENTROID_RANGE = (800, 3500)    # Hz - single voice
    DISCUSSION_CENTROID_RANGE = (2000, 6000)  # Hz - multiple voices
    DISTRACTION_ZCR = 0.25

    def classify(self, window: AudioWindow) -> str:
        """Classify an audio window into one of four categories."""
        if window.rms_energy < self.SILENCE_RMS:
            return AudioLabel.SILENCE.value

        if window.rms_energy > self.HIGH_ENERGY_RMS:
            if window.zero_crossing_rate > self.DISTRACTION_ZCR:
                return AudioLabel.DISTRACTION.value
            else:
                return AudioLabel.DISCUSSION.value

        if (
            self.LECTURE_CENTROID_RANGE[0]
            <= window.spectral_centroid
            <= self.LECTURE_CENTROID_RANGE[1]
        ):
            if window.voice_activity:
                return AudioLabel.INSTRUCTOR.value

        if window.spectral_centroid > self.DISCUSSION_CENTROID_RANGE[0]:
            if window.voice_activity:
                return AudioLabel.DISCUSSION.value

        if window.voice_activity:
            return AudioLabel.INSTRUCTOR.value

        return AudioLabel.SILENCE.value

    def compute_interaction_score(self, label: str, rms: float) -> float:
        """Compute a 0-100 interaction score based on classification."""
        base_scores = {
            AudioLabel.SILENCE.value: 10.0,
            AudioLabel.INSTRUCTOR.value: 50.0,
            AudioLabel.DISCUSSION.value: 85.0,
            AudioLabel.DISTRACTION.value: 25.0,
        }
        base = base_scores.get(label, 10.0)
        # Modulate by energy level
        energy_factor = min(rms / 0.1, 1.5)
        return min(100.0, base * energy_factor)


class AcousticProcessor:
    """Processes audio files into classified time windows."""

    def __init__(
        self,
        window_duration: float = 5.0,
        hop_duration: float = 2.5,
        target_sr: int = 22050,
    ):
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.target_sr = target_sr
        self.vad = VoiceActivityDetector()
        self.classifier = AudioClassifier()
        self.windows: List[AudioWindow] = []

    def process_audio_file(
        self, audio_path: str, progress_callback=None
    ) -> List[AudioWindow]:
        """
        Process an audio file and return classified windows.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            progress_callback: Optional callable(float) for progress updates.

        Returns:
            List of AudioWindow objects with classifications.
        """
        self.windows.clear()

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        duration = len(y) / sr

        window_samples = int(self.window_duration * sr)
        hop_samples = int(self.hop_duration * sr)
        total_windows = max(1, int((len(y) - window_samples) / hop_samples) + 1)

        for i in range(total_windows):
            start_sample = i * hop_samples
            end_sample = min(start_sample + window_samples, len(y))
            chunk = y[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # Skip very short chunks
                continue

            start_time = start_sample / sr
            end_time = end_sample / sr

            # Compute features
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            spectral_centroids = librosa.feature.spectral_centroid(
                y=chunk, sr=sr, n_fft=min(2048, len(chunk))
            )
            centroid = float(np.mean(spectral_centroids)) if spectral_centroids.size > 0 else 0.0

            zcr_values = librosa.feature.zero_crossing_rate(chunk)
            zcr = float(np.mean(zcr_values)) if zcr_values.size > 0 else 0.0

            voice_active = self.vad.detect(chunk, sr)

            window = AudioWindow(
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                rms_energy=round(rms, 6),
                spectral_centroid=round(centroid, 2),
                zero_crossing_rate=round(zcr, 6),
                voice_activity=voice_active,
            )

            # Classify
            window.label = self.classifier.classify(window)
            window.interaction_score = round(
                self.classifier.compute_interaction_score(window.label, rms), 2
            )

            self.windows.append(window)

            if progress_callback:
                progress_callback(min((i + 1) / total_windows, 1.0))

        return self.windows

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio track from a video file using librosa.
        Returns path to temporary WAV file.
        """
        import tempfile
        import os

        try:
            y, sr = librosa.load(video_path, sr=self.target_sr, mono=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, y, sr)
            return tmp.name
        except Exception:
            return None

    def get_summary(self) -> dict:
        """Return summary of audio analysis."""
        if not self.windows:
            return {
                "total_windows": 0,
                "duration": "00:00:00",
                "silence_pct": 0,
                "lecture_pct": 0,
                "discussion_pct": 0,
                "distraction_pct": 0,
                "avg_interaction": 0,
            }

        total = len(self.windows)
        labels = [w.label for w in self.windows]
        duration = self.windows[-1].end_time if self.windows else 0

        def pct(label):
            return round(labels.count(label) / total * 100, 1)

        return {
            "total_windows": total,
            "duration": f"{int(duration // 60):02d}:{int(duration % 60):02d}",
            "silence_pct": pct(AudioLabel.SILENCE.value),
            "lecture_pct": pct(AudioLabel.INSTRUCTOR.value),
            "discussion_pct": pct(AudioLabel.DISCUSSION.value),
            "distraction_pct": pct(AudioLabel.DISTRACTION.value),
            "avg_interaction": round(
                np.mean([w.interaction_score for w in self.windows]), 2
            ),
        }

    def get_windows_dataframe(self):
        """Return windows as a Pandas DataFrame."""
        import pandas as pd

        rows = []
        for w in self.windows:
            rows.append({
                "Start": f"{w.start_time:.1f}s",
                "End": f"{w.end_time:.1f}s",
                "RMS Energy": f"{w.rms_energy:.4f}",
                "Spectral Centroid": f"{w.spectral_centroid:.0f} Hz",
                "ZCR": f"{w.zero_crossing_rate:.4f}",
                "Voice Activity": "✓" if w.voice_activity else "✗",
                "Classification": w.label,
                "Interaction Score": f"{w.interaction_score:.1f}",
            })
        return pd.DataFrame(rows)

    def reset(self):
        """Reset processor state."""
        self.windows.clear()
