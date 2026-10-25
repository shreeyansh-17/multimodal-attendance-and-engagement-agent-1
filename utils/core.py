"""
Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5

Core utility functions and shared constants.
"""

import json
import datetime
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd


# ───────────────────────── Constants ─────────────────────────

SESSION_PRESENCE_THRESHOLD = 0.70  # 70% session time for "Present"
ENGAGEMENT_WINDOW_SEC = 10         # seconds per engagement window
LOW_ENGAGEMENT_THRESHOLD = 45      # below this triggers suggestion
LOW_ENGAGEMENT_DURATION = 300      # 5 minutes sustained low engagement

# Audio classification labels
class AudioLabel(Enum):
    SILENCE = "Silence"
    INSTRUCTOR = "Instructor Lecture"
    DISCUSSION = "Student Discussion"
    DISTRACTION = "High Ambient Distraction"


# Engagement weight factors
VISUAL_WEIGHT = 0.55
AUDIO_WEIGHT = 0.45


# ───────────────────── Data Classes ──────────────────────

@dataclass
class ParticipantRecord:
    """Tracks a single participant's session data."""
    student_id: str
    first_seen: float = 0.0          # timestamp in seconds
    last_seen: float = 0.0
    total_frames_detected: int = 0
    total_frames_possible: int = 0
    face_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    head_pose_scores: List[float] = field(default_factory=list)
    posture_scores: List[float] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def presence_ratio(self) -> float:
        if self.total_frames_possible == 0:
            return 0.0
        return self.total_frames_detected / self.total_frames_possible

    @property
    def status(self) -> str:
        return "Present" if self.presence_ratio >= SESSION_PRESENCE_THRESHOLD else "Absent"

    @property
    def avg_visual_engagement(self) -> float:
        scores = self.head_pose_scores + self.posture_scores
        return np.mean(scores) if scores else 0.0


@dataclass
class AudioWindow:
    """Single time-window of audio analysis."""
    start_time: float
    end_time: float
    rms_energy: float = 0.0
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0
    voice_activity: bool = False
    label: str = AudioLabel.SILENCE.value
    interaction_score: float = 0.0


@dataclass
class EngagementTimeStep:
    """Fused multimodal engagement at a single time step."""
    timestamp: float
    visual_score: float = 0.0
    audio_score: float = 0.0
    fused_score: float = 0.0


@dataclass
class SessionData:
    """Complete session data container."""
    session_id: str = ""
    start_time: str = ""
    end_time: str = ""
    total_duration_sec: float = 0.0
    participants: Dict[str, ParticipantRecord] = field(default_factory=dict)
    audio_windows: List[AudioWindow] = field(default_factory=list)
    engagement_timeline: List[EngagementTimeStep] = field(default_factory=list)
    suggestions: List[Dict] = field(default_factory=list)

    def generate_session_id(self):
        ts = datetime.datetime.now().isoformat()
        self.session_id = hashlib.md5(ts.encode()).hexdigest()[:12]
        self.start_time = ts

    def finalize(self):
        self.end_time = datetime.datetime.now().isoformat()


# ───────────────────── Helper Functions ──────────────────

def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a value to 0-100 range."""
    if max_val == min_val:
        return 50.0
    clamped = max(min_val, min(max_val, value))
    return ((clamped - min_val) / (max_val - min_val)) * 100.0


def compute_engagement_index(visual: float, audio: float) -> float:
    """Compute fused engagement index (0-100)."""
    return round(VISUAL_WEIGHT * visual + AUDIO_WEIGHT * audio, 2)


def attendance_to_dataframe(participants: Dict[str, ParticipantRecord]) -> pd.DataFrame:
    """Convert participant records to a clean DataFrame."""
    rows = []
    for sid, p in participants.items():
        rows.append({
            "Student ID": p.student_id,
            "First Seen": format_duration(p.first_seen),
            "Last Seen": format_duration(p.last_seen),
            "Total Duration": format_duration(p.total_duration),
            "Presence %": f"{p.presence_ratio * 100:.1f}%",
            "Status": p.status,
        })
    return pd.DataFrame(rows)


def session_to_json(session: SessionData) -> str:
    """Serialize session data to JSON for export."""
    data = {
        "session_id": session.session_id,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "total_duration_sec": session.total_duration_sec,
        "total_participants": len(session.participants),
        "present_count": sum(1 for p in session.participants.values() if p.status == "Present"),
        "absent_count": sum(1 for p in session.participants.values() if p.status == "Absent"),
        "avg_engagement": np.mean([e.fused_score for e in session.engagement_timeline]) if session.engagement_timeline else 0,
        "suggestions_count": len(session.suggestions),
    }
    return json.dumps(data, indent=2)
