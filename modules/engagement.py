"""
Module 3: Multimodal Fusion & Engagement Scoring (Tab 3 - Epic 4)

Fuses visual attentiveness indicators (head pose, posture) with
audio interaction energy into a unified engagement matrix.
Computes normalized Engagement Index (0-100).
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
import pandas as pd

from utils.core import (
    EngagementTimeStep,
    ParticipantRecord,
    AudioWindow,
    VISUAL_WEIGHT,
    AUDIO_WEIGHT,
    compute_engagement_index,
    normalize_score,
    ENGAGEMENT_WINDOW_SEC,
)


import os

CASCADE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "haarcascade_eye.xml")


class HeadPoseEstimator:
    """Estimate head pose and attention using MediaPipe Face Mesh or OpenCV fallback."""

    def __init__(self):
        self.use_mediapipe = False
        self.face_mesh = None
        self.face_cascade = None
        self.eye_cascade = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.use_mediapipe = True
            except Exception:
                self.use_mediapipe = False

        if not self.use_mediapipe:
            cascade_file = CASCADE_PATH if os.path.exists(CASCADE_PATH) else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_file):
                self.face_cascade = cv2.CascadeClassifier(cascade_file)
            eye_file = EYE_CASCADE_PATH if os.path.exists(EYE_CASCADE_PATH) else cv2.data.haarcascades + "haarcascade_eye.xml"
            if os.path.exists(eye_file):
                self.eye_cascade = cv2.CascadeClassifier(eye_file)

        # Key landmark indices for pose estimation
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        self.FOREHEAD = 10

    def estimate_attention(self, frame: np.ndarray) -> List[Dict]:
        """
        Estimate attention for all detected faces in a frame.

        Returns list of dicts with:
          - face_id: int
          - yaw: float (left-right head rotation)
          - pitch: float (up-down head rotation)
          - attention_score: float (0-100)
          - nose_pos: key face coordinates
        """
        attention_data = []

        if self.use_mediapipe and self.face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return attention_data

            h, w, _ = frame.shape

            for face_idx, face_lm in enumerate(results.multi_face_landmarks):
                landmarks = face_lm.landmark

                # Get 3D coordinates of key points
                nose = np.array([
                    landmarks[self.NOSE_TIP].x * w,
                    landmarks[self.NOSE_TIP].y * h,
                    landmarks[self.NOSE_TIP].z * w,
                ])
                left_eye = np.array([
                    landmarks[self.LEFT_EYE].x * w,
                    landmarks[self.LEFT_EYE].y * h,
                    landmarks[self.LEFT_EYE].z * w,
                ])
                right_eye = np.array([
                    landmarks[self.RIGHT_EYE].x * w,
                    landmarks[self.RIGHT_EYE].y * h,
                    landmarks[self.RIGHT_EYE].z * w,
                ])
                chin = np.array([
                    landmarks[self.CHIN].x * w,
                    landmarks[self.CHIN].y * h,
                    landmarks[self.CHIN].z * w,
                ])
                forehead = np.array([
                    landmarks[self.FOREHEAD].x * w,
                    landmarks[self.FOREHEAD].y * h,
                    landmarks[self.FOREHEAD].z * w,
                ])

                # Estimate yaw (horizontal rotation)
                eye_center = (left_eye + right_eye) / 2
                eye_width = np.linalg.norm(left_eye[:2] - right_eye[:2])
                nose_offset = nose[0] - eye_center[0]
                yaw = np.degrees(np.arctan2(nose_offset, eye_width)) if eye_width > 0 else 0

                # Estimate pitch (vertical rotation)
                face_height = np.linalg.norm(forehead[:2] - chin[:2])
                nose_vert_offset = nose[1] - eye_center[1]
                pitch = np.degrees(np.arctan2(nose_vert_offset, face_height)) if face_height > 0 else 0

                # Compute attention score
                yaw_penalty = min(abs(yaw) / 45.0, 1.0)
                pitch_penalty = min(abs(pitch - 10) / 35.0, 1.0)
                attention = max(0, 100 * (1 - 0.6 * yaw_penalty - 0.4 * pitch_penalty))

                attention_data.append({
                    "face_id": face_idx,
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "attention_score": round(attention, 2),
                    "nose_pos": (int(nose[0]), int(nose[1])),
                })
        elif self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for face_idx, (x, y, w, h) in enumerate(faces):
                roi_gray = gray[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray) if self.eye_cascade else []
                if len(eyes) >= 2:
                    score = 90.0
                    yaw, pitch = 2.0, 5.0
                elif len(eyes) == 1:
                    score = 72.0
                    yaw, pitch = 15.0, 8.0
                else:
                    score = 60.0
                    yaw, pitch = 25.0, 12.0

                attention_data.append({
                    "face_id": face_idx,
                    "yaw": yaw,
                    "pitch": pitch,
                    "attention_score": score,
                    "nose_pos": (int(x + w // 2), int(y + h // 2)),
                })

        return attention_data

    def close(self):
        if self.use_mediapipe and self.face_mesh and hasattr(self.face_mesh, "close"):
            self.face_mesh.close()


class EngagementFusion:
    """
    Fuses visual and audio signals into a unified engagement timeline.
    """

    def __init__(self, window_sec: float = ENGAGEMENT_WINDOW_SEC):
        self.window_sec = window_sec
        self.timeline: List[EngagementTimeStep] = []
        self.head_pose_estimator = HeadPoseEstimator()

    def compute_visual_score_from_frame(self, frame: np.ndarray) -> float:
        """Compute average visual attention score for a frame."""
        attention_data = self.head_pose_estimator.estimate_attention(frame)
        if not attention_data:
            return 50.0  # neutral when no faces
        return np.mean([a["attention_score"] for a in attention_data])

    def fuse_from_records(
        self,
        participant_records: Dict[str, ParticipantRecord],
        audio_windows: List[AudioWindow],
        total_duration: float,
    ) -> List[EngagementTimeStep]:
        """
        Create engagement timeline from pre-computed attendance and audio data.

        Args:
            participant_records: Dict of participant records with visual scores.
            audio_windows: List of classified audio windows.
            total_duration: Total session duration in seconds.

        Returns:
            List of EngagementTimeStep objects.
        """
        self.timeline.clear()
        n_steps = max(1, int(total_duration / self.window_sec))

        for i in range(n_steps):
            t = i * self.window_sec

            # Visual score: average of participants' visual engagement
            visual_scores = []
            for p in participant_records.values():
                if p.head_pose_scores:
                    # Find scores near this timestamp
                    idx = min(
                        int(t / total_duration * len(p.head_pose_scores)),
                        len(p.head_pose_scores) - 1,
                    )
                    visual_scores.append(p.head_pose_scores[idx])
                else:
                    # Use presence as a proxy
                    if p.first_seen <= t <= p.last_seen:
                        visual_scores.append(65.0)
                    else:
                        visual_scores.append(20.0)

            visual = np.mean(visual_scores) if visual_scores else 50.0

            # Audio score: find matching audio window
            audio = 50.0
            for aw in audio_windows:
                if aw.start_time <= t < aw.end_time:
                    audio = aw.interaction_score
                    break

            fused = compute_engagement_index(visual, audio)

            self.timeline.append(
                EngagementTimeStep(
                    timestamp=round(t, 2),
                    visual_score=round(visual, 2),
                    audio_score=round(audio, 2),
                    fused_score=round(fused, 2),
                )
            )

        return self.timeline

    def fuse_video_with_audio(
        self,
        video_path: str,
        audio_windows: List[AudioWindow],
        progress_callback=None,
    ) -> List[EngagementTimeStep]:
        """
        Process video frame-by-frame for visual attention and fuse with audio.
        """
        self.timeline.clear()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self.timeline

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        window_frames = int(self.window_sec * fps)
        frame_idx = 0
        window_scores = []
        current_window_start = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps

                # Sample every ~1 second for efficiency
                if frame_idx % int(fps) == 0:
                    score = self.compute_visual_score_from_frame(frame)
                    window_scores.append(score)

                # End of window
                if (frame_idx + 1) % window_frames == 0 or frame_idx == total_frames - 1:
                    visual = np.mean(window_scores) if window_scores else 50.0

                    # Match audio window
                    audio = 50.0
                    mid_t = current_window_start + self.window_sec / 2
                    for aw in audio_windows:
                        if aw.start_time <= mid_t < aw.end_time:
                            audio = aw.interaction_score
                            break

                    fused = compute_engagement_index(visual, audio)
                    self.timeline.append(
                        EngagementTimeStep(
                            timestamp=round(current_window_start, 2),
                            visual_score=round(visual, 2),
                            audio_score=round(audio, 2),
                            fused_score=round(fused, 2),
                        )
                    )

                    current_window_start += self.window_sec
                    window_scores = []

                frame_idx += 1

                if progress_callback and total_frames > 0:
                    progress_callback(min(frame_idx / total_frames, 1.0))
        finally:
            cap.release()

        return self.timeline

    def get_timeline_dataframe(self) -> pd.DataFrame:
        """Return timeline as DataFrame."""
        rows = []
        for step in self.timeline:
            rows.append({
                "Time (s)": step.timestamp,
                "Visual Score": step.visual_score,
                "Audio Score": step.audio_score,
                "Engagement Index": step.fused_score,
            })
        return pd.DataFrame(rows)

    def get_summary(self) -> dict:
        """Return engagement summary statistics."""
        if not self.timeline:
            return {
                "avg_engagement": 0,
                "peak_engagement": 0,
                "low_engagement": 0,
                "avg_visual": 0,
                "avg_audio": 0,
                "attention_peaks": 0,
                "attention_dips": 0,
            }

        scores = [s.fused_score for s in self.timeline]
        visual = [s.visual_score for s in self.timeline]
        audio = [s.audio_score for s in self.timeline]

        # Detect peaks and dips
        peaks = sum(1 for i in range(1, len(scores) - 1)
                    if scores[i] > scores[i-1] and scores[i] > scores[i+1] and scores[i] > 70)
        dips = sum(1 for i in range(1, len(scores) - 1)
                   if scores[i] < scores[i-1] and scores[i] < scores[i+1] and scores[i] < 40)

        return {
            "avg_engagement": round(np.mean(scores), 2),
            "peak_engagement": round(max(scores), 2),
            "low_engagement": round(min(scores), 2),
            "avg_visual": round(np.mean(visual), 2),
            "avg_audio": round(np.mean(audio), 2),
            "attention_peaks": peaks,
            "attention_dips": dips,
        }

    def reset(self):
        self.timeline.clear()

    def close(self):
        self.head_pose_estimator.close()
