"""
Module 4: AI Teaching Suggestion Engine (Tab 4 - Epic 5)

Heuristic advisory agent generating real-time, non-intrusive
pedagogical prompts based on engagement signals.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from utils.core import (
    EngagementTimeStep,
    LOW_ENGAGEMENT_THRESHOLD,
    LOW_ENGAGEMENT_DURATION,
    ENGAGEMENT_WINDOW_SEC,
    AudioLabel,
)


@dataclass
class Suggestion:
    """A single pedagogical suggestion."""
    timestamp: float
    severity: str           # "info", "warning", "critical"
    category: str           # "engagement", "audio", "fatigue", "participation"
    title: str
    message: str
    icon: str = "💡"


class SuggestionEngine:
    """
    Heuristic advisory agent that analyzes engagement patterns
    and generates pedagogical intervention suggestions.
    """

    # Engagement thresholds
    CRITICAL_THRESHOLD = 30
    WARNING_THRESHOLD = 45
    GOOD_THRESHOLD = 65

    # Time thresholds (in seconds)
    SUSTAINED_DROP_TIME = 300   # 5 minutes
    FATIGUE_ONSET_TIME = 1200   # 20 minutes
    LONG_SILENCE_TIME = 120     # 2 minutes

    def __init__(self):
        self.suggestions: List[Suggestion] = []
        self._last_suggestion_time: float = 0
        self._min_suggestion_interval: float = 60  # min 1 min between suggestions

    def analyze_timeline(
        self,
        timeline: List[EngagementTimeStep],
        audio_labels: Optional[List[str]] = None,
        window_sec: float = ENGAGEMENT_WINDOW_SEC,
    ) -> List[Suggestion]:
        """
        Analyze the full engagement timeline and generate suggestions.

        Args:
            timeline: List of engagement time steps.
            audio_labels: Optional list of audio classification labels.
            window_sec: Duration of each time window.

        Returns:
            List of Suggestion objects.
        """
        self.suggestions.clear()

        if not timeline:
            return self.suggestions

        scores = [s.fused_score for s in timeline]
        timestamps = [s.timestamp for s in timeline]

        # ─── Pattern 1: Sustained Low Engagement ───
        self._check_sustained_drop(timeline, window_sec)

        # ─── Pattern 2: Sudden Engagement Drop ───
        self._check_sudden_drops(timeline)

        # ─── Pattern 3: Fatigue Detection ───
        self._check_fatigue(timeline, window_sec)

        # ─── Pattern 4: Audio Pattern Analysis ───
        if audio_labels:
            self._check_audio_patterns(audio_labels, timestamps, window_sec)

        # ─── Pattern 5: Recovery Opportunities ───
        self._check_recovery_opportunities(timeline)

        # ─── Pattern 6: Overall Session Health ───
        self._check_session_health(timeline)

        # Sort by timestamp
        self.suggestions.sort(key=lambda s: s.timestamp)
        return self.suggestions

    def _check_sustained_drop(
        self, timeline: List[EngagementTimeStep], window_sec: float
    ):
        """Detect sustained periods of low engagement."""
        consecutive_low = 0
        low_start = 0.0

        for step in timeline:
            if step.fused_score < self.WARNING_THRESHOLD:
                if consecutive_low == 0:
                    low_start = step.timestamp
                consecutive_low += 1

                duration = consecutive_low * window_sec
                if duration >= self.SUSTAINED_DROP_TIME:
                    if step.fused_score < self.CRITICAL_THRESHOLD:
                        self.suggestions.append(Suggestion(
                            timestamp=step.timestamp,
                            severity="critical",
                            category="engagement",
                            title="🚨 Critical Engagement Drop",
                            message=(
                                f"Collective engagement has been below {self.WARNING_THRESHOLD}% "
                                f"for {int(duration // 60)} minutes (currently at {step.fused_score:.0f}%). "
                                f"Strongly recommend an immediate intervention:\n"
                                f"• Launch a quick interactive poll or quiz\n"
                                f"• Initiate a think-pair-share discussion\n"
                                f"• Call for a 2-minute stretch break\n"
                                f"• Switch to a hands-on activity or demonstration"
                            ),
                            icon="🚨",
                        ))
                    else:
                        self.suggestions.append(Suggestion(
                            timestamp=step.timestamp,
                            severity="warning",
                            category="engagement",
                            title="⚠️ Sustained Low Engagement",
                            message=(
                                f"Engagement dropped below {self.WARNING_THRESHOLD}% for "
                                f"{int(duration // 60)} minutes. Consider:\n"
                                f"• Asking an open-ended question to the class\n"
                                f"• Introducing a brief interactive element\n"
                                f"• Changing the pace or modality of instruction"
                            ),
                            icon="⚠️",
                        ))
                    consecutive_low = 0  # Reset after generating suggestion
            else:
                consecutive_low = 0

    def _check_sudden_drops(self, timeline: List[EngagementTimeStep]):
        """Detect sudden drops in engagement."""
        for i in range(1, len(timeline)):
            drop = timeline[i - 1].fused_score - timeline[i].fused_score
            if drop > 25:
                self.suggestions.append(Suggestion(
                    timestamp=timeline[i].timestamp,
                    severity="warning",
                    category="engagement",
                    title="📉 Rapid Engagement Decline",
                    message=(
                        f"Engagement dropped by {drop:.0f} points in one window "
                        f"(from {timeline[i-1].fused_score:.0f}% to {timeline[i].fused_score:.0f}%). "
                        f"The class may have hit a difficult concept. Consider:\n"
                        f"• Pausing to check understanding\n"
                        f"• Rephrasing the current topic with an analogy\n"
                        f"• Showing a quick visual aid or example"
                    ),
                    icon="📉",
                ))

    def _check_fatigue(
        self, timeline: List[EngagementTimeStep], window_sec: float
    ):
        """Detect fatigue patterns (declining trend over time)."""
        if len(timeline) < 10:
            return

        total_time = timeline[-1].timestamp
        if total_time < self.FATIGUE_ONSET_TIME:
            return

        # Check if second half has consistently lower engagement
        mid = len(timeline) // 2
        first_half_avg = sum(s.fused_score for s in timeline[:mid]) / mid
        second_half_avg = sum(s.fused_score for s in timeline[mid:]) / (len(timeline) - mid)

        if second_half_avg < first_half_avg - 15:
            self.suggestions.append(Suggestion(
                timestamp=timeline[mid].timestamp,
                severity="info",
                category="fatigue",
                title="😴 Fatigue Pattern Detected",
                message=(
                    f"Average engagement declined from {first_half_avg:.0f}% in the first half "
                    f"to {second_half_avg:.0f}% in the second half. "
                    f"Natural cognitive fatigue may be setting in. Suggestions:\n"
                    f"• Schedule a 5-minute energizer break\n"
                    f"• Switch to a group activity or collaborative exercise\n"
                    f"• Use a gamified review (quiz competition, Kahoot-style)"
                ),
                icon="😴",
            ))

    def _check_audio_patterns(
        self,
        audio_labels: List[str],
        timestamps: List[float],
        window_sec: float,
    ):
        """Analyze audio patterns for pedagogical insights."""
        # Check for extended silence
        consecutive_silence = 0
        for i, label in enumerate(audio_labels):
            if label == AudioLabel.SILENCE.value:
                consecutive_silence += 1
                duration = consecutive_silence * window_sec
                if duration >= self.LONG_SILENCE_TIME and consecutive_silence == int(
                    self.LONG_SILENCE_TIME / window_sec
                ):
                    t = timestamps[i] if i < len(timestamps) else 0
                    self.suggestions.append(Suggestion(
                        timestamp=t,
                        severity="info",
                        category="audio",
                        title="🔇 Extended Silence Detected",
                        message=(
                            f"The classroom has been silent for {int(duration // 60)} minutes. "
                            f"This could indicate:\n"
                            f"• Students working on an assignment (expected)\n"
                            f"• Disengagement or confusion\n"
                            f"Consider checking in with the class."
                        ),
                        icon="🔇",
                    ))
            else:
                consecutive_silence = 0

        # Check for excessive distraction
        distraction_count = sum(
            1 for l in audio_labels if l == AudioLabel.DISTRACTION.value
        )
        if len(audio_labels) > 0:
            distraction_pct = distraction_count / len(audio_labels) * 100
            if distraction_pct > 30:
                self.suggestions.append(Suggestion(
                    timestamp=timestamps[0] if timestamps else 0,
                    severity="warning",
                    category="audio",
                    title="🔊 High Ambient Noise Level",
                    message=(
                        f"Ambient distraction was detected in {distraction_pct:.0f}% of audio windows. "
                        f"High noise can impede learning. Consider:\n"
                        f"• Addressing the source of distraction\n"
                        f"• Using attention-focusing techniques (countdown, visual signal)\n"
                        f"• Transitioning to a structured activity"
                    ),
                    icon="🔊",
                ))

        # Check for lack of student participation
        discussion_count = sum(
            1 for l in audio_labels if l == AudioLabel.DISCUSSION.value
        )
        if len(audio_labels) > 0:
            discussion_pct = discussion_count / len(audio_labels) * 100
            if discussion_pct < 10 and len(audio_labels) > 10:
                self.suggestions.append(Suggestion(
                    timestamp=timestamps[-1] if timestamps else 0,
                    severity="info",
                    category="participation",
                    title="🗣️ Low Student Verbal Participation",
                    message=(
                        f"Student discussion was detected in only {discussion_pct:.0f}% of the session. "
                        f"Consider encouraging more participation:\n"
                        f"• Use think-pair-share activities\n"
                        f"• Ask targeted questions to specific students\n"
                        f"• Create small group discussion opportunities"
                    ),
                    icon="🗣️",
                ))

    def _check_recovery_opportunities(self, timeline: List[EngagementTimeStep]):
        """Identify moments where engagement recovered (positive feedback)."""
        for i in range(1, len(timeline)):
            rise = timeline[i].fused_score - timeline[i - 1].fused_score
            if rise > 20 and timeline[i].fused_score > self.GOOD_THRESHOLD:
                self.suggestions.append(Suggestion(
                    timestamp=timeline[i].timestamp,
                    severity="info",
                    category="engagement",
                    title="✅ Engagement Recovery Detected",
                    message=(
                        f"Engagement jumped by {rise:.0f} points to {timeline[i].fused_score:.0f}%. "
                        f"The current teaching approach is resonating well — "
                        f"consider noting what worked for future sessions!"
                    ),
                    icon="✅",
                ))

    def _check_session_health(self, timeline: List[EngagementTimeStep]):
        """Overall session-level observations."""
        if not timeline:
            return

        avg = sum(s.fused_score for s in timeline) / len(timeline)
        if avg >= 70:
            self.suggestions.append(Suggestion(
                timestamp=timeline[-1].timestamp,
                severity="info",
                category="engagement",
                title="🌟 Excellent Session Engagement",
                message=(
                    f"Overall average engagement is {avg:.0f}% — excellent! "
                    f"The session delivery and content are highly effective."
                ),
                icon="🌟",
            ))
        elif avg < 40:
            self.suggestions.append(Suggestion(
                timestamp=timeline[-1].timestamp,
                severity="critical",
                category="engagement",
                title="📊 Session Needs Improvement",
                message=(
                    f"Overall average engagement is only {avg:.0f}%. Consider:\n"
                    f"• Restructuring content delivery with more interactive elements\n"
                    f"• Breaking long lectures into shorter segments\n"
                    f"• Incorporating multimedia and varied modalities\n"
                    f"• Pre-session survey to gauge student readiness"
                ),
                icon="📊",
            ))

    def get_suggestions_by_severity(self, severity: str) -> List[Suggestion]:
        """Filter suggestions by severity level."""
        return [s for s in self.suggestions if s.severity == severity]

    def get_suggestion_count(self) -> Dict[str, int]:
        """Return counts by severity."""
        return {
            "critical": len(self.get_suggestions_by_severity("critical")),
            "warning": len(self.get_suggestions_by_severity("warning")),
            "info": len(self.get_suggestions_by_severity("info")),
            "total": len(self.suggestions),
        }

    def reset(self):
        self.suggestions.clear()
        self._last_suggestion_time = 0
