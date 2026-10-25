"""
Module 5: Session Analytics & Educator Report (Tab 5 - Epic 5 & 6)

Comprehensive post-session summary dashboard with KPIs,
participation rates, and downloadable reports.
"""

import json
import datetime
import io
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from utils.core import (
    SessionData,
    ParticipantRecord,
    AudioWindow,
    EngagementTimeStep,
    AudioLabel,
    format_duration,
    session_to_json,
)


class ReportGenerator:
    """Generates comprehensive session analytics and exportable reports."""

    # Color palette matching the dashboard theme
    COLORS = {
        "primary": "#6C63FF",
        "secondary": "#00D4AA",
        "accent": "#FF6B6B",
        "warning": "#FFB347",
        "success": "#4CAF50",
        "info": "#42A5F5",
        "bg_dark": "#0E1117",
        "bg_card": "#1A1D29",
        "text": "#FAFAFA",
        "text_muted": "#8B8D97",
        "grid": "#2A2D3A",
    }

    def __init__(self):
        plt.style.use("dark_background")

    def compute_session_kpis(self, session: SessionData) -> dict:
        """Compute key performance indicators for the session."""
        participants = session.participants
        timeline = session.engagement_timeline
        audio = session.audio_windows

        total_p = len(participants)
        present = sum(1 for p in participants.values() if p.status == "Present")
        absent = total_p - present

        engagement_scores = [s.fused_score for s in timeline] if timeline else [0]
        avg_eng = np.mean(engagement_scores)
        max_eng = max(engagement_scores) if engagement_scores else 0
        min_eng = min(engagement_scores) if engagement_scores else 0

        # Audio breakdown
        audio_labels = [w.label for w in audio]
        total_audio = len(audio_labels) or 1

        # Engagement stability (std deviation — lower is more stable)
        stability = 100 - min(np.std(engagement_scores), 100)

        return {
            "session_id": session.session_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "duration": format_duration(session.total_duration_sec),
            "total_participants": total_p,
            "present": present,
            "absent": absent,
            "attendance_rate": round(present / total_p * 100, 1) if total_p > 0 else 0,
            "avg_engagement": round(avg_eng, 1),
            "peak_engagement": round(max_eng, 1),
            "min_engagement": round(min_eng, 1),
            "engagement_stability": round(stability, 1),
            "silence_pct": round(audio_labels.count(AudioLabel.SILENCE.value) / total_audio * 100, 1),
            "lecture_pct": round(audio_labels.count(AudioLabel.INSTRUCTOR.value) / total_audio * 100, 1),
            "discussion_pct": round(audio_labels.count(AudioLabel.DISCUSSION.value) / total_audio * 100, 1),
            "distraction_pct": round(audio_labels.count(AudioLabel.DISTRACTION.value) / total_audio * 100, 1),
            "total_suggestions": len(session.suggestions),
            "critical_alerts": sum(1 for s in session.suggestions if s.get("severity") == "critical"),
        }

    def plot_engagement_timeline(
        self, timeline: List[EngagementTimeStep], figsize: tuple = (12, 5)
    ) -> Figure:
        """Plot engagement timeline with visual, audio, and fused scores."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLORS["bg_dark"])
        ax.set_facecolor(self.COLORS["bg_dark"])

        if not timeline:
            ax.text(0.5, 0.5, "No engagement data available",
                    ha="center", va="center", color=self.COLORS["text_muted"],
                    fontsize=14, transform=ax.transAxes)
            return fig

        times = [s.timestamp / 60 for s in timeline]  # Convert to minutes
        visual = [s.visual_score for s in timeline]
        audio = [s.audio_score for s in timeline]
        fused = [s.fused_score for s in timeline]

        # Plot lines
        ax.fill_between(times, fused, alpha=0.15, color=self.COLORS["primary"])
        ax.plot(times, fused, color=self.COLORS["primary"], linewidth=2.5,
                label="Engagement Index", zorder=3)
        ax.plot(times, visual, color=self.COLORS["secondary"], linewidth=1.5,
                alpha=0.7, linestyle="--", label="Visual Score")
        ax.plot(times, audio, color=self.COLORS["warning"], linewidth=1.5,
                alpha=0.7, linestyle=":", label="Audio Score")

        # Threshold lines
        ax.axhline(y=45, color=self.COLORS["accent"], linestyle="-.",
                   alpha=0.4, linewidth=1, label="Warning Threshold")
        ax.axhline(y=70, color=self.COLORS["success"], linestyle="-.",
                   alpha=0.4, linewidth=1, label="Good Threshold")

        # Styling
        ax.set_xlabel("Time (minutes)", color=self.COLORS["text"], fontsize=11)
        ax.set_ylabel("Score (0-100)", color=self.COLORS["text"], fontsize=11)
        ax.set_title("📊 Engagement Timeline", color=self.COLORS["text"],
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
        ax.grid(True, alpha=0.15, color=self.COLORS["grid"])
        ax.tick_params(colors=self.COLORS["text_muted"])
        for spine in ax.spines.values():
            spine.set_color(self.COLORS["grid"])

        plt.tight_layout()
        return fig

    def plot_attendance_donut(
        self, present: int, absent: int, figsize: tuple = (5, 5)
    ) -> Figure:
        """Create a donut chart for attendance breakdown."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLORS["bg_dark"])

        if present + absent == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=self.COLORS["text_muted"], fontsize=14,
                    transform=ax.transAxes)
            return fig

        sizes = [present, absent]
        colors = [self.COLORS["success"], self.COLORS["accent"]]
        labels = [f"Present ({present})", f"Absent ({absent})"]
        explode = (0.03, 0.03)

        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, labels=labels, autopct="%1.1f%%",
            startangle=90, explode=explode, pctdistance=0.75,
            textprops={"color": self.COLORS["text"], "fontsize": 10},
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight("bold")

        # Draw center circle for donut effect
        centre = plt.Circle((0, 0), 0.55, fc=self.COLORS["bg_dark"])
        ax.add_artist(centre)

        total = present + absent
        rate = present / total * 100 if total > 0 else 0
        ax.text(0, 0, f"{rate:.0f}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color=self.COLORS["primary"])

        ax.set_title("🎓 Attendance", color=self.COLORS["text"],
                     fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        return fig

    def plot_audio_breakdown(
        self, audio_windows: List[AudioWindow], figsize: tuple = (8, 5)
    ) -> Figure:
        """Plot audio classification breakdown as stacked bar."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLORS["bg_dark"])
        ax.set_facecolor(self.COLORS["bg_dark"])

        if not audio_windows:
            ax.text(0.5, 0.5, "No audio data available",
                    ha="center", va="center", color=self.COLORS["text_muted"],
                    fontsize=14, transform=ax.transAxes)
            return fig

        label_colors = {
            AudioLabel.SILENCE.value: "#4A4E69",
            AudioLabel.INSTRUCTOR.value: self.COLORS["info"],
            AudioLabel.DISCUSSION.value: self.COLORS["secondary"],
            AudioLabel.DISTRACTION.value: self.COLORS["accent"],
        }

        times = [w.start_time / 60 for w in audio_windows]
        for label_val in AudioLabel:
            heights = [1 if w.label == label_val.value else 0 for w in audio_windows]
            color = label_colors.get(label_val.value, "#666")
            ax.bar(times, heights, width=(times[1] - times[0]) if len(times) > 1 else 0.1,
                   color=color, alpha=0.8, label=label_val.value)

        ax.set_xlabel("Time (minutes)", color=self.COLORS["text"], fontsize=11)
        ax.set_ylabel("Classification", color=self.COLORS["text"], fontsize=11)
        ax.set_title("🔊 Audio Classification Timeline", color=self.COLORS["text"],
                     fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
        ax.grid(True, alpha=0.15, color=self.COLORS["grid"])
        ax.tick_params(colors=self.COLORS["text_muted"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["", "Active"])
        for spine in ax.spines.values():
            spine.set_color(self.COLORS["grid"])

        plt.tight_layout()
        return fig

    def plot_engagement_distribution(
        self, timeline: List[EngagementTimeStep], figsize: tuple = (6, 4)
    ) -> Figure:
        """Plot histogram of engagement score distribution."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLORS["bg_dark"])
        ax.set_facecolor(self.COLORS["bg_dark"])

        if not timeline:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=self.COLORS["text_muted"], fontsize=14,
                    transform=ax.transAxes)
            return fig

        scores = [s.fused_score for s in timeline]

        n, bins, patches = ax.hist(scores, bins=20, edgecolor=self.COLORS["bg_dark"],
                                   linewidth=0.5, alpha=0.85)

        # Color bins by score range
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge < 30:
                patch.set_facecolor(self.COLORS["accent"])
            elif left_edge < 45:
                patch.set_facecolor(self.COLORS["warning"])
            elif left_edge < 70:
                patch.set_facecolor(self.COLORS["info"])
            else:
                patch.set_facecolor(self.COLORS["success"])

        ax.axvline(np.mean(scores), color=self.COLORS["primary"], linestyle="--",
                   linewidth=2, label=f"Mean: {np.mean(scores):.1f}")
        ax.set_xlabel("Engagement Index", color=self.COLORS["text"], fontsize=11)
        ax.set_ylabel("Frequency", color=self.COLORS["text"], fontsize=11)
        ax.set_title("📈 Score Distribution", color=self.COLORS["text"],
                     fontsize=14, fontweight="bold", pad=15)
        ax.legend(fontsize=9, framealpha=0.3)
        ax.grid(True, alpha=0.15, color=self.COLORS["grid"])
        ax.tick_params(colors=self.COLORS["text_muted"])
        for spine in ax.spines.values():
            spine.set_color(self.COLORS["grid"])

        plt.tight_layout()
        return fig

    def generate_csv_report(self, session: SessionData) -> str:
        """Generate CSV content for attendance report."""
        df = pd.DataFrame()
        rows = []
        for sid, p in session.participants.items():
            rows.append({
                "Student ID": p.student_id,
                "First Seen": format_duration(p.first_seen),
                "Last Seen": format_duration(p.last_seen),
                "Total Duration": format_duration(p.total_duration),
                "Presence Ratio": f"{p.presence_ratio:.2%}",
                "Status": p.status,
            })
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def generate_json_report(self, session: SessionData) -> str:
        """Generate comprehensive JSON report."""
        kpis = self.compute_session_kpis(session)
        report = {
            "report_generated": datetime.datetime.now().isoformat(),
            "session_kpis": kpis,
            "attendance_records": [
                {
                    "student_id": p.student_id,
                    "first_seen": format_duration(p.first_seen),
                    "last_seen": format_duration(p.last_seen),
                    "duration": format_duration(p.total_duration),
                    "presence_pct": round(p.presence_ratio * 100, 1),
                    "status": p.status,
                }
                for p in session.participants.values()
            ],
            "engagement_summary": {
                "avg": kpis["avg_engagement"],
                "peak": kpis["peak_engagement"],
                "min": kpis["min_engagement"],
                "stability": kpis["engagement_stability"],
            },
            "audio_summary": {
                "silence_pct": kpis["silence_pct"],
                "lecture_pct": kpis["lecture_pct"],
                "discussion_pct": kpis["discussion_pct"],
                "distraction_pct": kpis["distraction_pct"],
            },
            "suggestions": [
                {
                    "timestamp": s.get("timestamp", 0),
                    "severity": s.get("severity", "info"),
                    "category": s.get("category", ""),
                    "title": s.get("title", ""),
                    "message": s.get("message", ""),
                }
                for s in session.suggestions
            ],
        }
        return json.dumps(report, indent=2)
