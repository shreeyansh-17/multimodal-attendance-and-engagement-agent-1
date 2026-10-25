"""
╔═══════════════════════════════════════════════════════════════════╗
║   MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT                       ║
║   Skill Wallet — Capstone Project 5                              ║
║                                                                   ║
║   Privacy-First | 100% Offline Edge Processing                   ║
║   Zero External Cloud Video Transmission                         ║
╚═══════════════════════════════════════════════════════════════════╝

Modules:
  Tab 1 — Automated Presence-Based Attendance (Epic 2)
  Tab 2 — Acoustic Interaction & Noise Dynamics (Epic 3)
  Tab 3 — Multimodal Fusion & Engagement Scoring (Epic 4)
  Tab 4 — AI Teaching Suggestion Engine (Epic 5)
  Tab 5 — Session Analytics & Educator Report (Epic 5 & 6)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import io
import time
import datetime
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from styles import get_custom_css
from utils.core import (
    SessionData,
    ParticipantRecord,
    AudioWindow,
    EngagementTimeStep,
    AudioLabel,
    format_duration,
    compute_engagement_index,
    ENGAGEMENT_WINDOW_SEC,
)
from modules.attendance import AttendanceProcessor
from modules.acoustic import AcousticProcessor
from modules.engagement import EngagementFusion
from modules.suggestions import SuggestionEngine, Suggestion
from modules.analytics import ReportGenerator


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "session_data": SessionData(),
        "attendance_processor": None,
        "acoustic_processor": None,
        "engagement_fusion": None,
        "suggestion_engine": None,
        "report_generator": None,
        "attendance_processed": False,
        "audio_processed": False,
        "engagement_processed": False,
        "suggestions_generated": False,
        "uploaded_video_path": None,
        "uploaded_audio_path": None,
        "uploaded_image_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Initialize processors
    if st.session_state.attendance_processor is None:
        st.session_state.attendance_processor = AttendanceProcessor()
    if st.session_state.acoustic_processor is None:
        st.session_state.acoustic_processor = AcousticProcessor()
    if st.session_state.engagement_fusion is None:
        st.session_state.engagement_fusion = EngagementFusion()
    if st.session_state.suggestion_engine is None:
        st.session_state.suggestion_engine = SuggestionEngine()
    if st.session_state.report_generator is None:
        st.session_state.report_generator = ReportGenerator()

    # Generate session ID if not set
    if not st.session_state.session_data.session_id:
        st.session_state.session_data.generate_session_id()


init_session_state()


# ═══════════════════════════════════════════════════════════
#  HELPER RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def render_hero():
    """Render the hero header section."""
    st.markdown("""
    <div class="hero-header">
        <h1>🎓 Multimodal Attendance & Engagement Agent</h1>
        <div class="hero-subtitle">
            AI-powered classroom analytics — face detection, acoustic analysis, 
            and real-time engagement scoring for modern educators.
        </div>
        <div class="privacy-badge">
            🔒 PRIVACY-FIRST · 100% OFFLINE EDGE PROCESSING · ZERO CLOUD TRANSMISSION
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(icon: str, value: str, label: str):
    """Render a single KPI metric card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_suggestion_card(suggestion):
    """Render a suggestion card with severity styling."""
    if isinstance(suggestion, Suggestion):
        severity = suggestion.severity
        title = suggestion.title
        message = suggestion.message
        timestamp = suggestion.timestamp
    else:
        severity = suggestion.get("severity", "info")
        title = suggestion.get("title", "")
        message = suggestion.get("message", "")
        timestamp = suggestion.get("timestamp", 0)

    css_class = f"suggestion-{severity}"
    st.markdown(f"""
    <div class="suggestion-card {css_class}">
        <div class="suggestion-title">{title}</div>
        <div class="suggestion-message">{message}</div>
        <div class="suggestion-timestamp">⏱ at {format_duration(timestamp)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(icon: str, text: str):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def save_uploaded_file(uploaded_file, suffix=None) -> str:
    """Save an uploaded file to a temporary path and return it."""
    if suffix is None:
        suffix = os.path.splitext(uploaded_file.name)[1]
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return tmp.name


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Session Controls")
    st.markdown(f"**Session ID:** `{st.session_state.session_data.session_id}`")
    st.markdown(f"**Started:** {st.session_state.session_data.start_time[:19] if st.session_state.session_data.start_time else 'Not started'}")

    st.divider()

    st.markdown("### 📊 Processing Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{'✅' if st.session_state.attendance_processed else '⬜'} Attendance")
        st.markdown(f"{'✅' if st.session_state.audio_processed else '⬜'} Audio")
    with col2:
        st.markdown(f"{'✅' if st.session_state.engagement_processed else '⬜'} Engagement")
        st.markdown(f"{'✅' if st.session_state.suggestions_generated else '⬜'} Suggestions")

    st.divider()

    st.markdown("### 🔧 Configuration")
    presence_threshold = st.slider(
        "Presence Threshold (%)",
        min_value=50, max_value=95, value=70, step=5,
        help="Minimum % of session time a participant must be detected to be marked Present."
    )
    engagement_window = st.slider(
        "Engagement Window (sec)",
        min_value=5, max_value=30, value=10, step=5,
        help="Time window for engagement scoring aggregation."
    )

    st.divider()

    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.markdown("""
    <div style="text-align:center; color: var(--text-muted); font-size: 0.75rem; padding: 0.5rem 0;">
        <div>Skill Wallet — Capstone Project 5</div>
        <div style="margin-top:4px;">Privacy-First · Edge Processing</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════

render_hero()


# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Attendance",
    "🔊 Acoustic Analysis",
    "📈 Engagement Fusion",
    "💡 AI Suggestions",
    "📋 Session Report",
])


# ───────────────────────────────────────────────────────────
#  TAB 1: AUTOMATED PRESENCE-BASED ATTENDANCE
# ───────────────────────────────────────────────────────────

with tab1:
    st.markdown('<div class="section-header">📷 Automated Presence-Based Attendance</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a classroom video or image to detect and track participants. "
        "Sustained presence logic requires **>70% session time** to mark attendance."
    )

    input_mode = st.radio(
        "Input Source",
        ["📁 Upload Video", "🖼️ Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "📁 Upload Video":
        uploaded_video = st.file_uploader(
            "Upload Classroom Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="video_uploader",
            help="Supported formats: MP4, AVI, MOV, MKV, WebM",
        )

        if uploaded_video is not None:
            # Save to temp
            video_path = save_uploaded_file(uploaded_video)
            st.session_state.uploaded_video_path = video_path

            col_preview, col_controls = st.columns([2, 1])
            with col_preview:
                st.video(uploaded_video)
            with col_controls:
                st.markdown("**Video Details**")
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                st.markdown(f"- **Duration:** {format_duration(duration)}")
                st.markdown(f"- **FPS:** {fps:.1f}")
                st.markdown(f"- **Resolution:** {w}×{h}")
                st.markdown(f"- **Frames:** {frame_count:,}")

            if st.button("🚀 Process Video for Attendance", use_container_width=True, key="process_video_btn"):
                processor = st.session_state.attendance_processor
                processor.reset()
                processor.presence_threshold = presence_threshold / 100.0

                progress_bar = st.progress(0, text="Processing video frames...")
                frame_display = st.empty()

                def update_progress(p):
                    progress_bar.progress(p, text=f"Processing... {p*100:.0f}%")

                last_frame = [None]
                def show_frame(frame):
                    last_frame[0] = frame

                try:
                    processor.process_video(
                        video_path,
                        progress_callback=update_progress,
                        frame_callback=show_frame,
                    )

                    # Show last annotated frame
                    if last_frame[0] is not None:
                        frame_rgb = cv2.cvtColor(last_frame[0], cv2.COLOR_BGR2RGB)
                        frame_display.image(frame_rgb, caption="Last Processed Frame", use_container_width=True)

                    st.session_state.session_data.participants = processor.participants
                    st.session_state.session_data.total_duration_sec = duration
                    st.session_state.attendance_processed = True
                    progress_bar.progress(1.0, text="✅ Processing complete!")

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

    elif input_mode == "🖼️ Upload Image":
        uploaded_image = st.file_uploader(
            "Upload Classroom Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader",
            help="Supported formats: JPG, PNG, BMP, WebP",
        )

        if uploaded_image is not None:
            image_path = save_uploaded_file(uploaded_image)
            st.session_state.uploaded_image_path = image_path

            # Read and process image using getvalue() to ensure buffer is not empty
            image_bytes = uploaded_image.getvalue() if hasattr(uploaded_image, "getvalue") else uploaded_image.read()
            file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) if len(file_bytes) > 0 else None

            if img is None:
                st.error("⚠️ Unable to decode image. Please upload a valid JPG, PNG, or BMP file.")
            else:
                if st.button("🚀 Detect Faces in Image", use_container_width=True, key="process_image_btn"):
                    processor = st.session_state.attendance_processor
                    annotated, detections = processor.process_image(img)

                    col_orig, col_detected = st.columns(2)
                    with col_orig:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                    with col_detected:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected: {len(detections)} faces", use_container_width=True)

                    st.session_state.session_data.participants = processor.participants
                    st.session_state.session_data.total_duration_sec = 1.0
                    st.session_state.attendance_processed = True

    # Display attendance results
    if st.session_state.attendance_processed:
        st.markdown('<div class="section-header">📋 Attendance Register</div>', unsafe_allow_html=True)

        processor = st.session_state.attendance_processor
        stats = processor.get_summary_stats()

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("👥", str(stats["total_detected"]), "Total Detected")
        with k2:
            render_kpi_card("✅", str(stats["present"]), "Present")
        with k3:
            render_kpi_card("❌", str(stats["absent"]), "Absent")
        with k4:
            render_kpi_card("📊", f"{stats['attendance_rate']:.0f}%", "Attendance Rate")

        st.markdown("")

        # Attendance table
        df = processor.get_attendance_df()
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Attendance CSV",
                data=csv_data,
                file_name=f"attendance_{st.session_state.session_data.session_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        render_empty_state("📷", "Upload a video or image to begin attendance detection")


# ───────────────────────────────────────────────────────────
#  TAB 2: ACOUSTIC INTERACTION & NOISE DYNAMICS
# ───────────────────────────────────────────────────────────

with tab2:
    st.markdown('<div class="section-header">🔊 Acoustic Interaction & Noise Dynamics</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload audio to analyze classroom acoustic patterns. "
        "Segments are classified into **Silence**, **Instructor Lecture**, "
        "**Student Discussion**, and **High Ambient Distraction**."
    )

    audio_source = st.radio(
        "Audio Source",
        ["🎵 Upload Audio File", "🎬 Extract from Video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    audio_path = None

    if audio_source == "🎵 Upload Audio File":
        uploaded_audio = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="audio_uploader",
        )
        if uploaded_audio:
            audio_path = save_uploaded_file(uploaded_audio)
            st.session_state.uploaded_audio_path = audio_path
            st.audio(uploaded_audio)

    elif audio_source == "🎬 Extract from Video":
        if st.session_state.uploaded_video_path:
            st.info(f"📁 Using previously uploaded video for audio extraction")
            if st.button("🎵 Extract Audio Track", key="extract_audio_btn"):
                with st.spinner("Extracting audio from video..."):
                    acoustic = st.session_state.acoustic_processor
                    extracted = acoustic.extract_audio_from_video(st.session_state.uploaded_video_path)
                    if extracted:
                        audio_path = extracted
                        st.session_state.uploaded_audio_path = audio_path
                        st.success("✅ Audio extracted successfully!")
                    else:
                        st.error("Failed to extract audio. Try uploading a separate audio file.")
        else:
            st.warning("⚠️ Upload a video in the Attendance tab first, or upload audio directly.")

    # Process audio
    if audio_path or st.session_state.uploaded_audio_path:
        target = audio_path or st.session_state.uploaded_audio_path

        if st.button("🚀 Analyze Audio", use_container_width=True, key="process_audio_btn"):
            acoustic = st.session_state.acoustic_processor
            acoustic.reset()
            acoustic.window_duration = float(engagement_window)
            acoustic.hop_duration = float(engagement_window) / 2

            progress_bar = st.progress(0, text="Analyzing audio signals...")

            try:
                windows = acoustic.process_audio_file(
                    target,
                    progress_callback=lambda p: progress_bar.progress(p, f"Analyzing... {p*100:.0f}%"),
                )

                st.session_state.session_data.audio_windows = windows
                st.session_state.audio_processed = True
                progress_bar.progress(1.0, text="✅ Audio analysis complete!")

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

    # Display audio results
    if st.session_state.audio_processed:
        acoustic = st.session_state.acoustic_processor
        summary = acoustic.get_summary()

        st.markdown('<div class="section-header">📊 Audio Analysis Results</div>', unsafe_allow_html=True)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("⏱️", summary["duration"], "Duration")
        with k2:
            render_kpi_card("📢", f"{summary['lecture_pct']}%", "Lecture Time")
        with k3:
            render_kpi_card("🗣️", f"{summary['discussion_pct']}%", "Discussion Time")
        with k4:
            render_kpi_card("🎯", f"{summary['avg_interaction']}", "Avg Interaction")

        st.markdown("")

        # Audio classification charts
        col_chart, col_pie = st.columns([2, 1])

        with col_chart:
            report_gen = st.session_state.report_generator
            fig = report_gen.plot_audio_breakdown(st.session_state.session_data.audio_windows)
            st.pyplot(fig)
            plt.close(fig)

        with col_pie:
            # Audio distribution pie chart
            fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor="#0A0D14")
            ax2.set_facecolor("#0A0D14")

            labels_data = {
                "Silence": summary["silence_pct"],
                "Lecture": summary["lecture_pct"],
                "Discussion": summary["discussion_pct"],
                "Distraction": summary["distraction_pct"],
            }
            # Filter out zero values
            labels_data = {k: v for k, v in labels_data.items() if v > 0}

            if labels_data:
                colors_map = {
                    "Silence": "#4A4E69",
                    "Lecture": "#42A5F5",
                    "Discussion": "#00D4AA",
                    "Distraction": "#FF6B6B",
                }
                pie_colors = [colors_map[k] for k in labels_data.keys()]
                ax2.pie(
                    labels_data.values(),
                    labels=labels_data.keys(),
                    colors=pie_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": "#F0F2F6", "fontsize": 10},
                )
                ax2.set_title("🔊 Audio Distribution", color="#F0F2F6",
                             fontsize=14, fontweight="bold", pad=15)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # Audio windows table
        with st.expander("📋 Detailed Audio Windows", expanded=False):
            df_audio = acoustic.get_windows_dataframe()
            st.dataframe(df_audio, use_container_width=True, hide_index=True)
    else:
        render_empty_state("🔊", "Upload audio or extract from video to analyze classroom acoustics")


# ───────────────────────────────────────────────────────────
#  TAB 3: MULTIMODAL FUSION & ENGAGEMENT SCORING
# ───────────────────────────────────────────────────────────

with tab3:
    st.markdown('<div class="section-header">📈 Multimodal Fusion & Engagement Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        "Fuses **visual attentiveness** (head pose, posture) and **audio interaction energy** "
        "into a unified Engagement Index (0-100) with real-time timeline curves."
    )

    # Check prerequisites
    has_attendance = st.session_state.attendance_processed
    has_audio = st.session_state.audio_processed

    if not has_attendance and not has_audio:
        st.warning("⚠️ Process both **Attendance** (Tab 1) and **Audio** (Tab 2) first for best results.")
        render_empty_state("📈", "Complete attendance and audio processing to generate engagement scores")
    else:
        status_msg = []
        if has_attendance:
            status_msg.append("✅ Visual data ready")
        else:
            status_msg.append("⬜ No visual data — using defaults")
        if has_audio:
            status_msg.append("✅ Audio data ready")
        else:
            status_msg.append("⬜ No audio data — using defaults")

        st.markdown(" · ".join(status_msg))

        if st.button("🚀 Compute Engagement Fusion", use_container_width=True, key="compute_engagement_btn"):
            fusion = st.session_state.engagement_fusion

            session = st.session_state.session_data
            duration = session.total_duration_sec or 60.0  # default 1 min

            with st.spinner("Computing multimodal engagement fusion..."):
                try:
                    # If we have video, do frame-by-frame visual analysis
                    if st.session_state.uploaded_video_path and has_audio:
                        progress_bar = st.progress(0, text="Analyzing visual attention from video...")
                        timeline = fusion.fuse_video_with_audio(
                            st.session_state.uploaded_video_path,
                            session.audio_windows,
                            progress_callback=lambda p: progress_bar.progress(p, f"Fusing... {p*100:.0f}%"),
                        )
                        progress_bar.progress(1.0, text="✅ Fusion complete!")
                    else:
                        # Use pre-computed attendance records
                        timeline = fusion.fuse_from_records(
                            session.participants,
                            session.audio_windows,
                            duration,
                        )

                    session.engagement_timeline = timeline
                    st.session_state.engagement_processed = True

                except Exception as e:
                    st.error(f"Error computing engagement: {str(e)}")

    # Display engagement results
    if st.session_state.engagement_processed:
        fusion = st.session_state.engagement_fusion
        eng_summary = fusion.get_summary()

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi_card("📊", f"{eng_summary['avg_engagement']}", "Avg Engagement")
        with k2:
            render_kpi_card("🔝", f"{eng_summary['peak_engagement']}", "Peak Score")
        with k3:
            render_kpi_card("📉", f"{eng_summary['low_engagement']}", "Lowest Score")
        with k4:
            render_kpi_card("👁️", f"{eng_summary['avg_visual']}", "Visual Avg")
        with k5:
            render_kpi_card("🔊", f"{eng_summary['avg_audio']}", "Audio Avg")

        st.markdown("")

        # Engagement timeline chart
        report_gen = st.session_state.report_generator
        fig = report_gen.plot_engagement_timeline(st.session_state.session_data.engagement_timeline)
        st.pyplot(fig)
        plt.close(fig)

        # Distribution and details
        col_dist, col_data = st.columns([1, 1])

        with col_dist:
            fig2 = report_gen.plot_engagement_distribution(st.session_state.session_data.engagement_timeline)
            st.pyplot(fig2)
            plt.close(fig2)

        with col_data:
            st.markdown("**📋 Engagement Data**")
            df_eng = fusion.get_timeline_dataframe()
            st.dataframe(df_eng, use_container_width=True, hide_index=True, height=350)


# ───────────────────────────────────────────────────────────
#  TAB 4: AI TEACHING SUGGESTION ENGINE
# ───────────────────────────────────────────────────────────

with tab4:
    st.markdown('<div class="section-header">💡 AI Teaching Suggestion Engine</div>', unsafe_allow_html=True)
    st.markdown(
        "Heuristic advisory agent generating **non-intrusive pedagogical prompts** "
        "based on engagement patterns — designed for collective feedback, not surveillance."
    )

    if not st.session_state.engagement_processed:
        st.warning("⚠️ Compute **Engagement Fusion** (Tab 3) first to generate suggestions.")
        render_empty_state("💡", "Process engagement data to receive AI teaching recommendations")
    else:
        if st.button("🧠 Generate Teaching Suggestions", use_container_width=True, key="gen_suggestions_btn"):
            engine = st.session_state.suggestion_engine
            engine.reset()

            timeline = st.session_state.session_data.engagement_timeline
            audio_labels = [w.label for w in st.session_state.session_data.audio_windows]

            with st.spinner("Analyzing patterns and generating suggestions..."):
                suggestions = engine.analyze_timeline(
                    timeline,
                    audio_labels=audio_labels,
                    window_sec=float(engagement_window),
                )

                # Save to session data
                st.session_state.session_data.suggestions = [
                    {
                        "timestamp": s.timestamp,
                        "severity": s.severity,
                        "category": s.category,
                        "title": s.title,
                        "message": s.message,
                        "icon": s.icon,
                    }
                    for s in suggestions
                ]
                st.session_state.suggestions_generated = True

    if st.session_state.suggestions_generated:
        engine = st.session_state.suggestion_engine
        counts = engine.get_suggestion_count()

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("📝", str(counts["total"]), "Total Suggestions")
        with k2:
            render_kpi_card("🚨", str(counts["critical"]), "Critical Alerts")
        with k3:
            render_kpi_card("⚠️", str(counts["warning"]), "Warnings")
        with k4:
            render_kpi_card("ℹ️", str(counts["info"]), "Informational")

        st.markdown("")

        # Filter options
        severity_filter = st.selectbox(
            "Filter by severity",
            ["All", "Critical", "Warning", "Info"],
            key="severity_filter",
        )

        # Display suggestions
        suggestions = engine.suggestions
        if severity_filter != "All":
            suggestions = engine.get_suggestions_by_severity(severity_filter.lower())

        if suggestions:
            for s in suggestions:
                render_suggestion_card(s)
        else:
            render_empty_state("✨", "No suggestions matching the selected filter")


# ───────────────────────────────────────────────────────────
#  TAB 5: SESSION ANALYTICS & EDUCATOR REPORT
# ───────────────────────────────────────────────────────────

with tab5:
    st.markdown('<div class="section-header">📋 Session Analytics & Educator Report</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive post-session summary with key KPIs, participation rates, and downloadable reports.")

    session = st.session_state.session_data
    report_gen = st.session_state.report_generator

    # Check if any data is available
    has_any_data = (
        st.session_state.attendance_processed
        or st.session_state.audio_processed
        or st.session_state.engagement_processed
    )

    if not has_any_data:
        st.warning("⚠️ Process data in the other tabs first to generate the session report.")
        render_empty_state("📋", "Complete at least one analysis module to view the session report")
    else:
        # Finalize session
        session.finalize()
        kpis = report_gen.compute_session_kpis(session)

        # ── Session Overview KPIs ──
        st.markdown("### 📊 Session Overview")

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi_card("⏱️", kpis["duration"], "Duration")
        with k2:
            render_kpi_card("👥", str(kpis["total_participants"]), "Participants")
        with k3:
            render_kpi_card("✅", f"{kpis['attendance_rate']}%", "Attendance Rate")
        with k4:
            render_kpi_card("📈", f"{kpis['avg_engagement']}", "Avg Engagement")
        with k5:
            render_kpi_card("📝", str(kpis["total_suggestions"]), "Suggestions")

        st.markdown("")

        # ── Charts Row ──
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            if st.session_state.engagement_processed:
                fig = report_gen.plot_engagement_timeline(session.engagement_timeline)
                st.pyplot(fig)
                plt.close(fig)

        with col_chart2:
            if st.session_state.attendance_processed:
                fig = report_gen.plot_attendance_donut(kpis["present"], kpis["absent"])
                st.pyplot(fig)
                plt.close(fig)

        # ── Detailed Breakdown ──
        st.markdown("### 📋 Detailed Breakdown")

        col_att, col_aud = st.columns(2)

        with col_att:
            st.markdown("**🎓 Attendance Summary**")
            if st.session_state.attendance_processed:
                att_data = {
                    "Metric": ["Total Detected", "Present", "Absent", "Attendance Rate", "Session Duration"],
                    "Value": [
                        kpis["total_participants"],
                        kpis["present"],
                        kpis["absent"],
                        f"{kpis['attendance_rate']}%",
                        kpis["duration"],
                    ],
                }
                st.dataframe(pd.DataFrame(att_data), use_container_width=True, hide_index=True)
            else:
                st.info("No attendance data processed yet.")

        with col_aud:
            st.markdown("**🔊 Audio Summary**")
            if st.session_state.audio_processed:
                aud_data = {
                    "Category": ["Silence", "Instructor Lecture", "Student Discussion", "Ambient Distraction"],
                    "Percentage": [
                        f"{kpis['silence_pct']}%",
                        f"{kpis['lecture_pct']}%",
                        f"{kpis['discussion_pct']}%",
                        f"{kpis['distraction_pct']}%",
                    ],
                }
                st.dataframe(pd.DataFrame(aud_data), use_container_width=True, hide_index=True)
            else:
                st.info("No audio data processed yet.")

        # ── Engagement Summary ──
        if st.session_state.engagement_processed:
            st.markdown("**📈 Engagement Summary**")
            eng_data = {
                "Metric": [
                    "Average Engagement", "Peak Engagement", "Minimum Engagement",
                    "Engagement Stability", "Attention Peaks", "Attention Dips",
                ],
                "Value": [
                    f"{kpis['avg_engagement']}",
                    f"{kpis['peak_engagement']}",
                    f"{kpis['min_engagement']}",
                    f"{kpis['engagement_stability']}%",
                    str(st.session_state.engagement_fusion.get_summary().get("attention_peaks", 0)),
                    str(st.session_state.engagement_fusion.get_summary().get("attention_dips", 0)),
                ],
            }
            st.dataframe(pd.DataFrame(eng_data), use_container_width=True, hide_index=True)

        # ── Export Section ──
        st.markdown("### 📥 Export Reports")

        col_csv, col_json = st.columns(2)

        with col_csv:
            if st.session_state.attendance_processed:
                csv_data = report_gen.generate_csv_report(session)
                st.download_button(
                    label="📥 Download Attendance CSV",
                    data=csv_data,
                    file_name=f"attendance_report_{session.session_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_json:
            json_data = report_gen.generate_json_report(session)
            st.download_button(
                label="📥 Download Full Session JSON",
                data=json_data,
                file_name=f"session_report_{session.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )

        # ── Key Suggestions Preview ──
        if st.session_state.suggestions_generated and session.suggestions:
            st.markdown("### 💡 Key Recommendations")
            # Show top 3 critical/warning suggestions
            critical_warnings = [
                s for s in session.suggestions
                if s.get("severity") in ("critical", "warning")
            ][:3]

            if critical_warnings:
                for s in critical_warnings:
                    render_suggestion_card(s)
            else:
                st.success("🌟 No critical issues detected — great session!")


# ═══════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div class="app-footer">
    <strong>Multimodal Attendance & Engagement Agent</strong> · Skill Wallet Capstone Project 5<br>
    🔒 Privacy-First · 100% Offline Edge Processing · Zero Cloud Transmission<br>
    Built with Streamlit · OpenCV · MediaPipe · Librosa
</div>
""", unsafe_allow_html=True)
