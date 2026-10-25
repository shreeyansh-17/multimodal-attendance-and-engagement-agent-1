"""
Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5

Custom CSS styling for the premium educational dashboard.
"""


def get_custom_css() -> str:
    """Return custom CSS for the Streamlit dashboard."""
    return """
    <style>
    /* ═══════════════════════════════════════════════════════════
       MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT — DASHBOARD CSS
       Premium Dark Educational Theme
       ═══════════════════════════════════════════════════════════ */

    /* ── Google Fonts Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ── */
    :root {
        --primary: #6C63FF;
        --primary-glow: rgba(108, 99, 255, 0.25);
        --secondary: #00D4AA;
        --secondary-glow: rgba(0, 212, 170, 0.2);
        --accent-red: #FF6B6B;
        --accent-orange: #FFB347;
        --accent-blue: #42A5F5;
        --bg-main: #0A0D14;
        --bg-card: #12151F;
        --bg-card-hover: #1A1D2E;
        --bg-surface: #1E2130;
        --border: rgba(108, 99, 255, 0.15);
        --border-hover: rgba(108, 99, 255, 0.35);
        --text-primary: #F0F2F6;
        --text-secondary: #8B8D97;
        --text-muted: #5A5C66;
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
        --shadow-glow: 0 0 20px var(--primary-glow);
        --radius: 14px;
        --radius-sm: 8px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global Body ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: var(--bg-main) !important;
    }

    .stApp > header {
        background: transparent !important;
    }

    /* ── Main Content Area ── */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #12151F 0%, #1A1040 50%, #12151F 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 50%, var(--primary-glow) 0%, transparent 60%);
        animation: pulse-glow 8s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes pulse-glow {
        0% { opacity: 0.3; transform: scale(1); }
        100% { opacity: 0.6; transform: scale(1.1); }
    }

    .hero-header h1 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #F0F2F6 0%, var(--primary) 50%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem !important;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid rgba(0, 212, 170, 0.25);
        color: var(--secondary);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* ── KPI Metric Cards ── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
    }

    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 0 0 var(--radius) var(--radius);
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .kpi-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.3rem;
    }

    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }

    /* ── Status Badges ── */
    .status-present {
        background: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .status-absent {
        background: rgba(255, 107, 107, 0.15);
        color: var(--accent-red);
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--border);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem !important;
        margin-top: 1.5rem !important;
    }

    /* ── Suggestion Cards ── */
    .suggestion-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }

    .suggestion-card:hover {
        transform: translateX(4px);
    }

    .suggestion-critical {
        border-left: 4px solid var(--accent-red);
        background: linear-gradient(90deg, rgba(255, 107, 107, 0.06), var(--bg-card));
    }

    .suggestion-warning {
        border-left: 4px solid var(--accent-orange);
        background: linear-gradient(90deg, rgba(255, 179, 71, 0.06), var(--bg-card));
    }

    .suggestion-info {
        border-left: 4px solid var(--accent-blue);
        background: linear-gradient(90deg, rgba(66, 165, 245, 0.06), var(--bg-card));
    }

    .suggestion-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }

    .suggestion-message {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.6;
    }

    .suggestion-timestamp {
        font-size: 0.72rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.4rem;
    }

    /* ── Tabs Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 4px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-secondary);
        background: transparent;
        border: none;
        transition: var(--transition);
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: var(--bg-surface);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), #5a52d5) !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Data Tables ── */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), #5a52d5) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: var(--transition) !important;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px var(--primary-glow) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--secondary), #00B894) !important;
        color: #0A0D14 !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 700 !important;
    }

    .stDownloadButton > button:hover {
        box-shadow: 0 4px 15px var(--secondary-glow) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader {
        border-radius: var(--radius) !important;
    }

    .stFileUploader > div {
        border: 2px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--bg-card) !important;
        transition: var(--transition);
    }

    .stFileUploader > div:hover {
        border-color: var(--primary) !important;
        background: var(--bg-card-hover) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    }

    /* ── Selectbox / Slider ── */
    .stSelectbox [data-baseweb="select"] {
        background: var(--bg-card);
        border-color: var(--border);
        border-radius: var(--radius-sm);
    }

    /* ── Metric Widget ── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        transition: var(--transition);
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-glow);
    }

    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Alert / Info boxes ── */
    .stAlert {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--text-muted);
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    .empty-state-text {
        font-size: 1rem;
        color: var(--text-secondary);
    }

    /* ── Audio Label Pills ── */
    .audio-label {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .audio-silence { background: rgba(74, 78, 105, 0.3); color: #9EA2B8; }
    .audio-lecture { background: rgba(66, 165, 245, 0.15); color: #42A5F5; }
    .audio-discussion { background: rgba(0, 212, 170, 0.15); color: #00D4AA; }
    .audio-distraction { background: rgba(255, 107, 107, 0.15); color: #FF6B6B; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-main); }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.8rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-header h1 { font-size: 1.3rem !important; }
        .kpi-value { font-size: 1.5rem; }
        .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
    </style>
    """
