"""
Vortex Theme Engine
====================

Shared theming module for all PyQt5-based Vortex Desktop apps.
Provides dark and light themes with accent color customization.

Usage:
    from vortex_theme import apply_theme, DARK, LIGHT

    app = QApplication(sys.argv)
    apply_theme(app, mode=DARK, accent='#007acc')
"""

DARK = 'dark'
LIGHT = 'light'

# Default accent colors
ACCENT_BLUE = '#007acc'
ACCENT_GREEN = '#2ecc71'
ACCENT_PURPLE = '#9b59b6'
ACCENT_ORANGE = '#e67e22'
ACCENT_RED = '#e74c3c'
ACCENT_TEAL = '#1abc9c'

ACCENT_PRESETS = {
    'Blue': ACCENT_BLUE,
    'Green': ACCENT_GREEN,
    'Purple': ACCENT_PURPLE,
    'Orange': ACCENT_ORANGE,
    'Red': ACCENT_RED,
    'Teal': ACCENT_TEAL,
}


def _darken(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color by a factor (0-1)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


def _lighten(hex_color: str, factor: float = 0.3) -> str:
    """Lighten a hex color by blending toward white."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


def get_dark_stylesheet(accent: str = ACCENT_BLUE) -> str:
    """Generate the dark theme stylesheet."""
    accent_hover = _lighten(accent, 0.15)
    accent_pressed = _darken(accent, 0.8)
    accent_faint = _darken(accent, 0.3)

    return f"""
    /* ===== DARK THEME — Vortex Desktop ===== */

    QWidget {{
        color: #e0e0e0;
        font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
        font-size: 13px;
        background-color: transparent;
    }}

    QMainWindow {{
        background-color: #1a1a1f;
    }}

    QDialog {{
        background-color: #1e1e23;
    }}

    /* ---- Sidebar ---- */
    QFrame#sidebar {{
        background-color: #16161a;
        border-right: 1px solid #2a2a2f;
    }}

    QPushButton.sidebar-btn {{
        background-color: transparent;
        border: none;
        border-radius: 6px;
        padding: 10px;
        margin: 2px 4px;
        color: #808088;
        font-size: 18px;
    }}
    QPushButton.sidebar-btn:hover {{
        background-color: #2a2a30;
        color: #c0c0c8;
    }}
    QPushButton.sidebar-btn:checked {{
        background-color: {accent_faint};
        color: {accent};
    }}

    /* ---- Cards ---- */
    QFrame.card {{
        background-color: #22222a;
        border: 1px solid #2e2e36;
        border-radius: 10px;
    }}
    QFrame.card:hover {{
        border-color: {accent};
    }}

    /* ---- Panels ---- */
    QFrame#left-panel {{
        background-color: #1e1e24;
        border-right: 1px solid #2a2a30;
    }}
    QFrame#right-panel {{
        background-color: #1e1e24;
        border-left: 1px solid #2a2a30;
    }}
    QFrame#graph-container {{
        background-color: #1a1a20;
    }}

    /* ---- Labels ---- */
    QLabel {{
        color: #e0e0e0;
        background-color: transparent;
    }}
    QLabel.heading {{
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }}
    QLabel.subheading {{
        font-size: 14px;
        font-weight: 600;
        color: #c0c0c8;
    }}
    QLabel.muted {{
        color: #707078;
        font-size: 12px;
    }}
    QLabel.accent {{
        color: {accent};
    }}

    /* ---- Buttons ---- */
    QPushButton {{
        background-color: #2e2e36;
        color: #e0e0e0;
        border: 1px solid #3a3a42;
        border-radius: 6px;
        padding: 7px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: #3a3a44;
        border-color: #4a4a54;
    }}
    QPushButton:pressed {{
        background-color: {accent};
        color: #ffffff;
        border-color: {accent};
    }}
    QPushButton:disabled {{
        background-color: #1e1e24;
        color: #505058;
        border-color: #2a2a30;
    }}
    QPushButton.primary {{
        background-color: {accent};
        color: #ffffff;
        border: none;
        font-weight: 600;
    }}
    QPushButton.primary:hover {{
        background-color: {accent_hover};
    }}
    QPushButton.primary:pressed {{
        background-color: {accent_pressed};
    }}
    QPushButton.danger {{
        background-color: #3a2028;
        color: #ff6b6b;
        border-color: #5a2030;
    }}
    QPushButton.danger:hover {{
        background-color: #4a2838;
        color: #ff8080;
    }}

    /* ---- Inputs ---- */
    QLineEdit {{
        background-color: #2a2a32;
        border: 1px solid #3a3a42;
        border-radius: 6px;
        padding: 7px 10px;
        color: #e0e0e0;
        selection-background-color: {accent};
    }}
    QLineEdit:focus {{
        border-color: {accent};
    }}

    QTextEdit {{
        background-color: #2a2a32;
        border: 1px solid #3a3a42;
        border-radius: 6px;
        padding: 6px;
        color: #e0e0e0;
        selection-background-color: {accent};
    }}

    QSpinBox, QDoubleSpinBox {{
        background-color: #2a2a32;
        border: 1px solid #3a3a42;
        border-radius: 6px;
        padding: 5px;
        color: #e0e0e0;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {accent};
    }}

    QComboBox {{
        background-color: #2a2a32;
        border: 1px solid #3a3a42;
        border-radius: 6px;
        padding: 6px 10px;
        color: #e0e0e0;
        min-width: 100px;
    }}
    QComboBox:hover {{
        border-color: #4a4a54;
    }}
    QComboBox:focus {{
        border-color: {accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #2a2a32;
        border: 1px solid #3a3a42;
        selection-background-color: {accent};
        color: #e0e0e0;
        outline: 0px;
    }}

    QSlider::groove:horizontal {{
        background-color: #3a3a42;
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background-color: {accent};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {accent};
        border-radius: 2px;
    }}

    QCheckBox {{
        spacing: 8px;
        color: #e0e0e0;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #3a3a42;
        background-color: #2a2a32;
    }}
    QCheckBox::indicator:checked {{
        background-color: {accent};
        border-color: {accent};
    }}

    /* ---- Scroll Areas ---- */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        border: none;
        background-color: transparent;
        width: 8px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background-color: #3a3a42;
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: #505058;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
        height: 0px;
    }}
    QScrollBar:horizontal {{
        border: none;
        background-color: transparent;
        height: 8px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: #3a3a42;
        min-width: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: #505058;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        border: none;
        background: none;
        width: 0px;
    }}

    /* ---- Tab Widget ---- */
    QTabWidget::pane {{
        border: 1px solid #2a2a30;
        background-color: #1e1e24;
        border-radius: 6px;
    }}
    QTabBar::tab {{
        background-color: #22222a;
        color: #808088;
        border: none;
        padding: 8px 18px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{
        background-color: #1e1e24;
        color: {accent};
        border-bottom: 2px solid {accent};
    }}
    QTabBar::tab:hover {{
        background-color: #2a2a34;
        color: #c0c0c8;
    }}

    /* ---- Group Box ---- */
    QGroupBox {{
        border: 1px solid #2e2e36;
        border-radius: 8px;
        margin-top: 16px;
        padding-top: 16px;
        font-weight: 600;
        color: #c0c0c8;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: {accent};
    }}

    /* ---- List / Table ---- */
    QListWidget {{
        background-color: #22222a;
        border: 1px solid #2e2e36;
        border-radius: 6px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 6px 10px;
        border-radius: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {accent_faint};
        color: {accent};
    }}
    QListWidget::item:hover {{
        background-color: #2a2a34;
    }}

    QTableWidget {{
        background-color: #22222a;
        border: 1px solid #2e2e36;
        border-radius: 6px;
        gridline-color: #2e2e36;
        outline: none;
    }}
    QTableWidget::item {{
        padding: 4px 8px;
    }}
    QTableWidget::item:selected {{
        background-color: {accent_faint};
        color: {accent};
    }}
    QHeaderView::section {{
        background-color: #1e1e24;
        color: #808088;
        border: none;
        border-bottom: 1px solid #2e2e36;
        padding: 6px 10px;
        font-weight: 600;
    }}

    /* ---- Tooltips ---- */
    QToolTip {{
        background-color: #2a2a32;
        color: #e0e0e0;
        border: 1px solid #3a3a42;
        border-radius: 4px;
        padding: 4px 8px;
    }}

    /* ---- Status Bar ---- */
    QStatusBar {{
        background-color: #16161a;
        color: #808088;
        border-top: 1px solid #2a2a30;
        font-size: 12px;
    }}

    /* ---- Menu Bar ---- */
    QMenuBar {{
        background-color: #1a1a1f;
        color: #c0c0c8;
        border-bottom: 1px solid #2a2a30;
    }}
    QMenuBar::item:selected {{
        background-color: #2e2e36;
    }}
    QMenu {{
        background-color: #22222a;
        border: 1px solid #2e2e36;
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {accent_faint};
        color: {accent};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: #2e2e36;
        margin: 4px 8px;
    }}

    /* ---- Progress Bar ---- */
    QProgressBar {{
        background-color: #2a2a32;
        border: none;
        border-radius: 4px;
        height: 6px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {accent};
        border-radius: 4px;
    }}

    /* ---- Splitter ---- */
    QSplitter::handle {{
        background-color: #2a2a30;
        width: 1px;
        height: 1px;
    }}

    /* ---- Toolbar ---- */
    QToolBar {{
        background-color: #1a1a1f;
        border: none;
        spacing: 4px;
        padding: 2px;
    }}
    """


def get_light_stylesheet(accent: str = ACCENT_BLUE) -> str:
    """Generate the light theme stylesheet."""
    accent_hover = _darken(accent, 0.85)
    accent_pressed = _darken(accent, 0.7)
    accent_faint = _lighten(accent, 0.85)

    return f"""
    /* ===== LIGHT THEME — Vortex Desktop ===== */

    QWidget {{
        color: #1a1a1f;
        font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
        font-size: 13px;
        background-color: transparent;
    }}

    QMainWindow {{
        background-color: #f5f5f7;
    }}

    QDialog {{
        background-color: #ffffff;
    }}

    /* ---- Sidebar ---- */
    QFrame#sidebar {{
        background-color: #eaeaef;
        border-right: 1px solid #d5d5da;
    }}

    QPushButton.sidebar-btn {{
        background-color: transparent;
        border: none;
        border-radius: 6px;
        padding: 10px;
        margin: 2px 4px;
        color: #808088;
        font-size: 18px;
    }}
    QPushButton.sidebar-btn:hover {{
        background-color: #d5d5dc;
        color: #505058;
    }}
    QPushButton.sidebar-btn:checked {{
        background-color: {accent_faint};
        color: {accent};
    }}

    /* ---- Cards ---- */
    QFrame.card {{
        background-color: #ffffff;
        border: 1px solid #e0e0e5;
        border-radius: 10px;
    }}
    QFrame.card:hover {{
        border-color: {accent};
    }}

    /* ---- Panels ---- */
    QFrame#left-panel {{
        background-color: #f0f0f5;
        border-right: 1px solid #dddde2;
    }}
    QFrame#right-panel {{
        background-color: #f0f0f5;
        border-left: 1px solid #dddde2;
    }}
    QFrame#graph-container {{
        background-color: #fafafa;
    }}

    /* ---- Labels ---- */
    QLabel {{
        color: #1a1a1f;
        background-color: transparent;
    }}
    QLabel.heading {{
        font-size: 18px;
        font-weight: bold;
        color: #0a0a0f;
    }}
    QLabel.subheading {{
        font-size: 14px;
        font-weight: 600;
        color: #404048;
    }}
    QLabel.muted {{
        color: #909098;
        font-size: 12px;
    }}
    QLabel.accent {{
        color: {accent};
    }}

    /* ---- Buttons ---- */
    QPushButton {{
        background-color: #e8e8ed;
        color: #1a1a1f;
        border: 1px solid #d0d0d8;
        border-radius: 6px;
        padding: 7px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: #dddde4;
        border-color: #c0c0c8;
    }}
    QPushButton:pressed {{
        background-color: {accent};
        color: #ffffff;
        border-color: {accent};
    }}
    QPushButton:disabled {{
        background-color: #f0f0f5;
        color: #b0b0b8;
        border-color: #e0e0e5;
    }}
    QPushButton.primary {{
        background-color: {accent};
        color: #ffffff;
        border: none;
        font-weight: 600;
    }}
    QPushButton.primary:hover {{
        background-color: {accent_hover};
    }}
    QPushButton.primary:pressed {{
        background-color: {accent_pressed};
    }}
    QPushButton.danger {{
        background-color: #fff0f0;
        color: #cc3333;
        border-color: #ffcccc;
    }}
    QPushButton.danger:hover {{
        background-color: #ffe0e0;
        color: #aa2222;
    }}

    /* ---- Inputs ---- */
    QLineEdit {{
        background-color: #ffffff;
        border: 1px solid #d0d0d8;
        border-radius: 6px;
        padding: 7px 10px;
        color: #1a1a1f;
        selection-background-color: {accent};
        selection-color: #ffffff;
    }}
    QLineEdit:focus {{
        border-color: {accent};
    }}

    QTextEdit {{
        background-color: #ffffff;
        border: 1px solid #d0d0d8;
        border-radius: 6px;
        padding: 6px;
        color: #1a1a1f;
    }}

    QSpinBox, QDoubleSpinBox {{
        background-color: #ffffff;
        border: 1px solid #d0d0d8;
        border-radius: 6px;
        padding: 5px;
        color: #1a1a1f;
    }}

    QComboBox {{
        background-color: #ffffff;
        border: 1px solid #d0d0d8;
        border-radius: 6px;
        padding: 6px 10px;
        color: #1a1a1f;
        min-width: 100px;
    }}
    QComboBox:hover {{
        border-color: #b0b0b8;
    }}
    QComboBox:focus {{
        border-color: {accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #ffffff;
        border: 1px solid #d0d0d8;
        selection-background-color: {accent};
        selection-color: #ffffff;
        color: #1a1a1f;
        outline: 0px;
    }}

    QSlider::groove:horizontal {{
        background-color: #d0d0d8;
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background-color: {accent};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {accent};
        border-radius: 2px;
    }}

    QCheckBox {{
        spacing: 8px;
        color: #1a1a1f;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #c0c0c8;
        background-color: #ffffff;
    }}
    QCheckBox::indicator:checked {{
        background-color: {accent};
        border-color: {accent};
    }}

    /* ---- Scroll Areas ---- */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        border: none;
        background-color: transparent;
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background-color: #c0c0c8;
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: #a0a0a8;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
        height: 0px;
    }}
    QScrollBar:horizontal {{
        border: none;
        background-color: transparent;
        height: 8px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: #c0c0c8;
        min-width: 30px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        border: none;
        background: none;
        width: 0px;
    }}

    /* ---- Tab Widget ---- */
    QTabWidget::pane {{
        border: 1px solid #dddde2;
        background-color: #ffffff;
        border-radius: 6px;
    }}
    QTabBar::tab {{
        background-color: #f0f0f5;
        color: #808088;
        border: none;
        padding: 8px 18px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{
        background-color: #ffffff;
        color: {accent};
        border-bottom: 2px solid {accent};
    }}
    QTabBar::tab:hover {{
        background-color: #e8e8ed;
        color: #505058;
    }}

    /* ---- Group Box ---- */
    QGroupBox {{
        border: 1px solid #dddde2;
        border-radius: 8px;
        margin-top: 16px;
        padding-top: 16px;
        font-weight: 600;
        color: #404048;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: {accent};
    }}

    /* ---- List / Table ---- */
    QListWidget {{
        background-color: #ffffff;
        border: 1px solid #dddde2;
        border-radius: 6px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 6px 10px;
        border-radius: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {accent_faint};
        color: {accent};
    }}
    QListWidget::item:hover {{
        background-color: #f0f0f5;
    }}

    QTableWidget {{
        background-color: #ffffff;
        border: 1px solid #dddde2;
        border-radius: 6px;
        gridline-color: #eaeaef;
        outline: none;
    }}
    QTableWidget::item:selected {{
        background-color: {accent_faint};
        color: {accent};
    }}
    QHeaderView::section {{
        background-color: #f5f5f7;
        color: #606068;
        border: none;
        border-bottom: 1px solid #dddde2;
        padding: 6px 10px;
        font-weight: 600;
    }}

    /* ---- Tooltips ---- */
    QToolTip {{
        background-color: #ffffff;
        color: #1a1a1f;
        border: 1px solid #d0d0d8;
        border-radius: 4px;
        padding: 4px 8px;
    }}

    /* ---- Status Bar ---- */
    QStatusBar {{
        background-color: #eaeaef;
        color: #606068;
        border-top: 1px solid #d5d5da;
        font-size: 12px;
    }}

    /* ---- Menu Bar ---- */
    QMenuBar {{
        background-color: #f5f5f7;
        color: #1a1a1f;
        border-bottom: 1px solid #dddde2;
    }}
    QMenuBar::item:selected {{
        background-color: #e0e0e5;
    }}
    QMenu {{
        background-color: #ffffff;
        border: 1px solid #dddde2;
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {accent_faint};
        color: {accent};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: #eaeaef;
        margin: 4px 8px;
    }}

    /* ---- Progress Bar ---- */
    QProgressBar {{
        background-color: #e0e0e5;
        border: none;
        border-radius: 4px;
        height: 6px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {accent};
        border-radius: 4px;
    }}

    /* ---- Splitter ---- */
    QSplitter::handle {{
        background-color: #dddde2;
        width: 1px;
        height: 1px;
    }}

    /* ---- Toolbar ---- */
    QToolBar {{
        background-color: #f5f5f7;
        border: none;
        spacing: 4px;
        padding: 2px;
    }}
    """


def apply_theme(app, mode: str = DARK, accent: str = ACCENT_BLUE):
    """
    Apply a Vortex theme to a QApplication instance.

    Args:
        app: QApplication instance
        mode: 'dark' or 'light'
        accent: Hex color string for the accent color
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    if mode == LIGHT:
        stylesheet = get_light_stylesheet(accent)
    else:
        stylesheet = get_dark_stylesheet(accent)

    app.setStyleSheet(stylesheet)


def get_matplotlib_style(mode: str = DARK) -> dict:
    """
    Get matplotlib rcParams dict that matches the Vortex theme.

    Usage:
        style = get_matplotlib_style('dark')
        plt.rcParams.update(style)
        # or
        with plt.style.context(style):
            fig, ax = plt.subplots()
    """
    if mode == DARK:
        return {
            'figure.facecolor': '#1a1a20',
            'axes.facecolor': '#22222a',
            'axes.edgecolor': '#3a3a42',
            'axes.labelcolor': '#c0c0c8',
            'text.color': '#e0e0e0',
            'xtick.color': '#808088',
            'ytick.color': '#808088',
            'grid.color': '#2e2e36',
            'grid.alpha': 0.5,
            'legend.facecolor': '#22222a',
            'legend.edgecolor': '#3a3a42',
            'legend.labelcolor': '#e0e0e0',
            'figure.edgecolor': '#1a1a20',
            'savefig.facecolor': '#1a1a20',
            'savefig.edgecolor': '#1a1a20',
        }
    else:
        return {
            'figure.facecolor': '#fafafa',
            'axes.facecolor': '#ffffff',
            'axes.edgecolor': '#d0d0d8',
            'axes.labelcolor': '#404048',
            'text.color': '#1a1a1f',
            'xtick.color': '#606068',
            'ytick.color': '#606068',
            'grid.color': '#eaeaef',
            'grid.alpha': 0.8,
            'legend.facecolor': '#ffffff',
            'legend.edgecolor': '#d0d0d8',
            'legend.labelcolor': '#1a1a1f',
            'figure.edgecolor': '#fafafa',
            'savefig.facecolor': '#fafafa',
            'savefig.edgecolor': '#fafafa',
        }
