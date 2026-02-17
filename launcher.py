import sys
import os
import json
import time
import subprocess
import ctypes
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QProgressBar, 
                             QFrame, QGridLayout, QScrollArea, QMessageBox, QLineEdit, QSizePolicy)

# Extension system (for displaying counts)
try:
    from extensions.manager import ExtensionManager
    HAS_EXTENSIONS = True
except ImportError:
    HAS_EXTENSIONS = False

# Launcher Configuration
LAUNCHER_VERSION = "1.1.0"
CONFIG_FILE = "launcher_state.json"
LAUNCHER_UPDATE_FILE = "launcher_update.py"  # New launcher version will be saved here
APPS = [
    {
        "id": "gui_sim",
        "name": "HexaKinetic",
        "desc": "State of the art 6DOF rocket simulation and config interface.",
        "version": "0.5.2",
        "exec": [sys.executable, "gui.py"],
        "icon": "hexakinetic_icon.png"
    },
    {
        "id": "cli_sim",
        "name": "TextKinetic",
        "desc": "Lightweight, terminal-based simulation interface based on HexaKinetic.",
        "version": "0.4.7",
        "exec": ["powershell", "-NoExit", "-Command", f"& '{sys.executable}' cli.py"],
        "icon": "textkinetic_icon.png"
    },
    {
        "id": "visualizer",
        "name": "HexaVisual",
        "desc": "3D Flight visualization and analysis tool.",
        "version": "0.3.1",
        "exec": [sys.executable, "advanced_visualizer.py"],
        "icon": "hexavisual_icon.png"
    },
    {
        "id": "plotvisual",
        "name": "PlotVisual",
        "desc": "Advanced plotting and analysis for simulation results. Extensions, theming, and more.",
        "version": "2.0.0",
        "exec": [sys.executable, "PlotVisual.py"],
        "icon": "hexavisual_icon.png"
    },
    {
        "id": "extension_manager",
        "name": "Vortex Extension Manager",
        "desc": "Manage and discover extensions across all Vortex desktop apps.",
        "version": "1.0.0",
        "exec": [sys.executable, "vortex_extension_manager.py"],
        "icon": "vortex_icon.png"
    },
    {
        "id": "control",
        "name": "Vortex Mission Control",
        "desc": "Direct hardware link and mission management.",
        "version": "0.19.1",
        "exec": [sys.executable, "vortex-control.py"],
        "icon": "missioncontrol_icon.png"
    }
]

class AppCard(QFrame):
    def __init__(self, app_info, launcher):
        super().__init__()
        self.app_info = app_info
        self.launcher = launcher
        # Make cards vertical-list friendly: flexible width, fixed height
        self.setMinimumHeight(130)
        self.setMaximumWidth(420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #2a2a2e;
                border: 1px solid #3e3e42;
                border-radius: 12px;
            }
            #card:hover {
                border: 1px solid #007acc;
            }
            QLabel { color: #ffffff; }
            QPushButton { 
                border-radius: 6px; 
                padding: 8px; 
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(self)
        
        # Icon
        icon_label = QLabel()
        icon_path = os.path.join("assets", app_info["icon"])
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pix)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # Title
        title = QLabel(app_info["name"])
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Optional badge (e.g. Legacy, Beta)
        self.badge_label = QLabel()
        self.badge_label.setAlignment(Qt.AlignCenter)
        self.badge_label.setStyleSheet("background-color: #ffcc00; color: #222; border-radius: 8px; padding: 2px 6px; font-size: 10px;")
        self.badge_label.hide()
        layout.addWidget(self.badge_label) 

        # Description
        desc = QLabel(app_info["desc"])
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(desc)

        # Version Label
        self.version_label = QLabel()
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setStyleSheet("color: #666666; font-size: 10px; padding-top: 5px;")
        layout.addWidget(self.version_label)

        layout.addStretch()

        # Progress Bar (for "Installing")
        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(6)
        self.pbar.setTextVisible(False)
        self.pbar.hide()
        layout.addWidget(self.pbar)

        # Buttons
        self.btn_action = QPushButton()
        self.btn_action.clicked.connect(self.handle_action)
        layout.addWidget(self.btn_action)

        self.btn_uninstall = QPushButton("Uninstall")
        self.btn_uninstall.setStyleSheet("background-color: transparent; color: #ff5555; border: 1px solid #ff5555;")
        self.btn_uninstall.clicked.connect(self.uninstall)
        layout.addWidget(self.btn_uninstall)
        
        self.update_ui()

    def update_ui(self):
        # Badge (optional)
        badge_text = self.app_info.get('badge')
        if badge_text:
            self.badge_label.setText(badge_text.upper())
            self.badge_label.show()
        else:
            self.badge_label.hide()

        versions = self.launcher.state.get("installed_versions", {})
        installed_version = versions.get(self.app_info["id"])
        target_version = self.app_info["version"]
        
        is_installed = self.app_info["id"] in versions
        has_update = is_installed and installed_version != target_version
        
        self.btn_uninstall.setVisible(is_installed)
        
        if is_installed:
            self.version_label.setText(f"Installed: v{installed_version}")
        else:
            self.version_label.setText("Not Installed")

        if has_update:
            self.btn_action.setText(f"Update to v{target_version}")
            self.btn_action.setStyleSheet("background-color: #f1c40f; color: black;")
        elif is_installed:
            self.btn_action.setText("Launch")
            self.btn_action.setStyleSheet("background-color: #007acc; color: white;")
        else:
            self.btn_action.setText("Install App")
            self.btn_action.setStyleSheet("background-color: #3e3e42; color: #ffffff;")

    def handle_action(self):
        versions = self.launcher.state.get("installed_versions", {})
        installed_version = versions.get(self.app_info["id"])
        target_version = self.app_info["version"]
        
        is_installed = self.app_info["id"] in versions
        has_update = is_installed and installed_version != target_version
        
        if has_update:
            self.install(update=True)
        elif is_installed:
            # Launch
            try:
                subprocess.Popen(self.app_info["exec"])
            except Exception as e:
                QMessageBox.critical(self, "Launch Error", f"Failed to start: {e}")
        else:
            # Install (Fake)
            self.install()

    def install(self, update=False):
        self.btn_action.setEnabled(False)
        self.btn_action.setText("Updating..." if update else "Installing...")
        self.pbar.show()
        self.pbar.setValue(0)
        
        self.install_step = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_install)
        self.timer.start(30 if update else 50) # Updating is a bit faster

    def update_install(self):
        self.install_step += 2
        self.pbar.setValue(self.install_step)
        if self.install_step >= 100:
            self.timer.stop()
            self.pbar.hide()
            self.btn_action.setEnabled(True)
            self.launcher.mark_installed(self.app_info["id"], self.app_info["version"])
            self.update_ui()

    def uninstall(self):
        self.launcher.mark_uninstalled(self.app_info["id"])
        self.update_ui()

class VortexLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vortex Desktop Launcher")
        self.resize(1000, 500)
        # The global stylesheet set in the main block handles the theme
        
        # Load State
        self.load_state()

        # Premium Icon
        icon_path = os.path.join('assets', 'vortex_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header
        header = QLabel("VORTEX DESKTOP")
        header.setFont(QFont("Segoe UI Light", 24))
        header.setStyleSheet("color: #007acc; letter-spacing: 4px; padding: 20px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Launcher Update Banner (initially hidden)
        self.update_banner = QFrame()
        self.update_banner.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.update_banner.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f39c12, stop:1 #e67e22);
                border-radius: 8px;
                padding: 12px;
                margin: 10px 20px;
            }
            QLabel { color: white; font-weight: bold; }
            QPushButton {
                background-color: white;
                color: #e67e22;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        banner_layout = QHBoxLayout(self.update_banner)
        
        self.update_banner_label = QLabel()
        self.update_banner_label.setFont(QFont("Segoe UI", 11))
        banner_layout.addWidget(self.update_banner_label)
        banner_layout.addStretch()
        
        self.update_banner_btn = QPushButton("Update Now")
        self.update_banner_btn.clicked.connect(self.install_launcher_update)
        banner_layout.addWidget(self.update_banner_btn)
        
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(lambda: self.update_banner.hide())
        banner_layout.addWidget(dismiss_btn)
        
        self.update_banner.hide()  # Hidden by default
        main_layout.addWidget(self.update_banner)
        
        # Check for launcher updates
        self.check_and_show_launcher_update()


        # Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search apps...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 20px;
                padding: 10px 20px;
                color: white;
                font-size: 14px;
                margin: 0px 20px;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
            }
        """)
        self.search_input.textChanged.connect(self.filter_apps)
        search_layout.addWidget(self.search_input)
        main_layout.addLayout(search_layout)

        # App grid inside a vertical scroll area (responsive wrapping grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(16)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        
        self.app_cards = []  # list of tuples (app_dict, card_widget)
        for i, app in enumerate(APPS):
            card = AppCard(app, self)
            self.app_cards.append((app, card))
            # Initial placement (will be reflowed on show/resize)
            grid_layout.addWidget(card, i // 3, i % 3)

        scroll.setWidget(grid_widget)
        main_layout.addWidget(scroll)

        # Keep references for dynamic layout
        self.grid_widget = grid_widget
        self.grid_layout = grid_layout
        self.scroll = scroll
        
        # Perform an initial layout pass
        self.layout_app_cards()

        # Show more apps toggle and hidden container
        self.show_more_btn = QPushButton("Show more apps ▾")
        self.show_more_btn.setCheckable(True)
        self.show_more_btn.setStyleSheet("""
            QPushButton {
                background-color: #252526;
                color: #007acc;
                border: 1px solid #3e3e42;
                border-radius: 0px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 5px;
            }
            QPushButton:hover {
                background-color: #2d2d30;
                color: #3498db;
            }
            QPushButton:checked {
                background-color: #2d2d30;
                color: #ffffff;
            }
        """)
        self.show_more_btn.toggled.connect(self.toggle_more_apps)
        main_layout.addWidget(self.show_more_btn)

        self.more_container = QWidget()
        self.more_container.setVisible(False)
        more_layout = QVBoxLayout(self.more_container)
        more_layout.setSpacing(8)

        self.more_app_cards = []

        retro_app = {
            "id": "retro_visualizer",
            "name": "Retro Visualizer",
            "desc": "The legacy 3D visualization software. For the most features, use HexaVisual.",
            "version": "0.1.0",
            "exec": [sys.executable, "visualization.py"],
            "icon": "hexavisual_icon.png",
            "badge": "Legacy"
        }
        retro_card = AppCard(retro_app, self)
        self.more_app_cards.append((retro_app, retro_card))
        more_layout.addWidget(retro_card) 

        plotvisual_legacy_app = {
            "id": "plotvisual_legacy",
            "name": "PlotVisual (Legacy)",
            "desc": "Original visualization tool. Use the new PlotVisual for extensions and theming.",
            "version": "1.0.0",
            "exec": [sys.executable, "PlotVisual_legacy.py"],
            "icon": "hexavisual_icon.png",
            "badge": "Legacy"
        }
        legacy_pv_card = AppCard(plotvisual_legacy_app, self)
        self.more_app_cards.append((plotvisual_legacy_app, legacy_pv_card))
        more_layout.addWidget(legacy_pv_card)



        main_layout.addWidget(self.more_container)

        # Footer with version info and update check
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3e3e42;
                padding: 10px;
            }
            QLabel { color: #888888; font-size: 10px; }
            QPushButton {
                background-color: transparent;
                color: #007acc;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #007acc;
                color: white;
            }
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 5, 20, 5)
        
        version_label = QLabel(f"Vortex Desktop Launcher v{LAUNCHER_VERSION}")
        footer_layout.addWidget(version_label)

        # Extension count
        ext_count_text = ""
        if HAS_EXTENSIONS:
            try:
                _mgr = ExtensionManager()
                _ext_list = _mgr.list_installed()
                ext_count_text = f"  |  \U0001F9E9 {len(_ext_list)} extension(s)"
            except Exception:
                ext_count_text = ""
        ext_label = QLabel(ext_count_text)
        ext_label.setStyleSheet("color: #007acc; font-size: 10px;")
        footer_layout.addWidget(ext_label)

        footer_layout.addStretch()
        
        check_update_btn = QPushButton("Check for Updates")
        check_update_btn.clicked.connect(self.manual_update_check)
        footer_layout.addWidget(check_update_btn)
        
        main_layout.addWidget(footer)


    def load_state(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.state = json.load(f)
            # Migration path: if old 'installed' list exists, convert it
            if "installed" in self.state and "installed_versions" not in self.state:
                self.state["installed_versions"] = {app_id: "0.0.0" for app_id in self.state["installed"]}
        else:
            self.state = {"installed_versions": {}}
    
    def check_launcher_update(self):
        """Check if a launcher update is available by comparing versions."""
        # In a real implementation, this would check a remote server or GitHub
        # For now, we'll simulate by checking if there's a newer version in state
        latest_version = self.state.get("launcher_latest_version", LAUNCHER_VERSION)
        return self._compare_versions(latest_version, LAUNCHER_VERSION) > 0, latest_version
    
    def _compare_versions(self, v1, v2):
        """Compare two version strings. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
        try:
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]
            # Pad to same length
            while len(parts1) < len(parts2):
                parts1.append(0)
            while len(parts2) < len(parts1):
                parts2.append(0)
            for p1, p2 in zip(parts1, parts2):
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1
            return 0
        except:
            return 0
    
    def download_launcher_update(self, version):
        """Simulate downloading a launcher update. In production, this would fetch from a server."""
        # For demonstration, we'll create a mock update file
        # In reality, this would download from GitHub releases or similar
        return True
    
    def apply_launcher_update(self):
        """Apply the launcher update by replacing the current launcher file."""
        try:
            if os.path.exists(LAUNCHER_UPDATE_FILE):
                # Create a batch script to replace the launcher after it closes
                batch_script = f"""@echo off
timeout /t 2 /nobreak > nul
copy /Y "{LAUNCHER_UPDATE_FILE}" "launcher.py"
del "{LAUNCHER_UPDATE_FILE}"
start "" "{sys.executable}" "launcher.py"
"""
                with open("_update_launcher.bat", "w") as f:
                    f.write(batch_script)
                
                # Launch the batch script and exit
                subprocess.Popen(["cmd", "/c", "_update_launcher.bat"], 
                               creationflags=subprocess.CREATE_NO_WINDOW)
                QApplication.quit()
                return True
        except Exception as e:
            QMessageBox.critical(self, "Update Error", f"Failed to apply update: {e}")
            return False

    def check_and_show_launcher_update(self):
        """Check for launcher updates and show banner if available."""
        has_update, latest_version = self.check_launcher_update()
        if has_update:
            self.update_banner_label.setText(
                f"🚀 Launcher Update Available: v{LAUNCHER_VERSION} → v{latest_version}"
            )
            self.update_banner.show()
    
    def install_launcher_update(self):
        """Handle the launcher update installation process."""
        has_update, latest_version = self.check_launcher_update()
        if not has_update:
            QMessageBox.information(self, "No Update", "You're already on the latest version!")
            return
        
        # Disable button during download
        self.update_banner_btn.setEnabled(False)
        self.update_banner_btn.setText("Downloading...")
        
        # Simulate download (in production, this would be a real download)
        QTimer.singleShot(1500, lambda: self._finish_launcher_update(latest_version))
    
    def _finish_launcher_update(self, version):
        """Complete the launcher update process."""
        success = self.download_launcher_update(version)
        
        if success:
            reply = QMessageBox.question(
                self, 
                "Update Ready", 
                f"Launcher v{version} is ready to install.\n\n"
                "The launcher will restart to apply the update.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # For demonstration, we'll just show a message
                # In production, this would call apply_launcher_update()
                QMessageBox.information(
                    self,
                    "Update Complete",
                    f"Launcher updated to v{version}!\n\n"
                    "(In production, the launcher would restart automatically)"
                )
                # Update the state to reflect new version
                self.state["launcher_latest_version"] = LAUNCHER_VERSION
                self.save_state()
                self.update_banner.hide()
        else:
            QMessageBox.critical(self, "Update Failed", "Failed to download the update.")
        
        self.update_banner_btn.setEnabled(True)
        self.update_banner_btn.setText("Update Now")

    def manual_update_check(self):
        """Manually check for launcher updates."""
        has_update, latest_version = self.check_launcher_update()
        if has_update:
            self.update_banner_label.setText(
                f"🚀 Launcher Update Available: v{LAUNCHER_VERSION} → v{latest_version}"
            )
            self.update_banner.show()
            QMessageBox.information(
                self,
                "Update Available",
                f"A new version (v{latest_version}) is available!\n\n"
                "Click 'Update Now' in the banner above to install."
            )
        else:
            QMessageBox.information(
                self,
                "No Updates",
                f"You're running the latest version (v{LAUNCHER_VERSION})!"
            )


    def layout_app_cards(self):
        """Arrange visible app cards in a wrapping grid based on available width."""
        if not hasattr(self, 'grid_layout'):
            return
        # Remove all items from layout (keep widgets alive)
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            # Note: we don't delete the widget; just clear positions
        
        # Compute available width inside the scroll viewport
        viewport_width = self.scroll.viewport().width() if hasattr(self, 'scroll') else self.width()
        # Assume an approximate card width; use card max width as guideline
        card_target_w = 360
        spacing = self.grid_layout.spacing() or 16
        # Compute number of columns that fit (at least 1)
        columns = max(1, max(1, int(viewport_width / (card_target_w + spacing))))

        visible_cards = [card for app, card in self.app_cards if card.isVisible()]

        for idx, card in enumerate(visible_cards):
            r = idx // columns
            c = idx % columns
            self.grid_layout.addWidget(card, r, c)

        # If the 'more' container is visible, ensure it's below grid by adding a stretch
        # (Grid will naturally size; we don't need an explicit stretch here)

    def resizeEvent(self, event):
        # Reflow cards on resize
        try:
            self.layout_app_cards()
        except Exception:
            pass
        return super().resizeEvent(event)

    def save_state(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.state, f)

    def mark_installed(self, app_id, version):
        if "installed_versions" not in self.state:
            self.state["installed_versions"] = {}
        self.state["installed_versions"][app_id] = version
        self.save_state()

    def mark_uninstalled(self, app_id):
        if "installed_versions" in self.state and app_id in self.state["installed_versions"]:
            del self.state["installed_versions"][app_id]
            self.save_state()

    def filter_apps(self, text: str):
        """Filter visible app cards by search text (name, desc, id)."""
        t = text.strip().lower()
        for app, card in getattr(self, 'app_cards', []):
            visible = (t == '') or (t in app.get('name', '').lower()) or (t in app.get('desc', '').lower()) or (t in app.get('id', '').lower())
            card.setVisible(visible)

        more_has_match = False
        for app, card in getattr(self, 'more_app_cards', []):
            visible = (t == '') or (t in app.get('name', '').lower()) or (t in app.get('desc', '').lower()) or (t in app.get('id', '').lower())
            card.setVisible(visible)
            more_has_match = more_has_match or visible

        if t:
            self.more_container.setVisible(more_has_match)
            self.show_more_btn.setChecked(more_has_match)
            self.show_more_btn.setText("Hide more apps ▴" if more_has_match else "Show more apps ▾")
        else:
            self.more_container.setVisible(self.show_more_btn.isChecked())

        # Reflow after changing visibility
        try:
            self.layout_app_cards()
        except Exception:
            pass

    def toggle_more_apps(self, checked: bool):
        """Show or hide the 'more apps' container."""
        self.more_container.setVisible(checked)
        self.show_more_btn.setText("Hide more apps ▴" if checked else "Show more apps ▾")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('assets/vortex_icon.png'))
    # Professional dark style
    app.setStyle("Fusion")
    
    # Global Dark Theme for Dialogs and Generic Widgets
    app.setStyleSheet("""
        QWidget {
            color: #ffffff;
            font-family: 'Segoe UI', Arial;
        }
        QMainWindow, QDialog, QMessageBox {
            background-color: #1e1e1e;
        }
        QLabel {
            color: #ffffff;
        }
        QPushButton {
            background-color: #3e3e42;
            color: #ffffff;
            border: 1px solid #4d4d50;
            border-radius: 4px;
            padding: 6px 12px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #505054;
        }
        QPushButton:pressed {
            background-color: #007acc;
        }
        QLineEdit {
            background-color: #2d2d30;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            padding: 6px;
            color: #ffffff;
        }
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            border: none;
            background-color: #1e1e1e;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #3e3e42;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
    """)
    
    launcher = VortexLauncher()
    launcher.show()
    sys.exit(app.exec_())
