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
                             QFrame, QGridLayout, QScrollArea, QMessageBox)

# Launcher Configuration
CONFIG_FILE = "launcher_state.json"
APPS = [
    {
        "id": "gui_sim",
        "name": "HexaKinetic",
        "desc": "State of the art 6DOF rocket simulation and config interface.",
        "version": "0.4.7",
        "exec": [sys.executable, "gui.py"],
        "icon": "hexakinetic_icon.png"
    },
    {
        "id": "cli_sim",
        "name": "TextKinetic",
        "desc": "Lightweight, terminal-based simulation interface based on HexaKinetic.",
        "version": "0.4.5",
        "exec": ["powershell", "-NoExit", "-Command", f"& '{sys.executable}' cli.py"],
        "icon": "textkinetic_icon.png"
    },
    {
        "id": "visualizer",
        "name": "HexaVisual",
        "desc": "3D Flight visualization and analysis tool.",
        "version": "0.1.9",
        "exec": [sys.executable, "advanced_visualizer.py"],
        "icon": "hexavisual_icon.png"
    },
    {
        "id": "control",
        "name": "Vortex Mission Control",
        "desc": "Direct hardware link and mission management.",
        "version": "0.18.2",
        "exec": [sys.executable, "vortex-control.py"],
        "icon": "missioncontrol_icon.png"
    }
]

class AppCard(QFrame):
    def __init__(self, app_info, launcher):
        super().__init__()
        self.app_info = app_info
        self.launcher = launcher
        self.setFixedSize(280, 320)
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
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Load State
        self.load_state()

        # Premium Icon
        icon_path = os.path.join('assets', 'vortex_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            myappid = 'vortex.desktop.launcher.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

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

        # App Grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        grid_widget = QWidget()
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setSpacing(20)
        grid_layout.setAlignment(Qt.AlignCenter)
        
        for app in APPS:
            card = AppCard(app, self)
            grid_layout.addWidget(card)

        scroll.setWidget(grid_widget)
        main_layout.addWidget(scroll)

    def load_state(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.state = json.load(f)
            # Migration path: if old 'installed' list exists, convert it
            if "installed" in self.state and "installed_versions" not in self.state:
                self.state["installed_versions"] = {app_id: "0.0.0" for app_id in self.state["installed"]}
        else:
            self.state = {"installed_versions": {}}

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

if __name__ == '__main__':
    # Fix Taskbar Icon on Windows
    launcher_id = 'co.mishra.rockets.vortex.launcher' 
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(launcher_id)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('assets/vortex_icon.png'))
    # Professional dark style
    app.setStyle("Fusion")
    
    launcher = VortexLauncher()
    launcher.show()
    sys.exit(app.exec_())
