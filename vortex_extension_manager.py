"""
Vortex Extension Manager
========================

Standalone extension manager + marketplace for all Vortex Desktop apps.
"""

import os
import sys
import webbrowser

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
    QScrollArea,
    QFileDialog,
    QTabWidget,
    QComboBox,
    QLineEdit,
)

from extensions.manager import ExtensionManager
from extensions.catalog import ExtensionCatalog
from extensions.packaging import pack_extension


APP_TITLE = "Vortex Extension Manager"
APP_VERSION = "1.0.0"


class VortexExtensionManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} v{APP_VERSION}")
        self.resize(1200, 780)

        icon_path = os.path.join(os.path.dirname(__file__), "assets", "vortex_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.manager = ExtensionManager()
        self.catalog = ExtensionCatalog()

        self._build_ui()
        self.refresh_all()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        subtitle = QLabel("Manage and discover extensions for PlotVisual, HexaKinetic, HexaVisual, and Mission Control")
        subtitle.setStyleSheet("color: #555555;")

        title_col = QVBoxLayout()
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #007acc; font-weight: bold;")
        header.addWidget(self.stats_label)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_all)
        header.addWidget(refresh_btn)

        root.addLayout(header)

        tools = QFrame()
        tools.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e6e6e6; border-radius: 10px; }")
        tools_layout = QHBoxLayout(tools)
        tools_layout.setContentsMargins(12, 10, 12, 10)

        install_file_btn = QPushButton("Install from .vortexext")
        install_file_btn.clicked.connect(self.install_from_file)
        tools_layout.addWidget(install_file_btn)

        install_dir_btn = QPushButton("Install from Directory")
        install_dir_btn.clicked.connect(self.install_from_directory)
        tools_layout.addWidget(install_dir_btn)

        pack_btn = QPushButton("Pack Extension")
        pack_btn.clicked.connect(self.pack_extension_dialog)
        tools_layout.addWidget(pack_btn)

        tools_layout.addStretch()
        root.addWidget(tools)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.installed_tab = QWidget()
        self.market_tab = QWidget()
        self.tabs.addTab(self.installed_tab, "Installed")
        self.tabs.addTab(self.market_tab, "Marketplace")

        self._build_installed_tab()
        self._build_market_tab()

    def _build_installed_tab(self):
        layout = QVBoxLayout(self.installed_tab)
        layout.setContentsMargins(6, 8, 6, 6)

        self.installed_info = QLabel("")
        self.installed_info.setStyleSheet("color: #555555;")
        layout.addWidget(self.installed_info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.installed_container = QWidget()
        self.installed_layout = QVBoxLayout(self.installed_container)
        self.installed_layout.setSpacing(8)
        self.installed_layout.addStretch()
        scroll.setWidget(self.installed_container)
        layout.addWidget(scroll, stretch=1)

    def _build_market_tab(self):
        layout = QVBoxLayout(self.market_tab)
        layout.setContentsMargins(6, 8, 6, 6)

        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Type:"))
        self.market_type = QComboBox()
        self.market_type.addItems(["all", "plotvisual", "hexakinetic", "hexavisual", "mission_control", "universal"])
        self.market_type.currentTextChanged.connect(self.refresh_market)
        filter_row.addWidget(self.market_type)

        filter_row.addWidget(QLabel("Search:"))
        self.market_search = QLineEdit()
        self.market_search.setPlaceholderText("name, author, description, tags...")
        self.market_search.textChanged.connect(self.refresh_market)
        filter_row.addWidget(self.market_search, stretch=1)

        featured_btn = QPushButton("Show Featured")
        featured_btn.clicked.connect(self.show_featured)
        filter_row.addWidget(featured_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_market_filters)
        filter_row.addWidget(clear_btn)

        layout.addLayout(filter_row)

        self.market_info = QLabel("")
        self.market_info.setStyleSheet("color: #555555;")
        layout.addWidget(self.market_info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.market_container = QWidget()
        self.market_layout = QVBoxLayout(self.market_container)
        self.market_layout.setSpacing(8)
        self.market_layout.addStretch()
        scroll.setWidget(self.market_container)
        layout.addWidget(scroll, stretch=1)

    def refresh_all(self):
        self.manager.discover()
        self.refresh_installed()
        self.refresh_market()
        self._update_stats()

    def _update_stats(self):
        manifests = self.manager.list_installed()
        enabled = [m for m in manifests if self.manager.is_enabled(m.id)]
        by_type = {
            "plotvisual": len([m for m in manifests if m.extension_type == "plotvisual"]),
            "hexakinetic": len([m for m in manifests if m.extension_type == "hexakinetic"]),
            "hexavisual": len([m for m in manifests if m.extension_type == "hexavisual"]),
            "mission_control": len([m for m in manifests if m.extension_type == "mission_control"]),
            "universal": len([m for m in manifests if m.extension_type == "universal"]),
        }
        self.stats_label.setText(
            f"Installed: {len(manifests)}  |  Enabled: {len(enabled)}  |  "
            f"PV:{by_type['plotvisual']} HK:{by_type['hexakinetic']} HV:{by_type['hexavisual']} MC:{by_type['mission_control']} U:{by_type['universal']}"
        )

    def refresh_installed(self):
        while self.installed_layout.count() > 1:
            item = self.installed_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        manifests = sorted(self.manager.list_installed(), key=lambda m: (m.extension_type, m.name.lower()))
        self.installed_info.setText(f"{len(manifests)} installed extension(s)")

        if not manifests:
            empty = QLabel("No extensions installed yet.")
            empty.setStyleSheet("color: #555555;")
            self.installed_layout.insertWidget(0, empty)
            return

        for manifest in manifests:
            self.installed_layout.insertWidget(self.installed_layout.count() - 1, self._build_installed_card(manifest))

    def _build_installed_card(self, manifest):
        card = QFrame()
        card.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e6e6e6; border-radius: 10px; }")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 10, 12, 10)

        info = QVBoxLayout()
        title = QLabel(f"{manifest.name}  v{manifest.version}")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        info.addWidget(title)

        desc_text = manifest.description if manifest.description else "No description"
        desc = QLabel(desc_text)
        desc.setStyleSheet("color: #333333;")
        desc.setWordWrap(True)
        info.addWidget(desc)

        meta = QLabel(
            f"ID: {manifest.id}  |  Type: {manifest.extension_type}  |  "
            f"Author: {manifest.author or 'Unknown'}  |  {'Bundled' if manifest.bundled else 'Installed'}"
        )
        meta.setStyleSheet("color: #666666; font-size: 11px;")
        info.addWidget(meta)
        row.addLayout(info, stretch=1)

        actions = QVBoxLayout()
        is_enabled = self.manager.is_enabled(manifest.id)
        state_label = QLabel("Enabled" if is_enabled else "Disabled")
        state_label.setStyleSheet(f"color: {'#4caf50' if is_enabled else '#ff6b6b'}; font-weight: bold;")
        actions.addWidget(state_label, alignment=Qt.AlignRight)

        toggle_btn = QPushButton("Disable" if is_enabled else "Enable")
        toggle_btn.clicked.connect(lambda _=False, ext_id=manifest.id: self.toggle_enabled(ext_id))
        actions.addWidget(toggle_btn)

        uninstall_btn = QPushButton("Uninstall")
        uninstall_btn.setEnabled(not manifest.bundled)
        uninstall_btn.clicked.connect(lambda _=False, ext_id=manifest.id, name=manifest.name: self.uninstall_extension(ext_id, name))
        actions.addWidget(uninstall_btn)

        row.addLayout(actions)
        return card

    def refresh_market(self):
        while self.market_layout.count() > 1:
            item = self.market_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        entries = self.catalog.get_all()
        selected_type = self.market_type.currentText().strip().lower()
        query = self.market_search.text().strip().lower()

        manifests = {m.id for m in self.manager.list_installed()}

        def match(entry):
            if selected_type != "all" and entry.extension_type.lower() != selected_type:
                return False
            if not query:
                return True
            hay = " ".join([
                entry.id,
                entry.name,
                entry.author,
                entry.description,
                " ".join(entry.tags),
            ]).lower()
            return query in hay

        filtered = [e for e in entries if match(e)]
        self.market_info.setText(f"{len(filtered)} marketplace extension(s)")

        if not filtered:
            empty = QLabel("No marketplace entries match your filter.")
            empty.setStyleSheet("color: #555555;")
            self.market_layout.insertWidget(0, empty)
            return

        for entry in filtered:
            self.market_layout.insertWidget(self.market_layout.count() - 1, self._build_market_card(entry, entry.id in manifests))

    def _build_market_card(self, entry, is_installed):
        card = QFrame()
        card.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e6e6e6; border-radius: 10px; }")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 10, 12, 10)

        info = QVBoxLayout()
        title = QLabel(f"{entry.name}  v{entry.version}")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        info.addWidget(title)

        desc = QLabel(entry.description or "No description")
        desc.setStyleSheet("color: #333333;")
        desc.setWordWrap(True)
        info.addWidget(desc)

        meta = QLabel(
            f"Type: {entry.extension_type}  |  Author: {entry.author or 'Unknown'}  |  "
            f"Category: {entry.category}"
        )
        meta.setStyleSheet("color: #666666; font-size: 11px;")
        info.addWidget(meta)

        if entry.featured:
            featured = QLabel("Featured")
            featured.setStyleSheet("color: #f1c40f; font-weight: bold;")
            info.addWidget(featured)

        row.addLayout(info, stretch=1)

        actions = QVBoxLayout()
        if is_installed:
            installed = QLabel("Installed")
            installed.setStyleSheet("color: #4caf50; font-weight: bold;")
            actions.addWidget(installed)
        elif entry.bundled:
            bundled = QLabel("Bundled")
            bundled.setStyleSheet("color: #4caf50; font-weight: bold;")
            actions.addWidget(bundled)
        else:
            action_btn = QPushButton("Download" if entry.download_url else "Install")
            def _market_action(url=entry.download_url):
                if url:
                    # Open the remote download page
                    self.open_download(url)
                    QMessageBox.information(self, "Download", "Opened download URL in your browser. After downloading the .vortexext file, use 'Install from .vortexext' (top toolbar) to install it.")
                else:
                    # Prompt user to pick a local .vortexext to install
                    path, _ = QFileDialog.getOpenFileName(self, "Install Extension", "", "Vortex Extension (*.vortexext);;Zip files (*.zip);;All files (*.*)")
                    if path:
                        try:
                            manifest = self.manager.install_from_vortexext(path)
                            QMessageBox.information(self, "Installed", f"Installed: {manifest.name} v{manifest.version}")
                            self.refresh_all()
                        except Exception as exc:
                            QMessageBox.critical(self, "Install Failed", str(exc))
            action_btn.clicked.connect(_market_action)
            actions.addWidget(action_btn) 

        row.addLayout(actions)
        return card

    def toggle_enabled(self, ext_id):
        if self.manager.is_enabled(ext_id):
            self.manager.disable(ext_id)
        else:
            self.manager.enable(ext_id)
        self.refresh_all()

    def uninstall_extension(self, ext_id, ext_name):
        response = QMessageBox.question(
            self,
            "Confirm Uninstall",
            f"Uninstall '{ext_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        if self.manager.uninstall(ext_id):
            QMessageBox.information(self, "Uninstalled", f"'{ext_name}' was uninstalled.")
            self.refresh_all()
        else:
            QMessageBox.warning(self, "Uninstall Failed", "This extension could not be uninstalled.")

    def install_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Install Extension from File",
            "",
            "Vortex Extension (*.vortexext);;Zip files (*.zip);;All files (*.*)",
        )
        if not path:
            return

        try:
            manifest = self.manager.install_from_vortexext(path)
            QMessageBox.information(self, "Installed", f"Installed: {manifest.name} v{manifest.version}")
            self.refresh_all()
        except Exception as exc:
            QMessageBox.critical(self, "Install Failed", str(exc))

    def install_from_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Install Extension from Directory")
        if not folder:
            return

        try:
            manifest = self.manager.install_from_directory(folder)
            QMessageBox.information(self, "Installed", f"Installed: {manifest.name} v{manifest.version}")
            self.refresh_all()
        except Exception as exc:
            QMessageBox.critical(self, "Install Failed", str(exc))

    def pack_extension_dialog(self):
        source_dir = QFileDialog.getExistingDirectory(self, "Select Extension Directory to Pack")
        if not source_dir:
            return

        default_name = os.path.basename(source_dir.rstrip("/\\")) + ".vortexext"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Package",
            default_name,
            "Vortex Extension (*.vortexext)",
        )
        if not output_path:
            return

        try:
            output = pack_extension(source_dir, output_path)
            QMessageBox.information(self, "Pack Complete", f"Created package:\n{output}")
        except Exception as exc:
            QMessageBox.critical(self, "Pack Failed", str(exc))

    def open_download(self, url):
        if not url:
            QMessageBox.information(self, "No Download URL", "This extension does not have a download URL in the catalog.")
            return
        webbrowser.open(url)

    def show_featured(self):
        self.market_type.setCurrentText("all")
        self.market_search.setText("")

        while self.market_layout.count() > 1:
            item = self.market_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        entries = self.catalog.get_featured()
        manifests = {m.id for m in self.manager.list_installed()}
        self.market_info.setText(f"{len(entries)} featured extension(s)")

        if not entries:
            self.market_layout.insertWidget(0, QLabel("No featured extensions right now."))
            return

        for entry in entries:
            self.market_layout.insertWidget(self.market_layout.count() - 1, self._build_market_card(entry, entry.id in manifests))

    def clear_market_filters(self):
        self.market_type.setCurrentText("all")
        self.market_search.clear()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(APP_VERSION)

    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget { color: #222222; font-family: 'Segoe UI', Arial; }
        QMainWindow { background-color: #f6f7fb; }
        QTabBar::tab { background: #ffffff; padding: 8px 14px; border: 1px solid #e6e6e6; }
        QTabBar::tab:selected { background: #007acc; color: white; }
        QPushButton {
            background: #f0f6ff;
            border: 1px solid #d0e4ff;
            border-radius: 6px;
            padding: 7px 10px;
            min-width: 110px;
        }
        QPushButton:hover { border-color: #007acc; background: #e6f0ff; }
        QPushButton:disabled { color: #888; border-color: #ddd; background: #f5f5f5; }
        QLineEdit, QComboBox {
            background: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 6px;
            padding: 6px;
        }
        QScrollArea { border: none; background: transparent; }
        QLabel { color: #222222; }
        """
    )

    window = VortexExtensionManagerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
