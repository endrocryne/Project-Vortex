import sys
import os
import json
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QUrl, Qt, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtWebChannel import QWebChannel
import ctypes

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
except ImportError:
    print("Error: PyQtWebEngine not found. Please install it using: pip install PyQtWebEngine")
    sys.exit(1)

class SerialBridge(QObject):
    """
    Python backend that polyfills the Web Serial API for the embedded browser.
    Also exposes simple preference methods for UI version and banner actions via QWebChannel.
    """
    data_received = pyqtSignal(str) # Hex or Base64 encoded data to send to JS

    def __init__(self):
        super().__init__()
        self.ser = None
        self._prefs_path = os.path.join(os.path.dirname(__file__), 'vortex_prefs.json')
        # ensure prefs file exists with defaults
        if not os.path.exists(self._prefs_path):
            try:
                with open(self._prefs_path, 'w') as f:
                    json.dump({'ui_version': 'v0.19', 'banner_state': 'shown'}, f)
            except Exception:
                pass

    def _load_prefs(self):
        try:
            with open(self._prefs_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {'ui_version': 'v0.19', 'banner_state': 'shown'}

    def _save_prefs(self, prefs):
        try:
            with open(self._prefs_path + '.tmp', 'w') as f:
                json.dump(prefs, f)
            os.replace(self._prefs_path + '.tmp', self._prefs_path)
        except Exception as e:
            print('Failed to save prefs:', e)

    @pyqtSlot(str, result=str)
    def get_pref(self, key):
        prefs = self._load_prefs()
        val = prefs.get(key, '')
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        return str(val)

    @pyqtSlot(str, str, result=bool)
    def set_pref(self, key, value):
        prefs = self._load_prefs()
        # attempt to parse JSON value
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = value
        prefs[key] = parsed
        self._save_prefs(prefs)
        return True

    @pyqtSlot(str, result=bool)
    def banner_action(self, action):
        prefs = self._load_prefs()
        if action == 'dismiss':
            prefs['banner_state'] = 'dismissed'
        elif action == 'revert':
            prefs['banner_state'] = 'reverted'
            prefs['ui_version'] = 'v0.18'
            # if main window is attached, instruct to reload
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.load_ui_from_prefs()
        self._save_prefs(prefs)
        return True

    @pyqtSlot(str, result=bool)
    def set_ui_version(self, version):
        prefs = self._load_prefs()
        prefs['ui_version'] = version
        self._save_prefs(prefs)
        if hasattr(self, 'main_window') and self.main_window:
            self.main_window.load_ui_from_prefs()
        return True


    @pyqtSlot(result=list)
    def list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    @pyqtSlot(result=str)
    def select_port(self):
        ports = [f"{p.device} - {p.description}" for p in serial.tools.list_ports.comports()]
        if not ports:
            return ""
        
        port, ok = QInputDialog.getItem(None, "Select Serial Port", "Available Ports:", ports, 0, False)
        if ok and port:
            # Return just the COM port name (COMX)
            return port.split(" - ")[0]
        return ""

    @pyqtSlot(str, int, result=bool)
    def open_port(self, port_name, baud_rate):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(port_name, baud_rate, timeout=0.1)
            # Start a thread or timer to read
            self.start_reading()
            return True
        except Exception as e:
            print(f"Failed to open port {port_name}: {e}")
            return False

    @pyqtSlot(str)
    def write_data(self, data_hex):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(bytes.fromhex(data_hex))
            except Exception as e:
                print(f"Write error: {e}")

    def start_reading(self):
        from PyQt5.QtCore import QTimer
        self.read_timer = QTimer()
        self.read_timer.timeout.connect(self._read_logic)
        self.read_timer.start(10) # 100Hz

    def _read_logic(self):
        if self.ser and self.ser.is_open:
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                self.data_received.emit(data.hex())

class VortexControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vortex Mission Control - Vortex Desktop")
        self.resize(1200, 800)
        
        # Set Icon
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'missioncontrol_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            # Fix taskbar icon on Windows
            myappid = 'vortex.missioncontrol.v0.18'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        
        # Web View
        self.browser = QWebEngineView()
        
        # Setup Bridge
        self.bridge = SerialBridge()
        self.channel = QWebChannel()
        self.channel.registerObject("serial_bridge", self.bridge)
        self.browser.page().setWebChannel(self.channel)
        
        # Enable features
        settings = self.browser.settings()
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebGLEnabled, True)
        
        # Early Injection of Polyfill
        self.setup_polyfill()
        
        # Load user preference for UI version
        prefs = self.bridge._load_prefs()
        ui_version = prefs.get('ui_version', 'v0.19')
        if ui_version == 'v0.18':
            local_path = os.path.join(os.path.dirname(__file__), 'vortex-control-v0-18.html')
        else:
            # default to the new MWUI file
            local_path = os.path.join(os.path.dirname(__file__), 'vortex-control-mwui.html')

        if os.path.exists(local_path):
            self.browser.setUrl(QUrl.fromLocalFile(local_path))
        else:
            self.browser.setHtml(f"<h1>File Not Found</h1><p>Could not find {local_path}</p>")

        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        self.setCentralWidget(central_widget)
        self.create_menu()

        # Attach bridge backreference so settings actions can instruct the window
        self.bridge.main_window = self

        # Inject banner script after page loads to show persistent banner when appropriate
        self.browser.loadFinished.connect(self.inject_banner_script)

    def setup_polyfill(self):
        from PyQt5.QtWebEngineWidgets import QWebEngineScript
        
        # This script will run BEFORE the page's own scripts
        script_code = """
        // 1. Load QWebChannel (Standard Qt resource)
        var qt_script = document.createElement('script');
        qt_script.src = 'qrc:///qtwebchannel/qwebchannel.js';
        
        var interval = setInterval(function() {
            if (document.head || document.documentElement) {
                (document.head || document.documentElement).appendChild(qt_script);
                clearInterval(interval);
            }
        }, 10);

        // 2. Initialize Bridge
        window.addEventListener('load', function() {
            if (typeof QWebChannel === 'undefined') {
                console.error("QWebChannel not loaded. Serial polyfill failed.");
                return;
            }
            new QWebChannel(qt.webChannelTransport, function(channel) {
                const bridge = channel.objects.serial_bridge;
                
                const serialPolyfill = {
                    requestPort: async function() {
                        const selectedPort = await bridge.select_port();
                        if (!selectedPort) throw new Error("No port selected");
                        return new VortexSerialPort(selectedPort); 
                    },
                    getPorts: async function() { return []; }
                };

                class VortexSerialPort {
                    constructor(name) { 
                        this.name = name; 
                        this.opened = false;
                        this.readable = null;
                        this.writable = {
                             getWriter: () => ({
                                 write: (data) => {
                                     let hex = Array.from(data).map(b => b.toString(16).padStart(2, '0')).join('');
                                     bridge.write_data(hex);
                                 }
                             })
                        };
                    }
                    async open(options) {
                        const success = await bridge.open_port(this.name, options.baudRate || 9600);
                        if (!success) throw new Error("Could not open port");
                        this.opened = true;
                        
                        let controller;
                        this.readable = new ReadableStream({
                            start(c) { controller = c; }
                        });
                        bridge.data_received.connect((hex) => {
                            let bytes = new Uint8Array(hex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
                            if (controller) controller.enqueue(bytes);
                        });
                    }
                };

                // CRITICAL: Replace navigator.serial with our object
                Object.defineProperty(navigator, 'serial', {
                    value: serialPolyfill,
                    configurable: true,
                    writable: true
                });
                console.log("Vortex: Serial API redirected to Python Backend.");

                // Show top banner if needed (reads prefs via bridge)
                try {
                    const ui_version = bridge.get_pref('ui_version') || 'v0.19';
                    const banner_state = bridge.get_pref('banner_state') || 'shown';
                    if (ui_version === 'v0.19' && banner_state === 'shown') {
                        const banner = document.createElement('div');
                        banner.id = 'vortex-top-banner';
                        banner.style.position = 'fixed';
                        banner.style.top = '0';
                        banner.style.left = '0';
                        banner.style.right = '0';
                        banner.style.background = '#ffd54f';
                        banner.style.color = '#000';
                        banner.style.padding = '10px';
                        banner.style.zIndex = 99999;
                        banner.style.display = 'flex';
                        banner.style.justifyContent = 'space-between';
                        banner.style.alignItems = 'center';

                        const text = document.createElement('div');
                        text.innerText = 'Welcome to Vortex Mission Control v0.19. This new interface is powered by the Mishra WebUI Framework v2.2. Enjoy improved performance and customization options! This is still experimental and may have bugs. You can revert to the old one any time by clicking File -> Revert to old';

                        const controls = document.createElement('div');
                        const revert = document.createElement('button');
                        revert.innerText = 'Revert to old version';
                        revert.style.marginRight = '8px';
                        const dismiss = document.createElement('button');
                        dismiss.innerText = 'Dismiss';

                        revert.addEventListener('click', function() {
                            bridge.banner_action('revert');
                            banner.remove();
                        });
                        dismiss.addEventListener('click', function() {
                            bridge.banner_action('dismiss');
                            banner.remove();
                        });

                        controls.appendChild(revert);
                        controls.appendChild(dismiss);
                        banner.appendChild(text);
                        banner.appendChild(controls);
                        document.body.appendChild(banner);

                        // Push content down so banner doesn't overlap header
                        document.body.style.paddingTop = (parseInt(window.getComputedStyle(document.body).paddingTop || '0') + banner.offsetHeight) + 'px';
                    }
                } catch (e) { console.warn('Banner injection failed:', e); }

            });
        });
        """
        
        script = QWebEngineScript()
        script.setName("VortexSerial")
        script.setSourceCode(script_code)
        script.setInjectionPoint(QWebEngineScript.DocumentCreation)
        script.setWorldId(QWebEngineScript.MainWorld)
        self.browser.page().scripts().insert(script)
    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction('About', self.show_about)

        # Toggle to old UI version (persisted)
        prefs = self.bridge._load_prefs()
        self.use_old_ui_action = QAction('Use Old UI (v0.18)', self)
        self.use_old_ui_action.setCheckable(True)
        self.use_old_ui_action.setChecked(prefs.get('ui_version') == 'v0.18')
        self.use_old_ui_action.triggered.connect(self.toggle_ui_version)
        file_menu.addAction(self.use_old_ui_action)

        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close)
        
        view_menu = menubar.addMenu('View')
        view_menu.addAction("Zoom In", lambda: self.browser.setZoomFactor(self.browser.zoomFactor() + 0.1))
        view_menu.addAction("Zoom Out", lambda: self.browser.setZoomFactor(max(0.1, self.browser.zoomFactor() - 0.1)))
        view_menu.addAction("Reset Zoom", lambda: self.browser.setZoomFactor(1.0))

    def show_about(self):
        QMessageBox.about(self, "About Vortex Control", 
                        "Vortex Control\n\n"
                        "Part of the Vortex Desktop Suite\n"
                        "This tool lets you connect to and control any rocket equipped with VortexLink.\n"
                        "Powered by Mishra JSD, a lightweight and fast Electron alternative, running on PyQtWebEngine\n"
                        "© 2026 Agastya Mishra")

    def inject_banner_script(self, ok):
        # Run a small script in the page context to ensure banner is visible when needed.
        js = r'''
        (function() {
            try {
                if (!window.serial_banner_checked) {
                    window.serial_banner_checked = true;
                    if (typeof QWebChannel !== 'undefined') {
                        new QWebChannel(qt.webChannelTransport, function(channel) {
                            const bridge = channel.objects.serial_bridge;
                            try {
                                const ui_version = bridge.get_pref('ui_version') || 'v0.19';
                                const banner_state = bridge.get_pref('banner_state') || 'shown';
                                if (ui_version === 'v0.19' && banner_state === 'shown' && !document.getElementById('vortex-top-banner')) {
                                    const banner = document.createElement('div');
                                    banner.id = 'vortex-top-banner';
                                    banner.style.position='fixed'; banner.style.top='0'; banner.style.left='0'; banner.style.right='0';
                                    banner.style.background='#ffd54f'; banner.style.color='#000'; banner.style.padding='10px'; banner.style.zIndex=99999;
                                    banner.style.display='flex'; banner.style.justifyContent='space-between'; banner.style.alignItems='center';
                                    const text = document.createElement('div'); text.innerText='Placeholder banner text: New MWUI is available.';
                                    const controls = document.createElement('div');
                                    const revert = document.createElement('button'); revert.innerText='Revert to old version'; revert.style.marginRight='8px';
                                    const dismiss = document.createElement('button'); dismiss.innerText='Dismiss';
                                    revert.addEventListener('click', function() { bridge.banner_action('revert'); banner.remove(); });
                                    dismiss.addEventListener('click', function() { bridge.banner_action('dismiss'); banner.remove(); });
                                    controls.appendChild(revert); controls.appendChild(dismiss);
                                    banner.appendChild(text); banner.appendChild(controls);
                                    document.body.appendChild(banner);
                                    document.body.style.paddingTop = (parseInt(window.getComputedStyle(document.body).paddingTop || '0') + banner.offsetHeight) + 'px';
                                }
                            } catch (e) { }
                        });
                    }
                }
            } catch (e) { }
        })();
        '''
        self.browser.page().runJavaScript(js)
    def toggle_ui_version(self):
        # Toggle between v0.18 and v0.19
        checked = self.use_old_ui_action.isChecked()
        version = 'v0.18' if checked else 'v0.19'
        self.bridge.set_ui_version(version)
        # update menu check state (set_ui_version will call load_ui_from_prefs)
        self.use_old_ui_action.setChecked(checked)

    def load_ui_from_prefs(self):
        prefs = self.bridge._load_prefs()
        ui_version = prefs.get('ui_version', 'v0.19')
        if ui_version == 'v0.18':
            local_path = os.path.join(os.path.dirname(__file__), 'vortex-control-v0-18.html')
            self.use_old_ui_action.setChecked(True)
        else:
            local_path = os.path.join(os.path.dirname(__file__), 'vortex-control-mwui.html')
            self.use_old_ui_action.setChecked(False)
        if os.path.exists(local_path):
            self.browser.setUrl(QUrl.fromLocalFile(local_path))
        else:
            self.browser.setHtml(f"<h1>File Not Found</h1><p>Could not find {local_path}</p>")


if __name__ == '__main__':
    # Disable JS security to allow qrc cross-origin loading if needed
    sys.argv.append("--disable-web-security")
    sys.argv.append("--enable-experimental-web-platform-features")
    
    app = QApplication(sys.argv)
    window = VortexControlWindow()
    window.show()
    sys.exit(app.exec_())
