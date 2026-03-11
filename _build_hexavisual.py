"""
Build script for HexaVisual Pro (Demo) - assembles the complete HTML file
with embedded trajectory data from _traj_data_js.txt
"""
import os

# Read trajectory data block
with open('_traj_data_js.txt', 'r') as f:
    traj_data_block = f.read()

# Build the HTML
html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HexaVisual Pro (Demo) — Vortex Flight Visualizer</title>
<style>
/* ═══════════════════════════════════════════════════════════════
   CSS — HexaVisual Pro (Demo) — Glassmorphism + Dark Theme
   ═══════════════════════════════════════════════════════════════ */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --glass-bg:rgba(15,15,25,0.72);
  --glass-border:rgba(255,255,255,0.08);
  --glass-blur:16px;
  --accent:#00d4ff;
  --accent2:#7c3aed;
  --success:#4CAF50;
  --warning:#FF9800;
  --danger:#F44336;
  --text:#e8e8f0;
  --text-dim:#8888aa;
  --mono:'Consolas','SF Mono','Fira Code',monospace;
  --sans:'Segoe UI','Inter',system-ui,sans-serif;
  --sidebar-w:300px;
  --hud-w:280px;
  --control-h:64px;
  --body-bg:#000;
  --card-bg:rgba(255,255,255,0.03);
  --card-border:rgba(255,255,255,0.06);
  --card-hover-bg:rgba(255,255,255,0.07);
  --card-hover-border:rgba(255,255,255,0.12);
  --speed-option-bg:#1a1a2e;
  --select-bg:rgba(255,255,255,0.08);
  --select-border:rgba(255,255,255,0.1);
  --range-bg:rgba(255,255,255,0.12);
  --toggle-off-bg:rgba(255,255,255,0.12);
  --scrollbar-thumb:rgba(255,255,255,0.15);
  --loading-bg:#0a0a14;
  --upload-border:rgba(255,255,255,0.15);
  --divider:rgba(255,255,255,0.06);
}
/* ── Light theme ── */
body.light{
  --glass-bg:rgba(240,243,255,0.85);
  --glass-border:rgba(0,0,0,0.08);
  --text:#1a1a2e;
  --text-dim:#5566aa;
  --body-bg:#d0d8f0;
  --card-bg:rgba(0,0,0,0.03);
  --card-border:rgba(0,0,0,0.08);
  --card-hover-bg:rgba(0,0,0,0.06);
  --card-hover-border:rgba(0,0,0,0.14);
  --speed-option-bg:#e8ecff;
  --select-bg:rgba(0,0,0,0.07);
  --select-border:rgba(0,0,0,0.12);
  --range-bg:rgba(0,0,0,0.12);
  --toggle-off-bg:rgba(0,0,0,0.12);
  --scrollbar-thumb:rgba(0,0,0,0.18);
  --loading-bg:#d8dff7;
  --upload-border:rgba(0,0,0,0.15);
  --divider:rgba(0,0,0,0.07);
}
html,body{width:100%;height:100%;overflow:hidden;background:var(--body-bg);font-family:var(--sans);color:var(--text);transition:background 0.3s,color 0.3s}
canvas{display:block;position:absolute;top:0;left:0;z-index:0}

/* Glassmorphism base */
.glass{
  background:var(--glass-bg);
  backdrop-filter:blur(var(--glass-blur));
  -webkit-backdrop-filter:blur(var(--glass-blur));
  border:1px solid var(--glass-border);
  border-radius:12px;
}

/* ──── SIDEBAR (left) ──── */
#sidebar{
  position:fixed;top:0;left:0;width:var(--sidebar-w);height:100vh;
  z-index:100;padding:16px 12px;overflow-y:auto;overflow-x:hidden;
  border-radius:0 12px 12px 0;
  scrollbar-width:thin;scrollbar-color:var(--scrollbar-thumb) transparent;
}
#sidebar::-webkit-scrollbar{width:5px}
#sidebar::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb);border-radius:3px}
#sidebar h1{font-size:18px;font-weight:700;letter-spacing:0.5px;margin-bottom:4px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
#sidebar .subtitle{font-size:11px;color:var(--text-dim);margin-bottom:16px}

/* Demo cards */
.demo-card{
  padding:10px 12px;margin-bottom:8px;cursor:pointer;
  border-radius:10px;border:1px solid var(--card-border);
  background:var(--card-bg);transition:all 0.2s;position:relative;overflow:hidden;
}
.demo-card:hover{background:var(--card-hover-bg);border-color:var(--card-hover-border);transform:translateX(3px)}
.demo-card.active{border-color:var(--accent);background:rgba(0,212,255,0.08);box-shadow:0 0 20px rgba(0,212,255,0.1)}
.demo-card .card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.demo-card .card-title{font-size:13px;font-weight:600}
.demo-card .badge{font-size:9px;padding:2px 7px;border-radius:4px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase}
.badge-opt{background:rgba(124,58,237,0.25);color:#b794f6}
.badge-ml{background:rgba(0,212,255,0.2);color:#0090bb}
body.light .badge-ml{color:#0070a0}
.badge-success{background:rgba(76,175,80,0.2);color:#2e7d32}
.badge-fail{background:rgba(255,152,0,0.2);color:#e65100}
.badge-crash{background:rgba(244,67,54,0.2);color:#b71c1c}
.demo-card .card-stats{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:10px;color:var(--text-dim)}
.demo-card .card-stats span{font-family:var(--mono)}
.demo-card .fault-bar{height:3px;border-radius:2px;background:var(--range-bg);margin-top:6px;overflow:hidden}
.demo-card .fault-bar-fill{height:100%;border-radius:2px;transition:width 0.5s}

/* ──── HUD (top-left, offset from sidebar) ──── */
#hud{
  position:fixed;top:16px;left:calc(var(--sidebar-w) + 16px);
  width:var(--hud-w);z-index:90;padding:14px 16px;
}
#hud h2{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:10px;font-weight:600}
.hud-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:12px}
.hud-row .label{color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:0.5px}
.hud-row .value{font-family:var(--mono);font-weight:600;font-size:13px}
.hud-divider{height:1px;background:var(--divider);margin:6px 0}
#engine-status{
  display:inline-block;padding:2px 10px;border-radius:4px;font-size:10px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;margin-top:4px;
}

/* ──── ML OVERLAY (top-right) ──── */
#ml-overlay{
  position:fixed;top:16px;right:16px;width:260px;z-index:90;padding:14px 16px;
}
#ml-overlay h2{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:10px;font-weight:600}
.ml-row{display:flex;justify-content:space-between;padding:3px 0;font-size:11px}
.ml-row .label{color:var(--text-dim)}
.ml-row .value{font-family:var(--mono);font-weight:600}
.ml-meter{height:6px;border-radius:3px;background:var(--range-bg);margin:4px 0;overflow:hidden}
.ml-meter-fill{height:100%;border-radius:3px;transition:width 0.3s}

/* ──── FAULT TOAST NOTIFICATION ──── */
#fault-toast{
  position:fixed;top:80px;left:50%;transform:translateX(-50%) translateY(-20px);
  z-index:200;padding:10px 24px;border-radius:10px;
  background:rgba(244,67,54,0.18);border:1px solid rgba(244,67,54,0.4);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  color:#ff8a80;font-size:13px;font-weight:600;letter-spacing:0.5px;
  font-family:var(--mono);pointer-events:none;
  opacity:0;transition:opacity 0.4s ease, transform 0.4s ease;
  text-align:center;white-space:nowrap;
}
#fault-toast.visible{opacity:1;transform:translateX(-50%) translateY(0)}
#fault-toast.wind{background:rgba(33,150,243,0.18);border-color:rgba(33,150,243,0.4);color:#90caf9}
#fault-toast.drag{background:rgba(255,152,0,0.18);border-color:rgba(255,152,0,0.4);color:#ffcc80}
#fault-toast.mass{background:rgba(244,67,54,0.18);border-color:rgba(244,67,54,0.4);color:#ff8a80}
#fault-toast.combined{background:rgba(156,39,176,0.18);border-color:rgba(156,39,176,0.4);color:#ce93d8}

/* ──── CONTROLS BAR (bottom) ──── */
#controls-bar{
  position:fixed;bottom:0;left:var(--sidebar-w);right:0;height:var(--control-h);
  z-index:100;display:flex;align-items:center;padding:0 20px;gap:12px;
  border-radius:12px 12px 0 0;
}
#controls-bar button{
  background:var(--select-bg);border:1px solid var(--select-border);
  color:var(--text);cursor:pointer;border-radius:8px;padding:6px 10px;
  font-size:14px;transition:all 0.15s;display:flex;align-items:center;justify-content:center;
}
#controls-bar button:hover{background:var(--card-hover-bg);border-color:var(--card-hover-border)}
#controls-bar button.primary{background:rgba(0,212,255,0.15);border-color:rgba(0,212,255,0.3);min-width:44px;min-height:36px;font-size:16px}
#timeline-container{flex:1;display:flex;align-items:center;gap:10px}
#timeline{
  -webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;
  background:var(--range-bg);outline:none;cursor:pointer;
}
#timeline::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--accent);cursor:pointer;box-shadow:0 0 8px rgba(0,212,255,0.4);
}
#timeline::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--accent);border:none;cursor:pointer}
.time-display{font-family:var(--mono);font-size:12px;min-width:100px;text-align:center;color:var(--text-dim)}
#speed-input-wrap{display:flex;align-items:center;gap:4px}
#speed-select{
  background:var(--select-bg);border:1px solid var(--select-border);
  color:var(--text);padding:4px 8px;border-radius:6px 0 0 6px;font-size:11px;font-family:var(--mono);cursor:pointer;
}
#speed-select option{background:var(--speed-option-bg);color:var(--text)}
#speed-custom{
  width:52px;background:var(--select-bg);border:1px solid var(--select-border);border-left:none;
  color:var(--text);padding:4px 6px;border-radius:0 6px 6px 0;font-size:11px;font-family:var(--mono);
  outline:none;
}
#speed-custom:focus{border-color:var(--accent)}

/* ──── SETTINGS PANEL ──── */
#settings-panel{
  position:fixed;top:50%;right:-340px;transform:translateY(-50%);width:300px;
  z-index:110;padding:16px 20px 20px;transition:right 0.3s ease;
}
#settings-panel.open{right:16px}
#settings-panel .panel-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
#settings-panel h2{font-size:13px;font-weight:700}
#settings-close{
  width:26px;height:26px;border-radius:6px;cursor:pointer;
  background:var(--select-bg);border:1px solid var(--select-border);
  color:var(--text);font-size:14px;display:flex;align-items:center;justify-content:center;
  transition:all 0.15s;line-height:1;
}
#settings-close:hover{background:rgba(244,67,54,0.2);border-color:rgba(244,67,54,0.4);color:#e57373}
.setting-group{margin-bottom:14px}
.setting-group label{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--text-dim);margin-bottom:4px}
.setting-group input[type=range]{width:100%;height:3px;-webkit-appearance:none;appearance:none;background:var(--range-bg);border-radius:2px;outline:none}
.setting-group input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--accent);cursor:pointer}
.setting-group select{
  background:var(--select-bg);border:1px solid var(--select-border);
  color:var(--text);padding:2px 6px;border-radius:4px;font-size:11px;
}
.setting-group select option{background:var(--speed-option-bg);color:var(--text)}
.toggle{position:relative;width:36px;height:20px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;inset:0;background:var(--toggle-off-bg);border-radius:10px;transition:0.2s}
.toggle .slider::before{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:var(--text-dim);transition:0.2s}
.toggle input:checked+.slider{background:rgba(0,212,255,0.3)}
.toggle input:checked+.slider::before{transform:translateX(16px);background:var(--accent)}
#settings-btn{
  position:fixed;bottom:calc(var(--control-h) + 12px);right:16px;z-index:105;
  width:40px;height:40px;border-radius:50%;cursor:pointer;font-size:18px;
  display:flex;align-items:center;justify-content:center;
  background:var(--glass-bg);border:1px solid var(--glass-border);color:var(--text);
  backdrop-filter:blur(var(--glass-blur));transition:transform 0.3s;
}
#settings-btn:hover{transform:rotate(30deg)}

/* ──── FPS Counter ──── */
#fps-counter{
  position:fixed;bottom:calc(var(--control-h) + 12px);left:calc(var(--sidebar-w) + 16px);
  z-index:90;font-family:var(--mono);font-size:11px;color:var(--text-dim);
  padding:4px 10px;border-radius:6px;
}

/* ──── Camera mode indicator ──── */
#camera-mode{
  position:fixed;top:16px;left:50%;transform:translateX(-50%);z-index:90;
  padding:4px 16px;font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  color:var(--accent);border-radius:20px;
}

/* ──── Upload zone ──── */
.upload-btn{
  display:block;width:100%;padding:8px;text-align:center;border-radius:8px;
  border:1px dashed var(--upload-border);color:var(--text-dim);font-size:11px;
  cursor:pointer;transition:all 0.2s;margin-top:6px;background:transparent;
}
.upload-btn:hover{border-color:var(--accent);color:var(--accent)}
.upload-btn input{display:none}

/* ──── Scrollbar ──── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb);border-radius:3px}

/* ──── Loading screen ──── */
#loading{
  position:fixed;inset:0;z-index:999;display:flex;flex-direction:column;
  align-items:center;justify-content:center;background:var(--loading-bg);
  transition:opacity 0.5s;
}
#loading.hidden{opacity:0;pointer-events:none}
#loading h1{font-size:28px;font-weight:700;margin-bottom:8px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
#loading p{color:var(--text-dim);font-size:13px}
.loader{width:40px;height:40px;border:3px solid var(--range-bg);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;margin-top:20px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<!-- Loading screen -->
<div id="loading">
  <h1>HexaVisual Pro</h1>
  <p>Initializing 3D Engine...</p>
  <div class="loader"></div>
</div>

<!-- Sidebar -->
<div id="sidebar" class="glass">
  <h1>HexaVisual Pro</h1>
  <div class="subtitle">Vortex Flight Visualizer — Demo</div>
  <div id="demo-cards"></div>
  <label class="upload-btn" title="Upload custom trajectory CSV">
    &#128194; Upload CSV Trajectory
    <input type="file" id="csv-upload" accept=".csv">
  </label>
  <label class="upload-btn" title="Upload STL/OBJ/GLB rocket model" style="margin-top:4px">
    &#128640; Upload Rocket Model
    <input type="file" id="stl-upload" accept=".stl,.obj,.glb,.gltf">
  </label>
</div>

<!-- HUD -->
<div id="hud" class="glass">
  <h2>Flight Telemetry</h2>
  <div class="hud-row"><span class="label">Time</span><span class="value" id="h-time">0.000s</span></div>
  <div class="hud-row"><span class="label">Altitude</span><span class="value" id="h-alt">0.00 m</span></div>
  <div class="hud-row"><span class="label">Vert. Vel</span><span class="value" id="h-vvel">0.00 m/s</span></div>
  <div class="hud-row"><span class="label">Horiz. Vel</span><span class="value" id="h-hvel">0.00 m/s</span></div>
  <div class="hud-row"><span class="label">Speed</span><span class="value" id="h-speed">0.00 m/s</span></div>
  <div class="hud-divider"></div>
  <div class="hud-row"><span class="label">Mass</span><span class="value" id="h-mass">1.282 kg</span></div>
  <div class="hud-row"><span class="label">TTI</span><span class="value" id="h-tti">—</span></div>
  <div id="engine-status" style="background:rgba(255,255,255,0.08);color:var(--text-dim)">IDLE</div>
</div>

<!-- ML Overlay -->
<div id="ml-overlay" class="glass" style="display:none">
  <h2>ML Flight Computer</h2>
  <div class="ml-row"><span class="label">Mode</span><span class="value" id="ml-mode">—</span></div>
  <div class="ml-row"><span class="label">Faults</span><span class="value" id="ml-faults">None</span></div>
  <div class="ml-row"><span class="label">Fault Intensity</span><span class="value" id="ml-intensity">0.00</span></div>
  <div class="ml-meter"><div class="ml-meter-fill" id="ml-intensity-bar" style="width:0%;background:var(--success)"></div></div>
  <div class="hud-divider"></div>
  <div class="ml-row"><span class="label">Baseline Ign. Alt.</span><span class="value" id="ml-base-ign">— m</span></div>
  <div class="ml-row"><span class="label">ML Ign. Correction</span><span class="value" id="ml-ign-corr">— m</span></div>
  <div class="ml-row"><span class="label">Adjusted Ign. Alt.</span><span class="value" id="ml-ign-adj">— m</span></div>
  <div class="hud-divider"></div>
  <div class="ml-row"><span class="label">Landing Vel.</span><span class="value" id="ml-lvel">— m/s</span></div>
  <div class="ml-row"><span class="label">Landing Dist.</span><span class="value" id="ml-ldist">— m</span></div>
</div>

<!-- Fault Toast Notification -->
<div id="fault-toast"></div>

<!-- Camera mode indicator -->
<div id="camera-mode" class="glass">Chase Cam</div>

<!-- FPS counter -->
<div id="fps-counter" class="glass">— FPS</div>

<!-- Controls bar -->
<div id="controls-bar" class="glass">
  <button title="Skip back 5s" onclick="skipTime(-5)">&#9194;</button>
  <button title="Step back" onclick="stepFrame(-1)">&#9198;</button>
  <button class="primary" id="play-btn" onclick="togglePlay()" title="Play / Pause">&#9654;</button>
  <button title="Step forward" onclick="stepFrame(1)">&#9197;</button>
  <button title="Skip forward 5s" onclick="skipTime(5)">&#9193;</button>
  <div id="timeline-container">
    <input type="range" id="timeline" min="0" max="1000" value="0" step="1">
  </div>
  <div class="time-display" id="time-display">0.00 / 0.00</div>
  <div id="speed-input-wrap">
    <select id="speed-select" title="Playback speed preset">
      <option value="0.1">0.1x</option>
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="5">5x</option>
      <option value="10">10x</option>
      <option value="custom">custom</option>
    </select>
    <input type="number" id="speed-custom" value="1" min="0.01" max="100" step="0.1" title="Custom speed multiplier">
  </div>
</div>

<!-- Settings button -->
<div id="settings-btn" onclick="toggleSettings()">&#9881;</div>

<!-- Settings panel -->
<div id="settings-panel" class="glass">
  <div class="panel-header">
    <h2>Settings</h2>
    <div id="settings-close" onclick="closeSettings()" title="Close">&#10005;</div>
  </div>
  <div class="setting-group">
    <label>Camera Mode
      <select id="cam-mode-select" onchange="setCameraMode(this.value)">
        <option value="chase">Chase Cam</option>
        <option value="orbit">Orbit Cam</option>
        <option value="free">Free Cam</option>
      </select>
    </label>
  </div>
  <div class="setting-group">
    <label>Lateral Scale <span id="lat-scale-val">1x</span></label>
    <input type="range" id="lat-scale" min="1" max="50" value="1" step="1" oninput="state.lateralScale=+this.value;document.getElementById('lat-scale-val').textContent=this.value+'x'">
  </div>
  <div class="setting-group">
    <label>Vertical Scale <span id="vert-scale-val">1x</span></label>
    <input type="range" id="vert-scale" min="1" max="5" value="1" step="0.1" oninput="state.verticalScale=+this.value;document.getElementById('vert-scale-val').textContent=parseFloat(this.value).toFixed(1)+'x'">
  </div>
  <div class="setting-group">
    <label>Trail
      <label class="toggle"><input type="checkbox" id="tog-trail" checked onchange="state.showTrail=this.checked"><span class="slider"></span></label>
    </label>
  </div>
  <div class="setting-group">
    <label>Grid
      <label class="toggle"><input type="checkbox" id="tog-grid" checked onchange="groundGrid.visible=this.checked"><span class="slider"></span></label>
    </label>
  </div>
  <div class="setting-group">
    <label>Satellite Ground
      <label class="toggle"><input type="checkbox" id="tog-satellite" onchange="setSatellite(this.checked)"><span class="slider"></span></label>
    </label>
    <div id="sat-controls" style="display:none;margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">
      <div style="display:flex;gap:6px;margin-bottom:6px">
        <div style="flex:1">
          <div style="font-size:10px;color:var(--muted);margin-bottom:2px">Latitude</div>
          <input type="number" id="sat-lat" value="51.5074" step="0.001" style="width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 5px;font-size:11px;box-sizing:border-box">
        </div>
        <div style="flex:1">
          <div style="font-size:10px;color:var(--muted);margin-bottom:2px">Longitude</div>
          <input type="number" id="sat-lon" value="-0.1278" step="0.001" style="width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 5px;font-size:11px;box-sizing:border-box">
        </div>
        <div style="flex:0 0 52px">
          <div style="font-size:10px;color:var(--muted);margin-bottom:2px">Zoom</div>
          <select id="sat-zoom" style="width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 4px;font-size:11px">
            <option value="12">12</option><option value="13">13</option><option value="14" selected>14</option><option value="15">15</option><option value="16">16</option><option value="17">17</option>
          </select>
        </div>
      </div>
      <button onclick="loadSatTile()" style="width:100%;padding:5px;background:var(--accent);color:#000;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600;margin-bottom:8px">Load Tile</button>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:10px;color:var(--muted);white-space:nowrap">Brightness</span>
        <input type="range" id="sat-brightness" min="0.2" max="2.5" value="1.0" step="0.05" style="flex:1;accent-color:var(--accent)" oninput="updateSatBrightness(parseFloat(this.value))">
        <span id="sat-bri-val" style="font-size:10px;color:var(--text);min-width:26px;text-align:right">1.0</span>
      </div>
    </div>
  </div>
  <div class="setting-group">
    <label>Bloom
      <label class="toggle"><input type="checkbox" id="tog-bloom" checked onchange="state.bloomEnabled=this.checked"><span class="slider"></span></label>
    </label>
  </div>
  <div class="setting-group">
    <label>Shadows
      <label class="toggle"><input type="checkbox" id="tog-shadows" checked onchange="renderer.shadowMap.enabled=this.checked"><span class="slider"></span></label>
    </label>
  </div>
  <div class="setting-group">
    <label>Sound
      <label class="toggle"><input type="checkbox" id="tog-sound" onchange="state.soundEnabled=this.checked"><span class="slider"></span></label>
    </label>
  </div>
  <div class="setting-group">
    <label>Light Mode
      <label class="toggle"><input type="checkbox" id="tog-light" onchange="setLightMode(this.checked)"><span class="slider"></span></label>
    </label>
  </div>
</div>

<!-- Three.js + addons via importmap -->
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>

<script type="module">
// ═══════════════════════════════════════════════════════════════
// DATA — Embedded trajectory + manifest
// ═══════════════════════════════════════════════════════════════
''' + '\n' + traj_data_block + '\n' + r'''
const COLS = {T:0,X:1,Y:2,Z:3,VX:4,VY:5,VZ:6,QW:7,QX:8,QY:9,QZ:10,M:11,MLC:12,FT:13,FM:14,WX:15,WY:16,PH:17};

// ═══════════════════════════════════════════════════════════════
// THREE.JS IMPORTS
// ═══════════════════════════════════════════════════════════════
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ═══════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════
const state = {
  playing: false,
  currentTime: 0,
  speed: 1,
  currentRun: null,
  currentTrajectory: null,
  cameraMode: 'chase', // chase | orbit | free
  lateralScale: 1,
  verticalScale: 1,
  showTrail: true,
  bloomEnabled: true,
  soundEnabled: false,
  satelliteEnabled: false,
  lightMode: false,
};

// Expose state for inline HTML event handlers (onchange="state.xxx=...")
window.state = state;

// ═══════════════════════════════════════════════════════════════
// RENDERER & SCENE SETUP
// ═══════════════════════════════════════════════════════════════
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.insertBefore(renderer.domElement, document.body.firstChild);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0a18, 0.0008);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(15, 8, 20);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.08;
orbitControls.minDistance = 2;
orbitControls.maxDistance = 800;
orbitControls.enabled = false; // disabled in chase mode

// Post-processing
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.6, 0.4, 0.85);
composer.addPass(bloomPass);

// ═══════════════════════════════════════════════════════════════
// LIGHTING
// ═══════════════════════════════════════════════════════════════
const ambientLight = new THREE.HemisphereLight(0x4488cc, 0x332211, 0.6);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xfff5e0, 1.8);
sunLight.position.set(80, 150, 60);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.left = -100;
sunLight.shadow.camera.right = 100;
sunLight.shadow.camera.top = 100;
sunLight.shadow.camera.bottom = -100;
sunLight.shadow.camera.far = 500;
sunLight.shadow.bias = -0.0005;
scene.add(sunLight);

const fillLight = new THREE.DirectionalLight(0x88aaff, 0.3);
fillLight.position.set(-40, 60, -30);
scene.add(fillLight);

// ═══════════════════════════════════════════════════════════════
// SKY — Gradient shader
// ═══════════════════════════════════════════════════════════════
const skyGeo = new THREE.SphereGeometry(2000, 32, 32);
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide,
  uniforms: {
    uTop:    { value: new THREE.Color(0x000818) },
    uMiddle: { value: new THREE.Color(0x0a1428) },
    uBottom: { value: new THREE.Color(0x1a1020) },
  },
  vertexShader: `
    varying vec3 vWorldPos;
    void main(){
      vec4 wp = modelMatrix * vec4(position,1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }`,
  fragmentShader: `
    uniform vec3 uTop, uMiddle, uBottom;
    varying vec3 vWorldPos;
    void main(){
      float h = normalize(vWorldPos).y;
      vec3 col = h > 0.0 ? mix(uMiddle, uTop, h) : mix(uMiddle, uBottom, -h);
      gl_FragColor = vec4(col, 1.0);
    }`,
});
scene.add(new THREE.Mesh(skyGeo, skyMat));

// ═══════════════════════════════════════════════════════════════
// GROUND — Grid + optional satellite tile
// ═══════════════════════════════════════════════════════════════
const groundMat = new THREE.MeshStandardMaterial({ color: 0x0a0f1a, roughness: 0.9, metalness: 0.1 });
const groundPlane = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), groundMat);
groundPlane.rotation.x = -Math.PI / 2;
groundPlane.receiveShadow = true;
scene.add(groundPlane);

const groundGrid = new THREE.GridHelper(2000, 200, 0x1a2040, 0x111833);
groundGrid.position.y = 0.05;
scene.add(groundGrid);
// Expose for inline handler (onchange="groundGrid.visible=this.checked")
window.groundGrid = groundGrid;
// Expose for light-mode toggle
window.groundMat = groundMat;

// Launch pad
const padGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.15, 32);
const padMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.6, metalness: 0.5 });
const launchPad = new THREE.Mesh(padGeo, padMat);
launchPad.position.y = 0.075;
launchPad.castShadow = true;
launchPad.receiveShadow = true;
scene.add(launchPad);

// Pad markings
const ringGeo = new THREE.RingGeometry(1.2, 1.4, 32);
const ringMat = new THREE.MeshBasicMaterial({ color: 0xffcc00, side: THREE.DoubleSide });
const padRing = new THREE.Mesh(ringGeo, ringMat);
padRing.rotation.x = -Math.PI / 2;
padRing.position.y = 0.16;
scene.add(padRing);

// Satellite tile (loaded on demand via settings toggle, OFF by default)
let satelliteTex = null;
let activeSatUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/14/8413/5385';

window.setSatellite = function(enabled, overrideUrl) {
  state.satelliteEnabled = enabled;
  const ctrl = document.getElementById('sat-controls');
  if (ctrl) ctrl.style.display = enabled ? '' : 'none';
  if (enabled) {
    const urlToLoad = overrideUrl || activeSatUrl;
    // Force reload if URL changed
    if (overrideUrl && overrideUrl !== activeSatUrl) {
      activeSatUrl = overrideUrl;
      satelliteTex = null;
    }
    if (satelliteTex) {
      groundMat.map = satelliteTex;
      groundMat.needsUpdate = true;
    } else {
      new THREE.TextureLoader().load(urlToLoad, (tex) => {
        satelliteTex = tex;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(8, 8);
        groundMat.map = tex;
        // Re-apply emissiveMap for brightness > 1
        groundMat.emissiveMap = tex;
        groundMat.needsUpdate = true;
      }, undefined, () => {
        console.warn('Satellite tile failed (CORS/network). Try from a web server.');
        document.getElementById('tog-satellite').checked = false;
        state.satelliteEnabled = false;
        if (ctrl) ctrl.style.display = 'none';
      });
    }
  } else {
    groundMat.map = null;
    groundMat.emissiveMap = null;
    groundMat.needsUpdate = true;
  }
};

window.loadSatTile = function() {
  const lat = parseFloat(document.getElementById('sat-lat').value);
  const lon = parseFloat(document.getElementById('sat-lon').value);
  const zoom = parseInt(document.getElementById('sat-zoom').value, 10);
  if (isNaN(lat) || isNaN(lon) || lat < -85.05 || lat > 85.05 || lon < -180 || lon > 180) {
    alert('Invalid coordinates. Latitude: -85.05 to 85.05, Longitude: -180 to 180.');
    return;
  }
  const n = Math.pow(2, zoom);
  const tileX = Math.floor((lon + 180) / 360 * n);
  const latRad = lat * Math.PI / 180;
  const tileY = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
  const url = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${tileY}/${tileX}`;
  setSatellite(true, url);
};

window.updateSatBrightness = function(b) {
  document.getElementById('sat-bri-val').textContent = b.toFixed(1);
  // b <= 1: darken via color scalar; b > 1: full color + emissive boost
  groundMat.color.setScalar(Math.min(1, b));
  groundMat.emissive.setScalar(b > 1 ? 1 : 0);
  groundMat.emissiveIntensity = Math.max(0, b - 1);
  groundMat.needsUpdate = true;
};

// ═══════════════════════════════════════════════════════════════
// PROCEDURAL ROCKET + FLAME SYSTEM
// ═══════════════════════════════════════════════════════════════
const rocketGroup = new THREE.Group();
scene.add(rocketGroup);

// ── Flame (declared first so buildDefaultRocket can reference it) ──
const flameGroup = new THREE.Group();

function makeFlameLayer(color, radius, height, opacity) {
  const geo = new THREE.ConeGeometry(radius, height, 12, 1, true);
  const mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = Math.PI; // point downward (−Y)
  return mesh;
}

const flameCore   = makeFlameLayer(0xffffff, 0.06, 0.8, 1.0);
const flameMid    = makeFlameLayer(0xff8800, 0.10, 1.2, 0.7);
const flameOuter  = makeFlameLayer(0xff2200, 0.16, 1.8, 0.4);
flameGroup.add(flameCore, flameMid, flameOuter);
flameGroup.position.y = -0.3; // at nozzle exit
flameGroup.visible = false;

const glowCanvas = document.createElement('canvas');
glowCanvas.width = 64; glowCanvas.height = 64;
const gCtx = glowCanvas.getContext('2d');
const grad = gCtx.createRadialGradient(32, 32, 0, 32, 32, 32);
grad.addColorStop(0, 'rgba(255,150,50,0.7)');
grad.addColorStop(0.5, 'rgba(255,80,20,0.3)');
grad.addColorStop(1, 'rgba(255,40,10,0)');
gCtx.fillStyle = grad;
gCtx.fillRect(0, 0, 64, 64);
const glowTex = new THREE.CanvasTexture(glowCanvas);
const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
  map: glowTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
}));
glowSprite.scale.set(3, 3, 1);
flameGroup.add(glowSprite);

function animateFlame(t, isBurning) {
  if (!isBurning) { flameGroup.visible = false; return; }
  flameGroup.visible = true;
  const pulse = 1 + 0.15 * Math.sin(t * 40) + 0.08 * Math.sin(t * 67);
  flameCore.scale.set(pulse, pulse, pulse);
  flameMid.scale.set(pulse * 1.05, pulse, pulse * 1.05);
  flameOuter.scale.set(pulse * 0.95, pulse * 1.1, pulse * 0.95);
  glowSprite.material.opacity = 0.5 * pulse;
  glowSprite.scale.set(3 * pulse, 3 * pulse, 1);
}

// ── Default rocket geometry ──
function buildDefaultRocket() {
  while (rocketGroup.children.length > 0) rocketGroup.remove(rocketGroup.children[0]);

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.3, metalness: 0.7 });
  const noseMat = new THREE.MeshStandardMaterial({ color: 0xff3333, roughness: 0.4, metalness: 0.5 });
  const finMat  = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.5, metalness: 0.6 });

  // Body: naturally vertical in Y-up space. Nose at +Y, nozzle at -Y.
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 2, 16), bodyMat);
  body.position.y = 1;
  body.castShadow = true;
  rocketGroup.add(body);

  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.5, 16), noseMat);
  nose.position.y = 2.25;
  nose.castShadow = true;
  rocketGroup.add(nose);

  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 0.2, 12), finMat);
  nozzle.position.y = -0.1;
  rocketGroup.add(nozzle);

  const finShape = new THREE.Shape();
  finShape.moveTo(0, 0);
  finShape.lineTo(0.35, -0.1);
  finShape.lineTo(0.1, 0.5);
  finShape.lineTo(0, 0.5);
  const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: 0.02, bevelEnabled: false });
  for (let i = 0; i < 4; i++) {
    const fin = new THREE.Mesh(finGeo, finMat);
    fin.rotation.y = (i * Math.PI) / 2;
    fin.position.y = 0;
    fin.position.x = Math.cos((i * Math.PI) / 2) * 0.14;
    fin.position.z = Math.sin((i * Math.PI) / 2) * 0.14;
    fin.castShadow = true;
    rocketGroup.add(fin);
  }

  // Re-attach flame group
  rocketGroup.add(flameGroup);
}
buildDefaultRocket();

// ═══════════════════════════════════════════════════════════════
// PARTICLE EXHAUST SYSTEM — sprite-based Points
// ═══════════════════════════════════════════════════════════════
const MAX_PARTICLES = 200;
const particlePositions = new Float32Array(MAX_PARTICLES * 3);
const particleColors = new Float32Array(MAX_PARTICLES * 4);
const particleSizes = new Float32Array(MAX_PARTICLES);
const particleVelocities = [];
const particleAges = new Float32Array(MAX_PARTICLES);
const particleAlive = new Uint8Array(MAX_PARTICLES);

for (let i = 0; i < MAX_PARTICLES; i++) {
  particleVelocities.push(new THREE.Vector3());
  particleAges[i] = 0;
  particleAlive[i] = 0;
}

const particleGeo = new THREE.BufferGeometry();
particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
particleGeo.setAttribute('color', new THREE.BufferAttribute(particleColors, 4));
particleGeo.setAttribute('size', new THREE.BufferAttribute(particleSizes, 1));

// Particle sprite texture
const pCanvas = document.createElement('canvas');
pCanvas.width = 32; pCanvas.height = 32;
const pCtx = pCanvas.getContext('2d');
const pGrad = pCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
pGrad.addColorStop(0, 'rgba(255,255,255,1)');
pGrad.addColorStop(0.3, 'rgba(255,200,100,0.8)');
pGrad.addColorStop(0.7, 'rgba(255,80,20,0.3)');
pGrad.addColorStop(1, 'rgba(0,0,0,0)');
pCtx.fillStyle = pGrad;
pCtx.fillRect(0, 0, 32, 32);
const particleTex = new THREE.CanvasTexture(pCanvas);

const particleMat = new THREE.PointsMaterial({
  size: 0.5, map: particleTex, transparent: true, depthWrite: false,
  blending: THREE.AdditiveBlending, vertexColors: true,
  sizeAttenuation: true,
});
const particleSystem = new THREE.Points(particleGeo, particleMat);
scene.add(particleSystem);

let nextParticle = 0;
function emitParticle(origin, rocketQuat) {
  const i = nextParticle;
  nextParticle = (nextParticle + 1) % MAX_PARTICLES;

  const spread = 0.6;
  const dir = new THREE.Vector3(
    (Math.random() - 0.5) * spread,
    -1 - Math.random() * 0.5,
    (Math.random() - 0.5) * spread
  );
  dir.applyQuaternion(rocketQuat);
  dir.multiplyScalar(8 + Math.random() * 6);

  particlePositions[i * 3]     = origin.x;
  particlePositions[i * 3 + 1] = origin.y;
  particlePositions[i * 3 + 2] = origin.z;
  particleVelocities[i].copy(dir);
  particleAges[i] = 0;
  particleAlive[i] = 1;
  particleColors[i * 4]     = 1;
  particleColors[i * 4 + 1] = 0.9;
  particleColors[i * 4 + 2] = 0.6;
  particleColors[i * 4 + 3] = 1;
  particleSizes[i] = 0.4 + Math.random() * 0.3;
}

function updateParticles(dt) {
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!particleAlive[i]) continue;
    particleAges[i] += dt;
    if (particleAges[i] > 1.0) { particleAlive[i] = 0; particleColors[i * 4 + 3] = 0; continue; }

    const life = particleAges[i];
    particleVelocities[i].y -= 9.81 * dt * 0.3; // light gravity
    particleVelocities[i].multiplyScalar(1 - 2.0 * dt); // drag

    particlePositions[i * 3]     += particleVelocities[i].x * dt;
    particlePositions[i * 3 + 1] += particleVelocities[i].y * dt;
    particlePositions[i * 3 + 2] += particleVelocities[i].z * dt;

    // Color: white→orange→red→transparent
    const t = life;
    particleColors[i * 4]     = 1;
    particleColors[i * 4 + 1] = Math.max(0, 0.9 - t * 1.2);
    particleColors[i * 4 + 2] = Math.max(0, 0.6 - t * 2);
    particleColors[i * 4 + 3] = Math.max(0, 1 - t * 1.5);
    particleSizes[i] *= (1 + dt * 2);
  }
  particleGeo.attributes.position.needsUpdate = true;
  particleGeo.attributes.color.needsUpdate = true;
  particleGeo.attributes.size.needsUpdate = true;
}

// ═══════════════════════════════════════════════════════════════
// FAULT PARTICLE SYSTEM — wind streaks, mass debris, drag distortion
// ═══════════════════════════════════════════════════════════════
const MAX_FAULT_PARTICLES = 300;
const faultPositions = new Float32Array(MAX_FAULT_PARTICLES * 3);
const faultColors = new Float32Array(MAX_FAULT_PARTICLES * 4);
const faultSizes = new Float32Array(MAX_FAULT_PARTICLES);
const faultVelocities = [];
const faultAges = new Float32Array(MAX_FAULT_PARTICLES);
const faultAlive = new Uint8Array(MAX_FAULT_PARTICLES);
const faultTypes = new Uint8Array(MAX_FAULT_PARTICLES); // 1=wind, 2=drag, 3=mass
for (let i = 0; i < MAX_FAULT_PARTICLES; i++) {
  faultVelocities.push(new THREE.Vector3());
  faultAges[i] = 0;
  faultAlive[i] = 0;
  faultTypes[i] = 0;
}
const faultGeo = new THREE.BufferGeometry();
faultGeo.setAttribute('position', new THREE.BufferAttribute(faultPositions, 3));
faultGeo.setAttribute('color', new THREE.BufferAttribute(faultColors, 4));
faultGeo.setAttribute('size', new THREE.BufferAttribute(faultSizes, 1));

// Soft dot texture for fault particles
const faultCanvas = document.createElement('canvas');
faultCanvas.width = 32; faultCanvas.height = 32;
const fCtx = faultCanvas.getContext('2d');
const fGrad = fCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
fGrad.addColorStop(0, 'rgba(255,255,255,1)');
fGrad.addColorStop(0.5, 'rgba(255,255,255,0.4)');
fGrad.addColorStop(1, 'rgba(255,255,255,0)');
fCtx.fillStyle = fGrad;
fCtx.fillRect(0, 0, 32, 32);
const faultTex = new THREE.CanvasTexture(faultCanvas);

const faultMat = new THREE.PointsMaterial({
  size: 0.3, map: faultTex, transparent: true, depthWrite: false,
  blending: THREE.AdditiveBlending, vertexColors: true, sizeAttenuation: true,
});
const faultSystem = new THREE.Points(faultGeo, faultMat);
scene.add(faultSystem);

let nextFaultParticle = 0;
function emitFaultParticle(origin, velocity, color, size, type) {
  const i = nextFaultParticle;
  nextFaultParticle = (nextFaultParticle + 1) % MAX_FAULT_PARTICLES;
  faultPositions[i * 3] = origin.x;
  faultPositions[i * 3 + 1] = origin.y;
  faultPositions[i * 3 + 2] = origin.z;
  faultVelocities[i].copy(velocity);
  faultAges[i] = 0;
  faultAlive[i] = 1;
  faultTypes[i] = type;
  faultColors[i * 4] = color.r;
  faultColors[i * 4 + 1] = color.g;
  faultColors[i * 4 + 2] = color.b;
  faultColors[i * 4 + 3] = 1;
  faultSizes[i] = size;
}

function updateFaultParticles(dt) {
  for (let i = 0; i < MAX_FAULT_PARTICLES; i++) {
    if (!faultAlive[i]) continue;
    faultAges[i] += dt;
    const maxLife = faultTypes[i] === 3 ? 1.5 : 1.0; // debris lives longer
    if (faultAges[i] > maxLife) { faultAlive[i] = 0; faultColors[i * 4 + 3] = 0; continue; }
    const life = faultAges[i] / maxLife;
    if (faultTypes[i] === 3) faultVelocities[i].y -= 9.81 * dt; // gravity for debris
    faultPositions[i * 3] += faultVelocities[i].x * dt;
    faultPositions[i * 3 + 1] += faultVelocities[i].y * dt;
    faultPositions[i * 3 + 2] += faultVelocities[i].z * dt;
    faultColors[i * 4 + 3] = Math.max(0, 1 - life * life); // fade out
    if (faultTypes[i] === 1) faultSizes[i] *= (1 + dt * 0.5); // wind streaks grow
  }
  faultGeo.attributes.position.needsUpdate = true;
  faultGeo.attributes.color.needsUpdate = true;
  faultGeo.attributes.size.needsUpdate = true;
}

// Drag visual: store reference to body mesh so we can scale it
let dragScaleTarget = 1.0;
let dragScaleCurrent = 1.0;

// Fault toast state
let lastFaultType = 0;
let faultToastTimer = 0;
const FAULT_NAMES = { 1: 'WIND GUST', 2: 'DRAG CHANGE', 3: 'MASS LOSS', 4: 'COMBINED FAULTS' };
const FAULT_CLASSES = { 1: 'wind', 2: 'drag', 3: 'mass', 4: 'combined' };

function showFaultToast(faultType, magnitude) {
  const toast = document.getElementById('fault-toast');
  const name = FAULT_NAMES[faultType] || 'FAULT';
  const pct = (magnitude * 100).toFixed(0);
  toast.textContent = '\u26A0 ' + name + ' DETECTED — Intensity ' + pct + '%';
  toast.className = 'visible ' + (FAULT_CLASSES[faultType] || '');
  faultToastTimer = 3.0;
}

function updateFaultToast(dt) {
  if (faultToastTimer > 0) {
    faultToastTimer -= dt;
    if (faultToastTimer <= 0) {
      document.getElementById('fault-toast').className = '';
    }
  }
}

function updateFaultEffects(sample, dt) {
  if (!sample || !state.playing) {
    dragScaleTarget = 1.0;
    return;
  }
  const ft = Math.round(sample.faultType);
  const fm = sample.faultMag || 0;
  const rPos = rocketGroup.position;

  // Detect fault onset for toast
  if (ft !== 0 && lastFaultType === 0) {
    showFaultToast(ft, fm);
  }
  lastFaultType = ft;

  // WIND GUST particles (fault type 1 or 4)
  if ((ft === 1 || ft === 4) && fm > 0.01) {
    const windX = sample.windX || 0;
    const windY = sample.windY || 0;
    const windMag = Math.sqrt(windX * windX + windY * windY);
    if (windMag > 0.1) {
      // Emit horizontal wind streaks around rocket
      const count = Math.ceil(fm * 8);
      const windColor = new THREE.Color(0.5, 0.7, 1.0); // light blue
      for (let p = 0; p < count; p++) {
        // Spawn in a box around the rocket, offset upwind
        const spread = 4;
        const windDirX = windX / windMag;
        const windDirZ = windY / windMag; // sim Y → three Z
        const ox = rPos.x - windDirX * spread + (Math.random() - 0.5) * spread * 0.5;
        const oy = rPos.y + (Math.random() - 0.5) * 3;
        const oz = rPos.z - windDirZ * spread + (Math.random() - 0.5) * spread * 0.5;
        const vel = new THREE.Vector3(
          windDirX * windMag * 2 + (Math.random() - 0.5) * 0.5,
          (Math.random() - 0.5) * 0.3,
          windDirZ * windMag * 2 + (Math.random() - 0.5) * 0.5
        );
        emitFaultParticle(
          new THREE.Vector3(ox, oy, oz), vel, windColor,
          0.15 + Math.random() * 0.15, 1
        );
      }
    }
  }

  // MASS LOSS debris (fault type 3 or 4)
  if ((ft === 3 || ft === 4) && fm > 0.01) {
    const count = Math.ceil(fm * 4);
    const debrisColor = new THREE.Color(0.8, 0.4, 0.2); // brown/orange
    for (let p = 0; p < count; p++) {
      const ox = rPos.x + (Math.random() - 0.5) * 0.4;
      const oy = rPos.y + (Math.random() - 0.5) * 1.5;
      const oz = rPos.z + (Math.random() - 0.5) * 0.4;
      const vel = new THREE.Vector3(
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 2,
        (Math.random() - 0.5) * 3
      );
      emitFaultParticle(
        new THREE.Vector3(ox, oy, oz), vel, debrisColor,
        0.1 + Math.random() * 0.2, 3
      );
    }
  }

  // DRAG CHANGE geometry distortion (fault type 2 or 4)
  if ((ft === 2 || ft === 4) && fm > 0.01) {
    // Bulge the rocket body outward to indicate increased drag cross-section
    dragScaleTarget = 1.0 + fm * 0.6;
    // Also emit orange wisps around rocket to indicate turbulence
    if (Math.random() < fm * 2) {
      const dragColor = new THREE.Color(1.0, 0.6, 0.2); // orange
      const angle = Math.random() * Math.PI * 2;
      const r = 0.5 + Math.random() * 0.5;
      const ox = rPos.x + Math.cos(angle) * r;
      const oy = rPos.y + (Math.random() - 0.5) * 2;
      const oz = rPos.z + Math.sin(angle) * r;
      const vel = new THREE.Vector3(
        Math.cos(angle) * 1.5,
        Math.random() * 1.5,
        Math.sin(angle) * 1.5
      );
      emitFaultParticle(
        new THREE.Vector3(ox, oy, oz), vel, dragColor,
        0.2 + Math.random() * 0.15, 2
      );
    }
  } else {
    dragScaleTarget = 1.0;
  }

  // Smoothly interpolate drag visual scale on rocket body
  dragScaleCurrent += (dragScaleTarget - dragScaleCurrent) * Math.min(1, dt * 5);
  const body = rocketGroup.children[0]; // first child is the body cylinder
  if (body && body.isMesh) {
    body.scale.x = dragScaleCurrent;
    body.scale.z = dragScaleCurrent;
  }
}

// ═══════════════════════════════════════════════════════════════
// TRAIL — line showing past trajectory
// ═══════════════════════════════════════════════════════════════
const MAX_TRAIL = 2000;
const trailPositions = new Float32Array(MAX_TRAIL * 3);
const trailColors = new Float32Array(MAX_TRAIL * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailColors, 3));
const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 }));
trailLine.frustumCulled = false;
scene.add(trailLine);
let trailCount = 0;

function resetTrail() { trailCount = 0; trailGeo.setDrawRange(0, 0); }

function addTrailPoint(x, y, z, color) {
  if (trailCount >= MAX_TRAIL) return;
  const i = trailCount * 3;
  trailPositions[i] = x; trailPositions[i + 1] = y; trailPositions[i + 2] = z;
  trailColors[i] = color.r; trailColors[i + 1] = color.g; trailColors[i + 2] = color.b;
  trailCount++;
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;
  trailGeo.setDrawRange(0, trailCount);
}

// ═══════════════════════════════════════════════════════════════
// LANDING PREDICTION MARKER
// ═══════════════════════════════════════════════════════════════
const predRing = new THREE.Mesh(
  new THREE.RingGeometry(0.5, 0.7, 32),
  new THREE.MeshBasicMaterial({ color: 0xff4444, transparent: true, opacity: 0.6, side: THREE.DoubleSide })
);
predRing.rotation.x = -Math.PI / 2;
predRing.position.y = 0.1;
predRing.visible = false;
scene.add(predRing);

// ═══════════════════════════════════════════════════════════════
// CRASH EXPLOSION EFFECT
// ═══════════════════════════════════════════════════════════════
let crashActive = false, crashTime = 0;
const crashParticles = [];
const CRASH_COUNT = 80;

function triggerCrash(pos) {
  crashActive = true; crashTime = 0;
  crashParticles.length = 0;
  for (let i = 0; i < CRASH_COUNT; i++) {
    const vel = new THREE.Vector3(
      (Math.random() - 0.5) * 20,
      Math.random() * 15 + 5,
      (Math.random() - 0.5) * 20
    );
    crashParticles.push({ pos: pos.clone(), vel, age: 0 });
  }
}

// Crash flash sphere
const crashFlash = new THREE.Mesh(
  new THREE.SphereGeometry(2, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0xff6600, transparent: true, opacity: 0, blending: THREE.AdditiveBlending })
);
scene.add(crashFlash);

// ═══════════════════════════════════════════════════════════════
// INTERPOLATION ENGINE
// ═══════════════════════════════════════════════════════════════
function cubicInterp(y0, y1, y2, y3, t) {
  const a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
  const b = y0 - 2.5*y1 + 2*y2 - 0.5*y3;
  const c = -0.5*y0 + 0.5*y2;
  const d = y1;
  return a*t*t*t + b*t*t + c*t + d;
}

function slerpQuat(q1, q2, t) {
  // Data order: [qw, qx, qy, qz]. THREE.Quaternion constructor: (x, y, z, w).
  const qa = new THREE.Quaternion(q1[1], q1[2], q1[3], q1[0]); // x=qx, y=qy, z=qz, w=qw
  const qb = new THREE.Quaternion(q2[1], q2[2], q2[3], q2[0]);
  qa.slerp(qb, t);
  return qa;
}

function sampleTrajectory(traj, time) {
  if (!traj || traj.length < 2) return null;
  const tMax = traj[traj.length - 1][COLS.T];
  const tMin = traj[0][COLS.T];
  const t = Math.max(tMin, Math.min(tMax, time));

  // Binary search for bracket
  let lo = 0, hi = traj.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (traj[mid][COLS.T] <= t) lo = mid; else hi = mid;
  }

  const t0 = traj[lo][COLS.T], t1 = traj[hi][COLS.T];
  const frac = t1 > t0 ? (t - t0) / (t1 - t0) : 0;

  // Catmull-Rom with clamped neighbors
  const i0 = Math.max(0, lo - 1);
  const i3 = Math.min(traj.length - 1, hi + 1);
  const r0 = traj[i0], r1 = traj[lo], r2 = traj[hi], r3 = traj[i3];

  const x  = cubicInterp(r0[COLS.X], r1[COLS.X], r2[COLS.X], r3[COLS.X], frac);
  const y  = cubicInterp(r0[COLS.Y], r1[COLS.Y], r2[COLS.Y], r3[COLS.Y], frac);
  const z  = cubicInterp(r0[COLS.Z], r1[COLS.Z], r2[COLS.Z], r3[COLS.Z], frac);
  const vx = cubicInterp(r0[COLS.VX], r1[COLS.VX], r2[COLS.VX], r3[COLS.VX], frac);
  const vy = cubicInterp(r0[COLS.VY], r1[COLS.VY], r2[COLS.VY], r3[COLS.VY], frac);
  const vz = cubicInterp(r0[COLS.VZ], r1[COLS.VZ], r2[COLS.VZ], r3[COLS.VZ], frac);
  const mass = r1[COLS.M] + (r2[COLS.M] - r1[COLS.M]) * frac;

  // New columns: linear interpolation for ML correction, fault data, wind, phase
  const mlCorrection = r1[COLS.MLC] + (r2[COLS.MLC] - r1[COLS.MLC]) * frac;
  const faultType = r1[COLS.FT] > 0 ? r1[COLS.FT] : r2[COLS.FT]; // nearest-neighbor for discrete
  const faultMag = r1[COLS.FM] + (r2[COLS.FM] - r1[COLS.FM]) * frac;
  const windX = r1[COLS.WX] + (r2[COLS.WX] - r1[COLS.WX]) * frac;
  const windY = r1[COLS.WY] + (r2[COLS.WY] - r1[COLS.WY]) * frac;
  const phase = Math.round(r1[COLS.PH] + (r2[COLS.PH] - r1[COLS.PH]) * frac);

  // Detect motor burn from Phase column: 0=ascent_burn, 3=descent_burn
  const isBurning = (phase === 0 || phase === 3);

  // SLERP quaternion
  const q = slerpQuat(
    [r1[COLS.QW], r1[COLS.QX], r1[COLS.QY], r1[COLS.QZ]],
    [r2[COLS.QW], r2[COLS.QX], r2[COLS.QY], r2[COLS.QZ]],
    frac
  );

  return { t, x, y, z, vx, vy, vz, mass, quat: q, isBurning,
           mlCorrection, faultType, faultMag, windX, windY, phase };
}

// ═══════════════════════════════════════════════════════════════
// DEMO CARDS — Build sidebar
// ═══════════════════════════════════════════════════════════════
const cardsContainer = document.getElementById('demo-cards');
const runs = MANIFEST.runs;

runs.forEach((run, idx) => {
  const card = document.createElement('div');
  card.className = 'demo-card';
  card.dataset.id = run.id;

  const statusClass = run.status_label === 'SUCCESS' ? 'badge-success' : run.status_label === 'FAIL' ? 'badge-fail' : 'badge-crash';
  const modeClass = run.mode === 'ML' ? 'badge-ml' : 'badge-opt';
  const fi = run.fault_intensity;
  const fiColor = fi < 0.3 ? 'var(--success)' : fi < 0.6 ? 'var(--warning)' : 'var(--danger)';
  const faultStr = run.fault_types.length ? run.fault_types.join(', ') : 'None';

  card.innerHTML = `
    <div class="card-header">
      <span class="card-title">${run.name}</span>
      <span>
        <span class="badge ${modeClass}">${run.mode}</span>
        <span class="badge ${statusClass}">${run.status_label}</span>
      </span>
    </div>
    <div class="card-stats">
      <span>Faults: ${faultStr}</span>
      <span>FI: ${fi.toFixed(2)}</span>
      <span>V<sub>land</sub>: ${run.landing_velocity.toFixed(1)} m/s</span>
      <span>Dist: ${run.landing_distance.toFixed(1)} m</span>
    </div>
    <div class="fault-bar"><div class="fault-bar-fill" style="width:${fi * 100}%;background:${fiColor}"></div></div>
  `;

  card.addEventListener('click', () => loadRun(run.id));
  cardsContainer.appendChild(card);
});

// ═══════════════════════════════════════════════════════════════
// LOAD A RUN
// ═══════════════════════════════════════════════════════════════
function loadRun(runId) {
  const run = runs.find(r => r.id === runId);
  if (!run) return;

  state.currentRun = run;
  state.currentTrajectory = TRAJECTORY_DATA[runId];
  state.currentTime = 0;
  state.playing = false;
  document.getElementById('play-btn').innerHTML = '&#9654;';

  // Highlight card
  document.querySelectorAll('.demo-card').forEach(c => c.classList.remove('active'));
  document.querySelector(`.demo-card[data-id="${runId}"]`)?.classList.add('active');

  // Reset trail and crash
  resetTrail();
  crashActive = false;
  crashFlash.material.opacity = 0;
  predRing.visible = false;
  lastTrailTime = -1;
  lastFaultType = 0;
  faultToastTimer = 0;
  document.getElementById('fault-toast').className = '';
  dragScaleTarget = 1.0;
  dragScaleCurrent = 1.0;

  // ML overlay — show only for ML runs
  const isML = run.mode === 'ML';
  document.getElementById('ml-overlay').style.display = isML ? '' : 'none';
  if (isML) {
    document.getElementById('ml-mode').textContent = run.mode;
    document.getElementById('ml-faults').textContent = run.fault_types.length ? run.fault_types.join(', ') : 'None';
    document.getElementById('ml-intensity').textContent = run.fault_intensity.toFixed(2);
    const fi = run.fault_intensity;
    const fiBar = document.getElementById('ml-intensity-bar');
    fiBar.style.width = (fi * 100) + '%';
    fiBar.style.background = fi < 0.3 ? 'var(--success)' : fi < 0.6 ? 'var(--warning)' : 'var(--danger)';
    document.getElementById('ml-lvel').textContent = run.landing_velocity.toFixed(2) + ' m/s';
    document.getElementById('ml-ldist').textContent = run.landing_distance.toFixed(2) + ' m';
  }

  // Update timeline
  const tMax = state.currentTrajectory[state.currentTrajectory.length - 1][COLS.T];
  document.getElementById('timeline').max = 1000;
  updateTimeDisplay();

  // Camera reset
  if (state.cameraMode === 'chase') {
    camera.position.set(15, 8, 20);
  }
}

// ═══════════════════════════════════════════════════════════════
// PLAYBACK CONTROLS
// ═══════════════════════════════════════════════════════════════
window.togglePlay = function() {
  if (!state.currentTrajectory) return;
  state.playing = !state.playing;
  document.getElementById('play-btn').innerHTML = state.playing ? '&#9646;&#9646;' : '&#9654;';
};

window.skipTime = function(delta) {
  if (!state.currentTrajectory) return;
  const tMax = state.currentTrajectory[state.currentTrajectory.length - 1][COLS.T];
  state.currentTime = Math.max(0, Math.min(tMax, state.currentTime + delta));
  updateTimeDisplay();
};

window.stepFrame = function(dir) {
  if (!state.currentTrajectory) return;
  const dt = state.currentTrajectory.length > 1 ? (state.currentTrajectory[1][COLS.T] - state.currentTrajectory[0][COLS.T]) : 0.1;
  state.currentTime = Math.max(0, state.currentTime + dir * dt);
  updateTimeDisplay();
};

document.getElementById('timeline').addEventListener('input', (e) => {
  if (!state.currentTrajectory) return;
  const tMax = state.currentTrajectory[state.currentTrajectory.length - 1][COLS.T];
  state.currentTime = (e.target.value / 1000) * tMax;
  updateTimeDisplay();
});

document.getElementById('speed-select').addEventListener('change', (e) => {
  if (e.target.value === 'custom') {
    document.getElementById('speed-custom').focus();
  } else {
    state.speed = parseFloat(e.target.value);
    document.getElementById('speed-custom').value = state.speed;
  }
});

document.getElementById('speed-custom').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  if (v > 0 && isFinite(v)) {
    state.speed = v;
    const match = [...document.getElementById('speed-select').options].find(o => o.value !== 'custom' && Math.abs(parseFloat(o.value) - v) < 0.001);
    document.getElementById('speed-select').value = match ? match.value : 'custom';
  }
});

function updateTimeDisplay() {
  const tMax = state.currentTrajectory ? state.currentTrajectory[state.currentTrajectory.length - 1][COLS.T] : 0;
  document.getElementById('time-display').textContent = `${state.currentTime.toFixed(2)} / ${tMax.toFixed(2)}`;
  if (tMax > 0) {
    document.getElementById('timeline').value = (state.currentTime / tMax) * 1000;
  }
}

window.toggleSettings = function() {
  document.getElementById('settings-panel').classList.toggle('open');
};

window.closeSettings = function() {
  document.getElementById('settings-panel').classList.remove('open');
};

// Expose renderer for inline handler (onchange="renderer.shadowMap.enabled=this.checked")
window.renderer = renderer;

// Light mode toggle
window.setLightMode = function(enabled) {
  state.lightMode = enabled;
  document.body.classList.toggle('light', enabled);
  if (enabled) {
    skyMat.uniforms.uTop.value.setHex(0x87ceeb);
    skyMat.uniforms.uMiddle.value.setHex(0xb0d8f5);
    skyMat.uniforms.uBottom.value.setHex(0xc8e8f0);
    scene.fog.color.setHex(0xb0d8f5);
    groundMat.color.setHex(0x4a7a3a);
    groundGrid.material.color = new THREE.Color(0x335522);
    groundGrid.material.needsUpdate = true;
  } else {
    skyMat.uniforms.uTop.value.setHex(0x000818);
    skyMat.uniforms.uMiddle.value.setHex(0x0a1428);
    skyMat.uniforms.uBottom.value.setHex(0x1a1020);
    scene.fog.color.setHex(0x0a0a18);
    groundMat.color.setHex(0x0a0f1a);
    groundGrid.material.color = new THREE.Color(0x1a2040);
    groundGrid.material.needsUpdate = true;
  }
};

// ═══════════════════════════════════════════════════════════════
// CAMERA MODES
// ═══════════════════════════════════════════════════════════════
window.setCameraMode = function(mode) {
  state.cameraMode = mode;
  const label = mode === 'chase' ? 'Chase Cam' : mode === 'orbit' ? 'Orbit Cam' : 'Free Cam';
  document.getElementById('camera-mode').textContent = label;
  document.getElementById('cam-mode-select').value = mode;
  orbitControls.enabled = (mode !== 'chase');
};

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === '1') setCameraMode('chase');
  else if (e.key === '2') setCameraMode('orbit');
  else if (e.key === '3') setCameraMode('free');
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowRight') skipTime(1);
  else if (e.key === 'ArrowLeft') skipTime(-1);
  else if (e.key === 'Escape') closeSettings();
});

// ═══════════════════════════════════════════════════════════════
// STL / OBJ / GLB UPLOAD
// ═══════════════════════════════════════════════════════════════
document.getElementById('stl-upload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const ext = file.name.split('.').pop().toLowerCase();
  const reader = new FileReader();
  reader.onload = (ev) => {
    let loader, isBuffer = false;
    if (ext === 'stl') { loader = new STLLoader(); isBuffer = true; }
    else if (ext === 'obj') { loader = new OBJLoader(); }
    else if (ext === 'glb' || ext === 'gltf') { loader = new GLTFLoader(); isBuffer = true; }
    else return;

    try {
      if (ext === 'stl') {
        const geo = loader.parse(ev.target.result);
        geo.computeBoundingBox();
        const bb = geo.boundingBox;
        const h = bb.max.y - bb.min.y;
        const scale = 2.0 / Math.max(h, 0.001);
        geo.scale(scale, scale, scale);
        geo.computeBoundingBox();
        const center = new THREE.Vector3();
        geo.boundingBox.getCenter(center);
        geo.translate(-center.x, -geo.boundingBox.min.y, -center.z);

        while (rocketGroup.children.length > 0) rocketGroup.remove(rocketGroup.children[0]);
        const mat = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.3, metalness: 0.6 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        rocketGroup.add(mesh);
        rocketGroup.add(flameGroup);
      } else if (ext === 'obj') {
        const obj = loader.parse(ev.target.result);
        const bb = new THREE.Box3().setFromObject(obj);
        const h = bb.max.y - bb.min.y;
        const scale = 2.0 / Math.max(h, 0.001);
        obj.scale.set(scale, scale, scale);
        const center = new THREE.Vector3();
        bb.getCenter(center);
        obj.position.sub(center.multiplyScalar(scale));

        while (rocketGroup.children.length > 0) rocketGroup.remove(rocketGroup.children[0]);
        obj.traverse(c => { if (c.isMesh) c.castShadow = true; });
        rocketGroup.add(obj);
        rocketGroup.add(flameGroup);
      } else if (ext === 'glb' || ext === 'gltf') {
        loader.parse(ev.target.result, '', (gltf) => {
          const obj = gltf.scene;
          const bb = new THREE.Box3().setFromObject(obj);
          const h = bb.max.y - bb.min.y;
          const scale = 2.0 / Math.max(h, 0.001);
          obj.scale.set(scale, scale, scale);

          while (rocketGroup.children.length > 0) rocketGroup.remove(rocketGroup.children[0]);
          obj.traverse(c => { if (c.isMesh) c.castShadow = true; });
          rocketGroup.add(obj);
          rocketGroup.add(flameGroup);
        });
      }
    } catch (err) {
      console.error('Model load error:', err);
    }
  };
  if (ext === 'obj') reader.readAsText(file);
  else reader.readAsArrayBuffer(file);
});

// ═══════════════════════════════════════════════════════════════
// CSV UPLOAD — custom trajectory
// ═══════════════════════════════════════════════════════════════
document.getElementById('csv-upload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const text = ev.target.result;
    const lines = text.trim().split('\n');
    if (lines.length < 2) return;
    const header = lines[0].split(',').map(h => h.trim());
    const reqCols = ['Time','X','Y','Z','VX','VY','VZ','QW','QX','QY','QZ','Mass'];
    const colMap = {};
    reqCols.forEach(c => { colMap[c] = header.indexOf(c); });
    if (Object.values(colMap).some(v => v === -1)) {
      alert('CSV must have columns: ' + reqCols.join(', '));
      return;
    }

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',').map(Number);
      if (vals.length < 12) continue;
      rows.push(reqCols.map(c => vals[colMap[c]]));
    }

    const customId = 'custom_' + Date.now();
    TRAJECTORY_DATA[customId] = rows;

    // Add a synthetic run entry
    const customRun = {
      id: customId,
      name: 'Custom: ' + file.name.replace('.csv', ''),
      mode: 'Custom',
      fault_intensity: 0,
      fault_types: [],
      success: true,
      landing_velocity: 0,
      landing_distance: 0,
      color: '#00d4ff',
      status_label: 'CUSTOM',
    };
    runs.push(customRun);

    // Add card
    const card = document.createElement('div');
    card.className = 'demo-card';
    card.dataset.id = customId;
    card.innerHTML = `
      <div class="card-header">
        <span class="card-title">${customRun.name}</span>
        <span class="badge" style="background:rgba(0,212,255,0.2);color:#67e8f9">CUSTOM</span>
      </div>
      <div class="card-stats"><span>${rows.length} points, ${rows[rows.length-1][0].toFixed(1)}s</span></div>
    `;
    card.addEventListener('click', () => loadRun(customId));
    cardsContainer.appendChild(card);
    loadRun(customId);
  };
  reader.readAsText(file);
});

// ═══════════════════════════════════════════════════════════════
// SOUND ENGINE (optional) — Web Audio API
// ═══════════════════════════════════════════════════════════════
let audioCtx = null, engineOsc = null, engineGain = null, noiseNode = null, noiseGain = null;
function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  // Engine oscillator
  engineOsc = audioCtx.createOscillator();
  engineOsc.type = 'sawtooth';
  engineOsc.frequency.value = 80;
  engineGain = audioCtx.createGain();
  engineGain.gain.value = 0;
  const engineFilter = audioCtx.createBiquadFilter();
  engineFilter.type = 'lowpass';
  engineFilter.frequency.value = 400;
  engineOsc.connect(engineFilter).connect(engineGain).connect(audioCtx.destination);
  engineOsc.start();

  // Noise for wind
  const bufSize = audioCtx.sampleRate * 2;
  const noiseBuffer = audioCtx.createBuffer(1, bufSize, audioCtx.sampleRate);
  const output = noiseBuffer.getChannelData(0);
  for (let i = 0; i < bufSize; i++) output[i] = Math.random() * 2 - 1;
  noiseNode = audioCtx.createBufferSource();
  noiseNode.buffer = noiseBuffer;
  noiseNode.loop = true;
  noiseGain = audioCtx.createGain();
  noiseGain.gain.value = 0;
  const windFilter = audioCtx.createBiquadFilter();
  windFilter.type = 'lowpass';
  windFilter.frequency.value = 200;
  noiseNode.connect(windFilter).connect(noiseGain).connect(audioCtx.destination);
  noiseNode.start();
}

function updateAudio(isBurning, speed) {
  if (!state.soundEnabled || !audioCtx) return;
  if (engineGain) {
    engineGain.gain.linearRampToValueAtTime(isBurning ? 0.15 : 0, audioCtx.currentTime + 0.05);
    engineOsc.frequency.linearRampToValueAtTime(isBurning ? 280 : 80, audioCtx.currentTime + 0.05);
  }
  if (noiseGain) {
    noiseGain.gain.linearRampToValueAtTime(Math.min(0.1, speed * 0.002), audioCtx.currentTime + 0.05);
  }
}

// ═══════════════════════════════════════════════════════════════
// HUD UPDATE
// ═══════════════════════════════════════════════════════════════
function updateHUD(sample, isBurning) {
  if (!sample) return;
  const alt = sample.z;
  const vVel = sample.vz;
  const hVel = Math.sqrt(sample.vx * sample.vx + sample.vy * sample.vy);
  const speed = Math.sqrt(sample.vx * sample.vx + sample.vy * sample.vy + sample.vz * sample.vz);

  document.getElementById('h-time').textContent = sample.t.toFixed(3) + 's';

  const altEl = document.getElementById('h-alt');
  altEl.textContent = alt.toFixed(2) + ' m';
  altEl.style.color = alt > 50 ? 'var(--success)' : alt > 10 ? 'var(--warning)' : 'var(--danger)';

  const vvelEl = document.getElementById('h-vvel');
  vvelEl.textContent = vVel.toFixed(2) + ' m/s';
  vvelEl.style.color = vVel > 0 ? 'var(--success)' : vVel > -5 ? 'var(--warning)' : 'var(--danger)';

  document.getElementById('h-hvel').textContent = hVel.toFixed(2) + ' m/s';
  document.getElementById('h-speed').textContent = speed.toFixed(2) + ' m/s';
  document.getElementById('h-mass').textContent = sample.mass.toFixed(3) + ' kg';

  // TTI estimate
  if (vVel < 0 && alt > 0) {
    const tti = alt / Math.abs(vVel);
    document.getElementById('h-tti').textContent = tti.toFixed(1) + 's';
  } else {
    document.getElementById('h-tti').textContent = '—';
  }

  // Engine status
  const statusEl = document.getElementById('engine-status');
  const tMax = state.currentTrajectory ? state.currentTrajectory[state.currentTrajectory.length - 1][COLS.T] : 0;
  const isEnd = sample.t >= tMax - 0.01;
  const isCrash = state.currentRun && !state.currentRun.success && isEnd;

  if (isCrash && isEnd) {
    statusEl.textContent = state.currentRun.status_label === 'CRASH' ? 'CRASHED' : 'FAILED';
    statusEl.style.background = 'rgba(244,67,54,0.3)';
    statusEl.style.color = '#e57373';
  } else if (isEnd && state.currentRun?.success) {
    statusEl.textContent = 'LANDED';
    statusEl.style.background = 'rgba(76,175,80,0.3)';
    statusEl.style.color = '#81c784';
  } else if (isBurning) {
    statusEl.textContent = alt > 10 && vVel < 0 ? 'SUICIDE BURN' : 'POWERED ASCENT';
    statusEl.style.background = 'rgba(255,152,0,0.3)';
    statusEl.style.color = '#ffb74d';
  } else if (!state.playing) {
    statusEl.textContent = 'PAUSED';
    statusEl.style.background = 'var(--select-bg)';
    statusEl.style.color = 'var(--text-dim)';
  } else {
    statusEl.textContent = 'COAST';
    statusEl.style.background = 'rgba(0,212,255,0.15)';
    statusEl.style.color = 'var(--accent)';
  }
}

// ═══════════════════════════════════════════════════════════════
// ML OVERLAY UPDATE — ignition altitude correction
// ═══════════════════════════════════════════════════════════════
function computeBaselineIgnitionAlt(run) {
  // Scan trajectory for descent-burn start (phase 3) to find actual ignition altitude
  const traj = state.currentTrajectory;
  if (!traj) return 30;
  for (let i = 0; i < traj.length; i++) {
    if (traj[i][COLS.PH] === 3) return traj[i][COLS.Z];
  }
  return 30;
}

function updateMLOverlay(sample) {
  const overlay = document.getElementById('ml-overlay');
  if (!state.currentRun || state.currentRun.mode !== 'ML') {
    overlay.style.display = 'none';
    return;
  }
  overlay.style.display = '';
  if (!sample) return;

  const baselineIgn = computeBaselineIgnitionAlt(state.currentRun);
  const mlCorrection = sample.mlCorrection || 0;
  const adjustedIgn = baselineIgn + mlCorrection;

  document.getElementById('ml-base-ign').textContent = baselineIgn.toFixed(1) + ' m';
  const corrEl = document.getElementById('ml-ign-corr');
  corrEl.textContent = (mlCorrection >= 0 ? '+' : '') + mlCorrection.toFixed(2) + ' m';
  corrEl.style.color = Math.abs(mlCorrection) < 0.5 ? 'var(--text)' : mlCorrection > 0 ? 'var(--accent)' : 'var(--warning)';
  document.getElementById('ml-ign-adj').textContent = adjustedIgn.toFixed(1) + ' m';
}

// ═══════════════════════════════════════════════════════════════
// MAIN ANIMATION LOOP
// ═══════════════════════════════════════════════════════════════
const clock = new THREE.Clock();
let frameCount = 0, fpsTime = 0;
let lastTrailTime = -1;
let cameraShake = new THREE.Vector3();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);

  // FPS counter
  frameCount++;
  fpsTime += dt;
  if (fpsTime >= 0.5) {
    document.getElementById('fps-counter').textContent = Math.round(frameCount / fpsTime) + ' FPS';
    frameCount = 0; fpsTime = 0;
  }

  // Advance time
  if (state.playing && state.currentTrajectory) {
    const tMax = state.currentTrajectory[state.currentTrajectory.length - 1][COLS.T];
    state.currentTime += dt * state.speed;
    if (state.currentTime >= tMax) {
      state.currentTime = tMax;
      state.playing = false;
      document.getElementById('play-btn').innerHTML = '&#9654;';

      // Trigger crash if needed
      if (state.currentRun && !state.currentRun.success && state.currentRun.status_label === 'CRASH') {
        const s = sampleTrajectory(state.currentTrajectory, tMax);
        if (s) triggerCrash(new THREE.Vector3(
          s.x * state.lateralScale,
          s.z * state.verticalScale,
          s.y * state.lateralScale
        ));
      }
    }
    updateTimeDisplay();
  }

  // Sample trajectory
  const sample = state.currentTrajectory ? sampleTrajectory(state.currentTrajectory, state.currentTime) : null;

  if (sample) {
    // Map coordinate system: sim X→three X, sim Y→three Z, sim Z→three Y (altitude)
    const px = sample.x * state.lateralScale;
    const py = sample.z * state.verticalScale; // altitude = three Y
    const pz = sample.y * state.lateralScale;

    rocketGroup.position.set(px, py, pz);

    // Apply quaternion: sim Z-up → Three.js Y-up via proper conjugation C*q*C⁻¹.
    // C = Rx(+90°): maps sim +Z (up) → Three.js +Y (up), preserving handedness.
    const simQ = sample.quat;
    const C = new THREE.Quaternion(Math.SQRT1_2, 0, 0, Math.SQRT1_2); // Rx(+90°)
    const finalQ = C.clone().multiply(simQ).multiply(new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2)); // C*q*C⁻¹
    rocketGroup.quaternion.copy(finalQ);

    // Motor-on: detect by Phase column (0=ascent burn, 3=descent burn)
    const isBurning = sample.isBurning;
    animateFlame(state.currentTime, isBurning);

    // Particle emission
    if (isBurning && state.playing) {
      const nozzleWorld = new THREE.Vector3(0, -0.3, 0);
      nozzleWorld.applyQuaternion(rocketGroup.quaternion);
      nozzleWorld.add(rocketGroup.position);
      for (let p = 0; p < 6; p++) emitParticle(nozzleWorld, rocketGroup.quaternion);
    }

    // Trail
    if (state.showTrail) {
      const trailInterval = 0.05;
      if (sample.t - lastTrailTime > trailInterval) {
        const run = state.currentRun;
        const c = run ? new THREE.Color(run.color) : new THREE.Color(0x00d4ff);
        addTrailPoint(px, py, pz, c);
        lastTrailTime = sample.t;
      }
    }
    trailLine.visible = state.showTrail;

    // Landing prediction
    if (py > 2 && sample.vz < 0) {
      const tti = py / Math.abs(sample.vz * state.verticalScale);
      const predX = px + sample.vx * state.lateralScale * tti;
      const predZ = pz + sample.vy * state.lateralScale * tti;
      predRing.position.set(predX, 0.1, predZ);
      predRing.visible = true;
      const pulse = 0.8 + 0.2 * Math.sin(state.currentTime * 4);
      predRing.scale.set(pulse, pulse, pulse);
    } else {
      predRing.visible = false;
    }

    // Sound
    if (state.soundEnabled) {
      if (!audioCtx) initAudio();
      const spd = Math.sqrt(sample.vx**2 + sample.vy**2 + sample.vz**2);
      updateAudio(isBurning, spd);
    }

    // Update HUD + ML overlay
    updateHUD(sample, isBurning);
    updateMLOverlay(sample);

    // Fault effects (particles, geometry, toast)
    updateFaultEffects(sample, dt);
  }

  // Update particles
  updateParticles(dt);
  updateFaultParticles(dt);
  updateFaultToast(dt);

  // Crash effect
  if (crashActive) {
    crashTime += dt;
    if (crashTime > 2.0) {
      crashActive = false;
      crashFlash.material.opacity = 0;
    } else {
      // Flash
      crashFlash.material.opacity = Math.max(0, 0.8 * (1 - crashTime / 0.5));
      if (sample) crashFlash.position.copy(rocketGroup.position);

      // Camera shake
      if (crashTime < 1.0) {
        const intensity = 2 * (1 - crashTime);
        cameraShake.set(
          Math.sin(crashTime * 50) * intensity,
          Math.sin(crashTime * 60 + 1) * intensity * 0.5,
          Math.sin(crashTime * 45 + 2) * intensity
        );
      } else {
        cameraShake.set(0, 0, 0);
      }

      // Update crash debris particles
      for (const cp of crashParticles) {
        cp.vel.y -= 9.81 * dt;
        cp.pos.add(cp.vel.clone().multiplyScalar(dt));
        cp.age += dt;
        if (cp.pos.y < 0) { cp.pos.y = 0; cp.vel.y *= -0.3; cp.vel.multiplyScalar(0.5); }
      }

      // Render crash particles as temporary emits
      if (crashTime < 0.3) {
        for (const cp of crashParticles) {
          if (cp.age < 0.3 && Math.random() < 0.5) {
            const idx = nextParticle;
            nextParticle = (nextParticle + 1) % MAX_PARTICLES;
            particlePositions[idx * 3] = cp.pos.x;
            particlePositions[idx * 3 + 1] = cp.pos.y;
            particlePositions[idx * 3 + 2] = cp.pos.z;
            particleVelocities[idx].copy(cp.vel).multiplyScalar(0.2);
            particleAges[idx] = 0;
            particleAlive[idx] = 1;
            particleColors[idx * 4] = 1;
            particleColors[idx * 4 + 1] = 0.5;
            particleColors[idx * 4 + 2] = 0.1;
            particleColors[idx * 4 + 3] = 1;
            particleSizes[idx] = 0.3 + Math.random() * 0.5;
          }
        }
      }
    }
  }

  // Camera update
  const target = rocketGroup.position.clone();
  if (state.cameraMode === 'chase') {
    orbitControls.enabled = false;
    const offset = new THREE.Vector3(12, 6, 15);
    const idealPos = target.clone().add(offset);
    camera.position.lerp(idealPos, 2 * dt);
    camera.position.add(cameraShake);
    camera.lookAt(target);
  } else if (state.cameraMode === 'orbit') {
    orbitControls.enabled = true;
    orbitControls.target.lerp(target, 3 * dt);
    orbitControls.update();
  } else {
    orbitControls.enabled = true;
    orbitControls.update();
  }

  // Render
  if (state.bloomEnabled) {
    composer.render();
  } else {
    renderer.render(scene, camera);
  }
}

// ═══════════════════════════════════════════════════════════════
// RESIZE HANDLER
// ═══════════════════════════════════════════════════════════════
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.resolution.set(w, h);
});

// ═══════════════════════════════════════════════════════════════
// INIT — hide loading, start loop, auto-load first demo
// ═══════════════════════════════════════════════════════════════
setTimeout(() => {
  document.getElementById('loading').classList.add('hidden');
  setTimeout(() => document.getElementById('loading').remove(), 600);
}, 800);

loadRun('demo_01_baseline');
animate();

console.log('%c HexaVisual Pro (Demo) %c Loaded successfully ', 'background:#0a0a14;color:#00d4ff;font-weight:bold;padding:4px 8px;border-radius:4px 0 0 4px', 'background:#1a1a2e;color:#b794f6;padding:4px 8px;border-radius:0 4px 4px 0');
</script>
</body>
</html>
'''

# Write the HTML file
with open('hexavisual_pro_demo.html', 'w', encoding='utf-8') as f:
    f.write(html)

size = os.path.getsize('hexavisual_pro_demo.html')
print(f'Created hexavisual_pro_demo.html: {size} bytes ({size/1024:.1f} KB)')
