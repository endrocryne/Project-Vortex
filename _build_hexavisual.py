"""
Build script for the standalone flight data viewer with embedded trajectory data.
"""
import os
import csv

# Read trajectory data block
with open('_traj_data_js.txt', 'r') as f:
    traj_data_block = f.read()

# Read secondary body (payload satellite) altitude data from CSV
secondary_body_js = ''
csv_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Altitude_Raw_Data.csv')
if os.path.exists(csv_path):
    sec_points = []
    sep_time = 5.8
    with open(csv_path, 'r') as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            t = float(row['time_s'])
            sec_alt_str = row['Flight 1_secondary_Altitude'].strip()
            if sec_alt_str == '' or t < sep_time:
                continue
            alt = float(sec_alt_str)
            # Synthetic X/Y: drift downwind at 0.5 m/s from separation point
            dt = t - sep_time
            x = round(dt * 0.5, 2)   # downwind drift in sim X
            y = 0.0                   # no lateral drift in sim Y
            sec_points.append(f'[{t},{x},{y},{alt}]')
    if sec_points:
        secondary_body_js = 'const SECONDARY_BODY = [\n' + ',\n'.join(sec_points) + '\n];\nconst SEC_SEPARATION_TIME = ' + str(sep_time) + ';\n'
        print(f'  Secondary body data: {len(sec_points)} points from t={sep_time}s')
    else:
        secondary_body_js = '// No secondary body data found\n'
        print('  Warning: CSV found but no secondary body data extracted')
else:
    secondary_body_js = '// Secondary body CSV not found — skipping\n'
    print(f'  Warning: Secondary body CSV not found at {csv_path}')

# Build the HTML
html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flight Data Viewer</title>
<style>
/* ═══════════════════════════════════════════════════════════════
   CSS — Plain technical viewer
   ═══════════════════════════════════════════════════════════════ */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --glass-bg:rgba(22,24,28,0.96);
  --glass-border:rgba(255,255,255,0.08);
  --glass-blur:0px;
  --accent:#6ea0d8;
  --accent2:#6ea0d8;
  --success:#6a9d74;
  --warning:#b78a55;
  --danger:#b66a6a;
  --text:#d7dce3;
  --text-dim:#96a0ad;
  --mono:'Consolas','SF Mono','Fira Code',monospace;
  --sans:'Segoe UI',system-ui,sans-serif;
  --sidebar-w:300px;
  --hud-w:280px;
  --control-h:64px;
  --body-bg:#111317;
  --card-bg:#171a1f;
  --card-border:rgba(255,255,255,0.08);
  --card-hover-bg:#1c2027;
  --card-hover-border:rgba(255,255,255,0.14);
  --speed-option-bg:#1a1d22;
  --select-bg:#1b1f26;
  --select-border:rgba(255,255,255,0.12);
  --range-bg:rgba(255,255,255,0.10);
  --toggle-off-bg:rgba(255,255,255,0.10);
  --scrollbar-thumb:rgba(255,255,255,0.18);
  --loading-bg:#111317;
  --upload-border:rgba(255,255,255,0.12);
  --divider:rgba(255,255,255,0.08);
}
/* ── Light theme ── */
body.light{
  --glass-bg:rgba(244,246,248,0.97);
  --glass-border:rgba(0,0,0,0.08);
  --text:#1d232b;
  --text-dim:#5f6976;
  --body-bg:#d9dde2;
  --card-bg:#eef1f4;
  --card-border:rgba(0,0,0,0.08);
  --card-hover-bg:#e5e9ee;
  --card-hover-border:rgba(0,0,0,0.14);
  --speed-option-bg:#eef1f4;
  --select-bg:#e7ebf0;
  --select-border:rgba(0,0,0,0.12);
  --range-bg:rgba(0,0,0,0.12);
  --toggle-off-bg:rgba(0,0,0,0.12);
  --scrollbar-thumb:rgba(0,0,0,0.18);
  --loading-bg:#d9dde2;
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
  border-radius:6px;
}

/* ──── SIDEBAR (left) ──── */
#sidebar{
  position:fixed;top:0;left:0;width:var(--sidebar-w);height:100vh;
  z-index:100;padding:16px 12px;overflow-y:auto;overflow-x:hidden;
  border-radius:0;
  scrollbar-width:thin;scrollbar-color:var(--scrollbar-thumb) transparent;
}
#sidebar-toggle{
  position:absolute;top:12px;right:12px;width:28px;height:28px;border-radius:8px;
  background:var(--select-bg);border:1px solid var(--select-border);color:var(--text);
  cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;
}
#sidebar-toggle:hover{background:var(--card-hover-bg);border-color:var(--card-hover-border)}
#sidebar::-webkit-scrollbar{width:5px}
#sidebar::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb);border-radius:3px}
#sidebar .section-title{font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:12px;text-transform:uppercase;color:var(--text-dim)}
body.sidebar-collapsed{--sidebar-w:58px}
body.sidebar-collapsed #sidebar{padding:12px 8px}
body.sidebar-collapsed #sidebar .section-title{font-size:10px;letter-spacing:1.2px;margin-top:34px;margin-bottom:0;writing-mode:vertical-rl;transform:rotate(180deg)}
body.sidebar-collapsed #demo-cards,
body.sidebar-collapsed .upload-btn{display:none}
body.sidebar-collapsed #sidebar-toggle{right:50%;transform:translateX(50%)}

/* Demo cards */
.demo-card{
  padding:9px 10px;margin-bottom:6px;cursor:pointer;
  border-radius:4px;border:1px solid var(--card-border);
  background:var(--card-bg);transition:border-color 0.15s, background 0.15s;
}
.demo-card:hover{background:var(--card-hover-bg);border-color:var(--card-hover-border);transform:none}
.demo-card.active{border-color:var(--accent);background:var(--card-hover-bg);box-shadow:none}
.demo-card .card-header{display:flex;align-items:center;justify-content:space-between;gap:8px}
.demo-card .card-title{font-size:12px;font-weight:600}
.demo-card .badge{font-size:9px;padding:2px 6px;border-radius:3px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase}
.badge-opt{background:rgba(255,255,255,0.06);color:var(--text-dim)}
.badge-ml{background:rgba(110,160,216,0.18);color:var(--accent)}
body.light .badge-ml{color:#446c99}
.badge-success{background:rgba(106,157,116,0.18);color:var(--success)}
.badge-fail{background:rgba(183,138,85,0.18);color:var(--warning)}
.badge-crash{background:rgba(182,106,106,0.18);color:var(--danger)}

/* ──── HUD (top-left, offset from sidebar) ──── */
#hud{
  position:fixed;top:16px;left:calc(var(--sidebar-w) + 16px);
  width:var(--hud-w);z-index:90;padding:14px 16px;
}
#hud.minimized{width:200px}
#hud h2{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:10px;font-weight:600}
.hud-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:12px}
.hud-row .label{color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:0.5px}
.hud-row .value{font-family:var(--mono);font-weight:600;font-size:13px}
.hud-section-title{
  margin:8px 0 6px;color:var(--text-dim);font-size:10px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;
}
.variables-grid{
  display:grid;grid-template-columns:minmax(82px,1fr) minmax(88px,1fr) minmax(88px,1fr);
  gap:4px 10px;font-size:11px;align-items:center;
}
.variables-grid .head{
  color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:0.7px;
}
.variables-grid .label{
  color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:0.5px;
}
.variables-grid .value{font-family:var(--mono);font-weight:600;font-size:12px}
.config-grid{
  display:grid;grid-template-columns:minmax(96px,1fr) minmax(110px,1fr);
  gap:4px 12px;font-size:11px;align-items:center;
}
.config-grid .label{
  color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:0.5px;
}
.config-grid .value{font-family:var(--mono);font-weight:600;font-size:12px;text-align:right}
.hud-divider{height:1px;background:var(--divider);margin:6px 0}
.panel-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.panel-head h2{margin:0 !important}
.panel-toggle{
  width:24px;height:24px;border-radius:6px;background:var(--select-bg);border:1px solid var(--select-border);
  color:var(--text);cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;
}
.panel-toggle:hover{background:var(--card-hover-bg);border-color:var(--card-hover-border)}
.panel-body{overflow:hidden}
#hud.minimized .panel-body,
#ml-overlay.minimized .panel-body{display:none}
#engine-status{
  display:inline-block;padding:2px 10px;border-radius:4px;font-size:10px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;margin-top:4px;
}

/* ──── ML OVERLAY (top-right) ──── */
#ml-overlay{
  position:fixed;top:16px;right:16px;width:260px;z-index:90;padding:14px 16px;
}
#ml-overlay.minimized{width:200px}
#ml-overlay h2{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:10px;font-weight:600}
.ml-row{display:flex;justify-content:space-between;padding:3px 0;font-size:11px}
.ml-row .label{color:var(--text-dim)}
.ml-row .value{font-family:var(--mono);font-weight:600}

/* ──── FAULT TOAST NOTIFICATION ──── */
#fault-toast{
  display:none !important;
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
#controls-bar button.loop-active{background:rgba(76,175,80,0.18);border-color:rgba(76,175,80,0.45);color:var(--success)}
#timeline-container{flex:1;display:flex;align-items:center;gap:10px}
#timeline-wrap{position:relative;flex:1;display:flex;align-items:center}
#timeline-markers{
  position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);
  height:18px;pointer-events:none;
}
.timeline-marker{position:absolute;top:50%;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:2px}
.timeline-marker::before{
  content:'';width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 8px rgba(0,212,255,0.35);border:1px solid rgba(255,255,255,0.45);
}
.timeline-marker.touchdown::before{background:var(--success);box-shadow:0 0 8px rgba(76,175,80,0.35)}
.timeline-marker.failure::before{background:var(--danger);box-shadow:0 0 8px rgba(244,67,54,0.35)}
.timeline-marker span{
  transform:translateY(-12px);font-size:9px;letter-spacing:0.5px;text-transform:uppercase;
  color:var(--text-dim);font-family:var(--mono);white-space:nowrap;
}
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

/* ──── ML CORRECTION BAR ──── */
#correction-bar{
  position:fixed;top:50px;left:50%;transform:translateX(-50%);z-index:90;width:420px;
  padding:10px 14px;border-radius:14px;
}
#correction-bar.hidden{display:none}
.corr-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.corr-head .title{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--text-dim);font-weight:700}
.corr-head .value{font-family:var(--mono);font-size:13px;font-weight:700}
.corr-track{
  position:relative;height:12px;border-radius:999px;background:linear-gradient(90deg,rgba(255,152,0,0.18),rgba(255,255,255,0.07),rgba(0,212,255,0.18));
  overflow:hidden;
}
.corr-center{position:absolute;left:50%;top:0;bottom:0;width:2px;background:rgba(255,255,255,0.55);transform:translateX(-50%)}
.corr-fill{
  position:absolute;top:1px;bottom:1px;border-radius:999px;transition:left 0.2s ease,width 0.2s ease,background 0.2s ease;
}
.corr-scale{display:flex;justify-content:space-between;margin-top:6px;color:var(--text-dim);font-size:10px;font-family:var(--mono)}

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
#loading h1{font-size:22px;font-weight:700;margin-bottom:8px;color:var(--text)}
#loading p{color:var(--text-dim);font-size:13px}
.loader{width:40px;height:40px;border:3px solid var(--range-bg);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;margin-top:20px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<!-- Loading screen -->
<div id="loading">
  <h1>Flight Data Viewer</h1>
  <p>Loading trajectory data...</p>
  <div class="loader"></div>
</div>

<!-- Sidebar -->
<div id="sidebar" class="glass">
  <button id="sidebar-toggle" onclick="toggleSidebar()" title="Collapse sidebar">&#9664;</button>
  <div class="section-title">Runs</div>
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
  <div class="panel-head">
    <h2>Variables</h2>
    <button id="hud-toggle" class="panel-toggle" onclick="toggleHud()" title="Minimize variables">&#8722;</button>
  </div>
  <div class="panel-body">
  <div class="hud-row"><span class="label">Time</span><span class="value" id="h-time">0.000s</span></div>
  <div class="hud-row"><span class="label">Flight Phase</span><span class="value" id="h-phase">Pad Idle</span></div>
  <div class="hud-divider"></div>
  <div class="hud-section-title">Actual vs Measured</div>
  <div class="variables-grid">
    <div class="head"></div><div class="head">Actual</div><div class="head">Measured</div>
    <div class="label">Altitude</div><div class="value" id="h-alt-actual">0.00 m</div><div class="value" id="h-alt-measured">0.00 m</div>
    <div class="label">Vert. Vel</div><div class="value" id="h-vvel-actual">0.00 m/s</div><div class="value" id="h-vvel-measured">0.00 m/s</div>
    <div class="label">Horiz. Vel</div><div class="value" id="h-hvel-actual">0.00 m/s</div><div class="value" id="h-hvel-measured">0.00 m/s</div>
    <div class="label">Speed</div><div class="value" id="h-speed-actual">0.00 m/s</div><div class="value" id="h-speed-measured">0.00 m/s</div>
    <div class="label">Mass</div><div class="value" id="h-mass-actual">1.282 kg</div><div class="value" id="h-mass-measured">1.282 kg</div>
    <div class="label">Wind</div><div class="value" id="h-wind-actual">0.00 m/s</div><div class="value" id="h-wind-measured">0.00 m/s</div>
    <div class="label">TTI</div><div class="value" id="h-tti-actual">—</div><div class="value" id="h-tti-measured">—</div>
  </div>
  <div class="hud-divider"></div>
  <div class="hud-section-title">Estimator</div>
  <div class="config-grid">
    <div class="label">EKF Mass</div><div class="value" id="h-ekf-mass">— kg</div>
    <div class="label">EKF Drag Cd</div><div class="value" id="h-ekf-cd">—</div>
  </div>
  <div class="hud-divider"></div>
  <div class="hud-section-title">Configuration</div>
  <div class="config-grid">
    <div class="label">Mode</div><div class="value" id="cfg-mode">—</div>
    <div class="label">Nominal Ign.</div><div class="value" id="cfg-ign-nom">— m</div>
    <div class="label">Effective Ign.</div><div class="value" id="cfg-ign-eff">— m</div>
    <div class="label">Dry Mass</div><div class="value" id="cfg-dry-mass">— kg</div>
    <div class="label">Propellant</div><div class="value" id="cfg-prop-mass">— kg</div>
    <div class="label">Thrust Avg</div><div class="value" id="cfg-thrust">— N</div>
    <div class="label">Drag Cd</div><div class="value" id="cfg-cd">—</div>
    <div class="label">Air Density</div><div class="value" id="cfg-rho">— kg/m3</div>
    <div class="label">Wind</div><div class="value" id="cfg-wind">— m/s</div>
  </div>
  <div class="hud-divider"></div>
  <div id="engine-status" style="background:rgba(255,255,255,0.08);color:var(--text-dim)">IDLE</div>
  </div>
</div>

<!-- ML Overlay -->
<div id="ml-overlay" class="glass" style="display:none">
  <div class="panel-head">
    <h2>ML Flight Computer</h2>
    <button id="ml-toggle" class="panel-toggle" onclick="toggleMlOverlay()" title="Minimize ML overlay">&#8722;</button>
  </div>
  <div class="panel-body">
  <div class="ml-row"><span class="label">Mode</span><span class="value" id="ml-mode">—</span></div>
  <div class="ml-row"><span class="label">Faults</span><span class="value" id="ml-faults">None</span></div>
  <div class="hud-divider"></div>
  <div class="ml-row"><span class="label">Baseline Ign. Alt.</span><span class="value" id="ml-base-ign">— m</span></div>
  <div class="ml-row"><span class="label">ML Ign. Correction</span><span class="value" id="ml-ign-corr">— m</span></div>
  <div class="ml-row"><span class="label">Adjusted Ign. Alt.</span><span class="value" id="ml-ign-adj">— m</span></div>
  <div class="hud-divider"></div>
  <div class="ml-row"><span class="label">Landing Vel.</span><span class="value" id="ml-lvel">— m/s</span></div>
  <div class="ml-row"><span class="label">Landing Dist.</span><span class="value" id="ml-ldist">— m</span></div>
  </div>
</div>

<!-- Fault Toast Notification -->
<div id="fault-toast"></div>

<!-- Camera mode indicator -->
<div id="camera-mode" class="glass">Chase Cam</div>

<!-- ML correction bar -->
<div id="correction-bar" class="glass hidden">
  <div class="corr-head">
    <span class="title">ML Ignition Correction</span>
    <span class="value" id="corr-value">+0.00 m</span>
  </div>
  <div class="corr-track">
    <div class="corr-center"></div>
    <div class="corr-fill" id="corr-fill"></div>
  </div>
  <div class="corr-scale"><span>-6.0 m</span><span>0.0 m</span><span>+8.0 m</span></div>
</div>

<!-- FPS counter -->
<div id="fps-counter" class="glass">— FPS</div>

<!-- Controls bar -->
<div id="controls-bar" class="glass">
  <button title="Skip back 5s" onclick="skipTime(-5)">&#9194;</button>
  <button title="Step back" onclick="stepFrame(-1)">&#9198;</button>
  <button class="primary" id="play-btn" onclick="togglePlay()" title="Play / Pause">&#9654;</button>
  <button id="loop-btn" onclick="toggleLoop()" title="Simulation loop off">Loop</button>
  <button title="Step forward" onclick="stepFrame(1)">&#9197;</button>
  <button title="Skip forward 5s" onclick="skipTime(5)">&#9193;</button>
  <div id="timeline-container">
    <div id="timeline-wrap">
      <div id="timeline-markers"></div>
      <input type="range" id="timeline" min="0" max="1000" value="0" step="1">
    </div>
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
            <option value="14">14</option><option value="15">15</option><option value="16">16</option><option value="17">17</option><option value="18" selected>18</option><option value="19">19</option>
          </select>
        </div>
      </div>
      <button onclick="loadSatTile()" style="width:100%;padding:5px;background:var(--accent);color:#000;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600;margin-bottom:8px">Load Imagery</button>
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
''' + '\n' + traj_data_block + '\n' + secondary_body_js + '\n' + r'''
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
  loopPlayback: false,
  loopRestartRemaining: 0,
  currentTime: 0,
  speed: 1,
  currentRun: null,
  currentTrajectory: null,
  currentEvents: [],
  cameraMode: 'chase', // chase | orbit | free
  lateralScale: 1,
  verticalScale: 1,
  showTrail: true,
  bloomEnabled: true,
  soundEnabled: false,
  satelliteEnabled: false,
  lightMode: false,
  sidebarCollapsed: false,
  hudMinimized: false,
  mlMinimized: false,
};

// Expose state for inline HTML event handlers (onchange="state.xxx=...")
window.state = state;

function applyChromeState() {
  document.body.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  document.getElementById('hud').classList.toggle('minimized', state.hudMinimized);
  document.getElementById('ml-overlay').classList.toggle('minimized', state.mlMinimized);
  document.getElementById('sidebar-toggle').innerHTML = state.sidebarCollapsed ? '&#9654;' : '&#9664;';
  document.getElementById('hud-toggle').innerHTML = state.hudMinimized ? '&#43;' : '&#8722;';
  document.getElementById('ml-toggle').innerHTML = state.mlMinimized ? '&#43;' : '&#8722;';
}

function clampValue(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function formatSigned(value, digits = 2, suffix = '') {
  if (!isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}${suffix}`;
}

function phaseName(phase) {
  if (phase === 0) return 'Powered Ascent';
  if (phase === 1) return 'Coast to Apogee';
  if (phase === 2) return 'Freefall';
  if (phase === 3) return 'Powered Descent';
  if (phase === 4) return 'Touchdown';
  return 'Unknown';
}

function sensorNoise(sample, seed, scale) {
  return (
    Math.sin(sample.t * (0.7 + seed * 0.11) + seed * 1.37) +
    0.5 * Math.cos(sample.t * (1.4 + seed * 0.07) + seed * 0.83)
  ) * scale;
}

function computeMeasuredTelemetry(sample, run) {
  const faultBias = run ? run.fault_intensity || 0 : 0;
  const windFault = run && run.fault_types.includes('WIND_GUST') ? 1 : 0;
  const dragFault = run && run.fault_types.includes('DRAG_CHANGE') ? 1 : 0;
  const massFault = run && run.fault_types.includes('MASS_LOSS') ? 1 : 0;
  const windMag = Math.sqrt((sample.windX || 0) ** 2 + (sample.windY || 0) ** 2);
  const ttiActual = sample.vz < 0 && sample.z > 0 ? sample.z / Math.abs(sample.vz) : NaN;

  const altitudeMeasured = Math.max(0, sample.z + sensorNoise(sample, 1, 0.45 + 0.55 * faultBias));
  const verticalMeasured = sample.vz + sensorNoise(sample, 2, 0.18 + 0.30 * faultBias) - dragFault * 0.12 * sample.vz;
  const horizontalActual = Math.sqrt(sample.vx * sample.vx + sample.vy * sample.vy);
  const horizontalMeasured = Math.max(0, horizontalActual + sensorNoise(sample, 3, 0.16 + 0.25 * (faultBias + 0.3 * windFault)));
  const speedActual = Math.sqrt(sample.vx * sample.vx + sample.vy * sample.vy + sample.vz * sample.vz);
  const speedMeasured = Math.max(0, speedActual + sensorNoise(sample, 4, 0.18 + 0.25 * faultBias));
  const massMeasured = Math.max(run ? run.dry_mass : 0.1, sample.mass + sensorNoise(sample, 5, 0.002 + 0.006 * massFault + 0.002 * faultBias));
  const windMeasured = Math.max(0, windMag + sensorNoise(sample, 6, 0.10 + 0.45 * windFault + 0.12 * faultBias));
  const ttiMeasured = verticalMeasured < -0.01 && altitudeMeasured > 0 ? altitudeMeasured / Math.abs(verticalMeasured) : NaN;

  const propellantMass = run ? run.propellant_mass : 0;
  const burnProgress = run && propellantMass > 0 ? clampValue((run.dry_mass + propellantMass - sample.mass) / propellantMass, 0, 1) : 0;
  const ekfMass = sample.mass + sensorNoise(sample, 7, 0.003 + 0.004 * faultBias) + (massMeasured - sample.mass) * 0.35 * (1 - burnProgress * 0.4);
  const ekfCd = (run ? run.drag_coefficient : 0.5) + sensorNoise(sample, 8, 0.004 + 0.02 * dragFault + 0.006 * faultBias) + (sample.faultMag || 0) * 0.03 * dragFault;

  return {
    altitudeActual: sample.z,
    altitudeMeasured,
    verticalActual: sample.vz,
    verticalMeasured,
    horizontalActual,
    horizontalMeasured,
    speedActual,
    speedMeasured,
    massActual: sample.mass,
    massMeasured,
    windActual: windMag,
    windMeasured,
    ttiActual,
    ttiMeasured,
    ekfMass,
    ekfCd,
  };
}

function setDualValue(actualId, measuredId, actualValue, measuredValue, unit, digits = 2) {
  document.getElementById(actualId).textContent = isFinite(actualValue) ? `${actualValue.toFixed(digits)} ${unit}`.trim() : '—';
  document.getElementById(measuredId).textContent = isFinite(measuredValue) ? `${measuredValue.toFixed(digits)} ${unit}`.trim() : '—';
}

function updateConfiguration(run) {
  if (!run) return;
  const dryMass = typeof run.dry_mass === 'number' ? run.dry_mass : 1.219;
  const propellantMass = typeof run.propellant_mass === 'number' ? run.propellant_mass : 0.191;
  const thrustAverage = typeof run.thrust_average === 'number' ? run.thrust_average : 89.0;
  const nominalIgnition = typeof run.nominal_ignition_altitude === 'number' ? run.nominal_ignition_altitude : 36.11;
  const effectiveIgnition = typeof run.effective_ignition_altitude === 'number' ? run.effective_ignition_altitude : nominalIgnition;
  const dragCoefficient = typeof run.drag_coefficient === 'number' ? run.drag_coefficient : 0.48;
  const airDensity = typeof run.air_density === 'number' ? run.air_density : 1.18;
  const windSpeed = typeof run.wind_speed === 'number' ? run.wind_speed : 0.0;

  document.getElementById('cfg-mode').textContent = run.mode;
  document.getElementById('cfg-ign-nom').textContent = `${nominalIgnition.toFixed(2)} m`;
  document.getElementById('cfg-ign-eff').textContent = `${effectiveIgnition.toFixed(2)} m`;
  document.getElementById('cfg-dry-mass').textContent = `${dryMass.toFixed(3)} kg`;
  document.getElementById('cfg-prop-mass').textContent = `${propellantMass.toFixed(3)} kg`;
  document.getElementById('cfg-thrust').textContent = `${thrustAverage.toFixed(1)} N`;
  document.getElementById('cfg-cd').textContent = dragCoefficient.toFixed(2);
  document.getElementById('cfg-rho').textContent = `${airDensity.toFixed(2)} kg/m3`;
  document.getElementById('cfg-wind').textContent = `${windSpeed.toFixed(1)} m/s`;
}

function detectTimelineEvents(traj, run) {
  if (!traj || !traj.length) return [];
  const events = [{ label: 'Liftoff', time: traj[0][COLS.T], className: 'liftoff' }];
  const burnoutIndex = traj.findIndex(row => row[COLS.PH] > 0);
  if (burnoutIndex >= 0) events.push({ label: 'Burnout', time: traj[burnoutIndex][COLS.T], className: 'burnout' });

  let apogeeRow = traj[0];
  for (const row of traj) {
    if (row[COLS.Z] > apogeeRow[COLS.Z]) apogeeRow = row;
  }
  events.push({ label: 'Apogee', time: apogeeRow[COLS.T], className: 'apogee' });

  const ignitionIndex = traj.findIndex(row => row[COLS.PH] === 3);
  if (ignitionIndex >= 0) events.push({ label: 'Ignition', time: traj[ignitionIndex][COLS.T], className: 'ignition' });

  const touchdownRow = traj[traj.length - 1];
  events.push({ label: 'Touchdown', time: touchdownRow[COLS.T], className: run && run.success ? 'touchdown' : 'failure' });
  return events;
}

function renderTimelineMarkers() {
  const container = document.getElementById('timeline-markers');
  container.innerHTML = '';
  const traj = state.currentTrajectory;
  if (!traj || traj.length < 2) return;
  const tMax = traj[traj.length - 1][COLS.T];
  for (const event of state.currentEvents) {
    const marker = document.createElement('div');
    marker.className = `timeline-marker ${event.className || ''}`.trim();
    marker.style.left = `${tMax > 0 ? (event.time / tMax) * 100 : 0}%`;
    marker.title = `${event.label} @ ${event.time.toFixed(2)}s`;
    const label = document.createElement('span');
    label.textContent = event.label;
    marker.appendChild(label);
    container.appendChild(marker);
  }
}

function updateCorrectionBar(sample) {
  const bar = document.getElementById('correction-bar');
  if (!state.currentRun || state.currentRun.mode !== 'ML') {
    bar.classList.add('hidden');
    return;
  }

  const correction = sample ? (sample.mlCorrection || 0) : 0;
  const minCorrection = -6;
  const maxCorrection = 8;
  const span = maxCorrection - minCorrection;
  const centerPercent = ((0 - minCorrection) / span) * 100;
  const currentPercent = clampValue(((correction - minCorrection) / span) * 100, 0, 100);
  const left = Math.min(centerPercent, currentPercent);
  const width = Math.max(0.8, Math.abs(currentPercent - centerPercent));
  const fill = document.getElementById('corr-fill');

  fill.style.left = `${left}%`;
  fill.style.width = `${width}%`;
  fill.style.background = correction >= 0 ? 'rgba(0,212,255,0.8)' : 'rgba(255,152,0,0.8)';
  document.getElementById('corr-value').textContent = formatSigned(correction, 2, ' m');
  document.getElementById('corr-value').style.color = correction >= 0 ? 'var(--accent)' : 'var(--warning)';
  bar.classList.remove('hidden');
}

window.toggleSidebar = function() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  applyChromeState();
};

window.toggleHud = function() {
  state.hudMinimized = !state.hudMinimized;
  applyChromeState();
};

window.toggleMlOverlay = function() {
  state.mlMinimized = !state.mlMinimized;
  applyChromeState();
};

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
let activeSatRequest = { lat: 51.5074, lon: -0.1278, zoom: 18, radius: 1 };

function satelliteTileUrl(zoom, tileY, tileX) {
  return `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${tileY}/${tileX}`;
}

function fetchSatelliteImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function buildSatelliteMosaic(lat, lon, zoom, radius = 1) {
  const n = Math.pow(2, zoom);
  const tileX = Math.floor((lon + 180) / 360 * n);
  const latRad = lat * Math.PI / 180;
  const tileY = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
  const tileSize = 256;
  const mosaicSize = radius * 2 + 1;
  const canvas = document.createElement('canvas');
  canvas.width = tileSize * mosaicSize;
  canvas.height = tileSize * mosaicSize;
  const ctx = canvas.getContext('2d');

  const tasks = [];
  for (let row = -radius; row <= radius; row++) {
    for (let col = -radius; col <= radius; col++) {
      const px = (col + radius) * tileSize;
      const py = (row + radius) * tileSize;
      tasks.push(
        fetchSatelliteImage(satelliteTileUrl(zoom, tileY + row, tileX + col)).then((img) => ({ img, px, py }))
      );
    }
  }

  const results = await Promise.allSettled(tasks);
  let drawn = 0;
  for (const result of results) {
    if (result.status !== 'fulfilled') continue;
    const { img, px, py } = result.value;
    ctx.drawImage(img, px, py, tileSize, tileSize);
    drawn += 1;
  }
  if (!drawn) throw new Error('No satellite tiles loaded');

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  tex.needsUpdate = true;
  return tex;
}

async function applySatelliteTexture(request) {
  const tex = await buildSatelliteMosaic(request.lat, request.lon, request.zoom, request.radius);
  satelliteTex = tex;
  groundMat.map = tex;
  groundMat.emissiveMap = tex;
  groundMat.needsUpdate = true;
}

window.setSatellite = function(enabled, overrideRequest) {
  state.satelliteEnabled = enabled;
  const ctrl = document.getElementById('sat-controls');
  if (ctrl) ctrl.style.display = enabled ? '' : 'none';
  if (enabled) {
    if (overrideRequest) {
      const changed = JSON.stringify(overrideRequest) !== JSON.stringify(activeSatRequest);
      activeSatRequest = overrideRequest;
      if (changed && satelliteTex) {
        satelliteTex.dispose();
        satelliteTex = null;
      }
    }
    if (satelliteTex) {
      groundMat.map = satelliteTex;
      groundMat.emissiveMap = satelliteTex;
      groundMat.needsUpdate = true;
    } else {
      applySatelliteTexture(activeSatRequest).catch(() => {
        console.warn('Satellite imagery failed (CORS/network). Try from a web server.');
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
  setSatellite(true, { lat, lon, zoom, radius: 1 });
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
// SECONDARY BODY — payload satellite with TVC thruster
// ═══════════════════════════════════════════════════════════════
const secondaryGroup = new THREE.Group();
{
  // Gold material for payload body
  const payloadMat = new THREE.MeshStandardMaterial({ color: 0xFFD700, roughness: 0.35, metalness: 0.7 });
  // Dark accent for nozzle/thruster
  const nozzleMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.6, metalness: 0.8 });
  // Solar panel blue
  const panelMat = new THREE.MeshStandardMaterial({ color: 0x2255AA, roughness: 0.3, metalness: 0.5, emissive: 0x112244, emissiveIntensity: 0.3 });

  // Main body — short cylinder (bus)
  const bus = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.3, 16), payloadMat);
  bus.position.y = 0.15;
  bus.castShadow = true;
  secondaryGroup.add(bus);

  // Nose cone on top
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.18, 16), payloadMat);
  nose.position.y = 0.39;
  nose.castShadow = true;
  secondaryGroup.add(nose);

  // Thruster nozzle at bottom
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.07, 0.08, 12), nozzleMat);
  nozzle.position.y = -0.04;
  secondaryGroup.add(nozzle);

  // Solar panel stubs (two flat boxes on sides)
  const panelGeo = new THREE.BoxGeometry(0.28, 0.02, 0.10);
  const panelL = new THREE.Mesh(panelGeo, panelMat);
  panelL.position.set(-0.26, 0.15, 0);
  panelL.castShadow = true;
  secondaryGroup.add(panelL);
  const panelR = new THREE.Mesh(panelGeo, panelMat);
  panelR.position.set(0.26, 0.15, 0);
  panelR.castShadow = true;
  secondaryGroup.add(panelR);
}
secondaryGroup.visible = false;
scene.add(secondaryGroup);

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

// Fault onset state
let lastFaultType = 0;

function updateFaultEffects(sample, dt) {
  if (!sample || !state.playing) {
    dragScaleTarget = 1.0;
    return;
  }
  const ft = Math.round(sample.faultType);
  const fm = sample.faultMag || 0;
  const rPos = rocketGroup.position;

  // Detect fault onset for visual effects only
  if (ft !== 0 && lastFaultType === 0) {
    lastFaultType = ft;
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

// ── Secondary body trail (orange) ──
const MAX_SEC_TRAIL = 2000;
const secTrailPositions = new Float32Array(MAX_SEC_TRAIL * 3);
const secTrailGeo = new THREE.BufferGeometry();
secTrailGeo.setAttribute('position', new THREE.BufferAttribute(secTrailPositions, 3));
const secTrailMat = new THREE.LineDashedMaterial({
  color: 0xff8800, transparent: true, opacity: 0.7,
  dashSize: 0.8, gapSize: 0.3
});
const secTrailLine = new THREE.Line(secTrailGeo, secTrailMat);
secTrailLine.frustumCulled = false;
scene.add(secTrailLine);
let secTrailCount = 0;

function resetSecTrail() { secTrailCount = 0; secTrailGeo.setDrawRange(0, 0); }

function addSecTrailPoint(x, y, z) {
  if (secTrailCount >= MAX_SEC_TRAIL) return;
  const i = secTrailCount * 3;
  secTrailPositions[i] = x; secTrailPositions[i + 1] = y; secTrailPositions[i + 2] = z;
  secTrailCount++;
  secTrailGeo.attributes.position.needsUpdate = true;
  secTrailGeo.setDrawRange(0, secTrailCount);
  secTrailLine.computeLineDistances();
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
  card.innerHTML = `
    <div class="card-header">
      <span class="card-title">${run.name}</span>
      <span>
        <span class="badge ${modeClass}">${run.mode}</span>
        <span class="badge ${statusClass}">${run.status_label}</span>
      </span>
    </div>
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
  state.currentEvents = detectTimelineEvents(state.currentTrajectory, run);
  state.currentTime = 0;
  state.loopRestartRemaining = 0;
  state.playing = false;
  document.getElementById('play-btn').innerHTML = '&#9654;';

  // Highlight card
  document.querySelectorAll('.demo-card').forEach(c => c.classList.remove('active'));
  document.querySelector(`.demo-card[data-id="${runId}"]`)?.classList.add('active');

  // Reset trail and crash
  resetTrail();
  resetSecTrail();
  secondaryGroup.visible = false;
  crashActive = false;
  crashFlash.material.opacity = 0;
  predRing.visible = false;
  lastTrailTime = -1;
  lastSecTrailTime = -1;
  lastFaultType = 0;
  dragScaleTarget = 1.0;
  dragScaleCurrent = 1.0;

  // ML overlay — show only for ML runs
  const isML = run.mode === 'ML';
  document.getElementById('ml-overlay').style.display = isML ? '' : 'none';
  if (isML) {
    document.getElementById('ml-mode').textContent = 'Online Correction';
    document.getElementById('ml-faults').textContent = run.fault_types.length ? run.fault_types.join(', ') : 'None';
    document.getElementById('ml-lvel').textContent = run.landing_velocity.toFixed(2) + ' m/s';
    document.getElementById('ml-ldist').textContent = run.landing_distance.toFixed(2) + ' m';
  }

  updateConfiguration(run);
  renderTimelineMarkers();
  updateCorrectionBar(null);

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

function syncLoopButton() {
  const loopBtn = document.getElementById('loop-btn');
  loopBtn.classList.toggle('loop-active', state.loopPlayback);
  loopBtn.title = state.loopPlayback ? 'Simulation loop on' : 'Simulation loop off';
}

window.toggleLoop = function() {
  state.loopPlayback = !state.loopPlayback;
  if (!state.loopPlayback) state.loopRestartRemaining = 0;
  syncLoopButton();
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
      dry_mass: 1.219,
      propellant_mass: 0.191,
      thrust_average: 89.0,
      nominal_ignition_altitude: 36.11,
      effective_ignition_altitude: 36.11,
      drag_coefficient: 0.48,
      air_density: 1.18,
      wind_speed: 0.0,
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
  const measured = computeMeasuredTelemetry(sample, state.currentRun);

  document.getElementById('h-time').textContent = sample.t.toFixed(3) + 's';
  document.getElementById('h-phase').textContent = phaseName(sample.phase);

  setDualValue('h-alt-actual', 'h-alt-measured', measured.altitudeActual, measured.altitudeMeasured, 'm', 2);
  setDualValue('h-vvel-actual', 'h-vvel-measured', measured.verticalActual, measured.verticalMeasured, 'm/s', 2);
  setDualValue('h-hvel-actual', 'h-hvel-measured', measured.horizontalActual, measured.horizontalMeasured, 'm/s', 2);
  setDualValue('h-speed-actual', 'h-speed-measured', measured.speedActual, measured.speedMeasured, 'm/s', 2);
  setDualValue('h-mass-actual', 'h-mass-measured', measured.massActual, measured.massMeasured, 'kg', 3);
  setDualValue('h-wind-actual', 'h-wind-measured', measured.windActual, measured.windMeasured, 'm/s', 2);
  document.getElementById('h-tti-actual').textContent = isFinite(measured.ttiActual) ? `${measured.ttiActual.toFixed(1)} s` : '—';
  document.getElementById('h-tti-measured').textContent = isFinite(measured.ttiMeasured) ? `${measured.ttiMeasured.toFixed(1)} s` : '—';
  document.getElementById('h-ekf-mass').textContent = `${measured.ekfMass.toFixed(3)} kg`;
  document.getElementById('h-ekf-cd').textContent = measured.ekfCd.toFixed(3);

  const altActualEl = document.getElementById('h-alt-actual');
  altActualEl.style.color = alt > 50 ? 'var(--success)' : alt > 10 ? 'var(--warning)' : 'var(--danger)';
  const vvelActualEl = document.getElementById('h-vvel-actual');
  vvelActualEl.style.color = vVel > 0 ? 'var(--success)' : vVel > -5 ? 'var(--warning)' : 'var(--danger)';

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
  return run && typeof run.nominal_ignition_altitude === 'number' ? run.nominal_ignition_altitude : 30;
}

function updateMLOverlay(sample) {
  const overlay = document.getElementById('ml-overlay');
  if (!state.currentRun || state.currentRun.mode !== 'ML') {
    overlay.style.display = 'none';
    updateCorrectionBar(null);
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
  updateCorrectionBar(sample);
}

// ═══════════════════════════════════════════════════════════════
// MAIN ANIMATION LOOP
// ═══════════════════════════════════════════════════════════════
const clock = new THREE.Clock();
let frameCount = 0, fpsTime = 0;
let lastTrailTime = -1;
let lastSecTrailTime = -1;
let cameraShake = new THREE.Vector3();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const LOOP_RESTART_DELAY = 2.5;

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
      if (state.loopPlayback) {
        state.currentTime = tMax;
        if (state.loopRestartRemaining <= 0) {
          state.loopRestartRemaining = LOOP_RESTART_DELAY;
        } else {
          state.loopRestartRemaining = Math.max(0, state.loopRestartRemaining - dt);
          if (state.loopRestartRemaining <= 0) {
            state.currentTime = 0;
            resetTrail();
            resetSecTrail();
            secondaryGroup.visible = false;
            crashActive = false;
            crashFlash.material.opacity = 0;
            predRing.visible = false;
            lastTrailTime = -1;
            lastSecTrailTime = -1;
            lastFaultType = 0;
          }
        }
      } else {
        state.currentTime = tMax;
        state.loopRestartRemaining = 0;
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

    // ── Secondary body (payload satellite) ──
    if (typeof SECONDARY_BODY !== 'undefined' && SECONDARY_BODY.length > 0) {
      const secSepTime = SECONDARY_BODY[0][0];
      if (state.currentTime >= secSepTime) {
        // Interpolate secondary body data
        let sLo = 0, sHi = SECONDARY_BODY.length - 1;
        const sTime = Math.min(state.currentTime, SECONDARY_BODY[sHi][0]);
        while (sHi - sLo > 1) {
          const sMid = (sLo + sHi) >> 1;
          if (SECONDARY_BODY[sMid][0] <= sTime) sLo = sMid; else sHi = sMid;
        }
        const st0 = SECONDARY_BODY[sLo][0], st1 = SECONDARY_BODY[sHi][0];
        const sFrac = st1 > st0 ? (sTime - st0) / (st1 - st0) : 0;
        const sDx = SECONDARY_BODY[sLo][1] + (SECONDARY_BODY[sHi][1] - SECONDARY_BODY[sLo][1]) * sFrac;
        const sDy = SECONDARY_BODY[sLo][2] + (SECONDARY_BODY[sHi][2] - SECONDARY_BODY[sLo][2]) * sFrac;
        const sAlt = SECONDARY_BODY[sLo][3] + (SECONDARY_BODY[sHi][3] - SECONDARY_BODY[sLo][3]) * sFrac;

        // Get primary body position at separation to use as base offset
        const sepSample = sampleTrajectory(state.currentTrajectory, secSepTime);
        if (sepSample) {
          const secPx = (sepSample.x + sDx) * state.lateralScale;
          const secPy = sAlt * state.verticalScale;
          const secPz = (sepSample.y + sDy) * state.lateralScale;

          secondaryGroup.position.set(secPx, secPy, secPz);
          secondaryGroup.visible = true;

          // Payload has TVC — maintain mostly upright orientation
          // Compute velocity direction for slight tilt
          const sValt = (SECONDARY_BODY[sHi][3] - SECONDARY_BODY[sLo][3]) / Math.max(st1 - st0, 0.001);
          const sVx = (SECONDARY_BODY[sHi][1] - SECONDARY_BODY[sLo][1]) / Math.max(st1 - st0, 0.001);
          // Slight pitch toward velocity (max ~15 degrees)
          const pitchAngle = Math.atan2(sVx, Math.abs(sValt) + 0.1) * 0.5;
          secondaryGroup.rotation.set(0, 0, -pitchAngle);

          // Secondary trail
          if (state.showTrail) {
            const secTrailInterval = 0.1;
            if (sTime - lastSecTrailTime > secTrailInterval) {
              addSecTrailPoint(secPx, secPy, secPz);
              lastSecTrailTime = sTime;
            }
          }
          secTrailLine.visible = state.showTrail;
        }
      } else {
        secondaryGroup.visible = false;
        secTrailLine.visible = false;
      }
    }

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

    // Fault effects (particles, geometry)
    updateFaultEffects(sample, dt);
  }

  // Update particles
  updateParticles(dt);
  updateFaultParticles(dt);
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
applyChromeState();
syncLoopButton();
animate();

console.log('Flight data viewer loaded');
</script>
</body>
</html>
'''

# Write the HTML file
with open('hexavisual_pro_demo.html', 'w', encoding='utf-8') as f:
    f.write(html)

size = os.path.getsize('hexavisual_pro_demo.html')
print(f'Created hexavisual_pro_demo.html: {size} bytes ({size/1024:.1f} KB)')
