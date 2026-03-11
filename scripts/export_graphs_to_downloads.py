"""
Export PlotVisual sample graphs to the Downloads folder.
Generates:
  - Sim vs Flight Validation  (4-panel)
  - Success Rate Cliff Plot
"""

import os
import sys

# Make sure the repo root is on the path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plugins.data_store import VortexDataStore
from plugins.claude_graphs.plugin import ClaudeGraphsPlugin

DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
DPI = 200

def main():
    plugin = ClaudeGraphsPlugin()
    ds = VortexDataStore()

    print("Generating sample data …")
    plugin.generate_sample_data(ds)

    # ── 1. Sim vs Flight ────────────────────────────────────────────────────
    fig1 = plt.figure(figsize=(14, 9))
    plugin.render_graph('sim_vs_flight', ds, fig1)
    out1 = os.path.join(DOWNLOADS, 'vortex_sim_vs_flight.png')
    fig1.savefig(out1, dpi=DPI, bbox_inches='tight', facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f"Saved → {out1}")

    # ── 2. Cliff Plot ────────────────────────────────────────────────────────
    fig2 = plt.figure(figsize=(10, 6))
    plugin.render_graph('cliff_plot', ds, fig2)
    out2 = os.path.join(DOWNLOADS, 'vortex_success_rate_cliff.png')
    fig2.savefig(out2, dpi=DPI, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Saved → {out2}")

    print("Done.")

if __name__ == '__main__':
    main()
