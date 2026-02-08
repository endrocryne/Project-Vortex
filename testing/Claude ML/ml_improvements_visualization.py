"""
ML Training Pipeline Improvements - Quick Visual Summary

This script generates a comparison chart showing before/after improvements.
Run with: python ml_improvements_visualization.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('ML Training Pipeline: Before vs After Improvements', 
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# Plot 1: Feature Count Comparison
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
categories = ['Original', 'Improved']
feature_counts = [8, 26]
colors = ['#ff6b6b', '#51cf66']

bars = ax1.bar(categories, feature_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
ax1.set_title('Input Features', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 30)
ax1.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, feature_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# ============================================================================
# Plot 2: Rocket Mass Range
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
x_pos = [0, 1]
original_range = [0.5, 18.5]
improved_range = [0.2, 70]

ax2.barh([0], [original_range[1] - original_range[0]], left=original_range[0], 
         height=0.4, color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=2, label='Original')
ax2.barh([1], [improved_range[1] - improved_range[0]], left=improved_range[0], 
         height=0.4, color='#51cf66', alpha=0.7, edgecolor='black', linewidth=2, label='Improved')

ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Original', 'Improved'])
ax2.set_xlabel('Mass (kg)', fontsize=11, fontweight='bold')
ax2.set_title('Rocket Mass Coverage', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 75)
ax2.grid(axis='x', alpha=0.3)
ax2.text(9.5, 0, '0.5-18.5 kg', ha='center', va='center', fontsize=9, fontweight='bold')
ax2.text(35, 1, '0.2-70 kg', ha='center', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# Plot 3: Wind Speed Range
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
original_wind = [0, 12]
improved_wind = [0, 25]

ax3.barh([0], [original_wind[1]], height=0.4, color='#ff6b6b', alpha=0.7, 
         edgecolor='black', linewidth=2, label='Original')
ax3.barh([1], [improved_wind[1]], height=0.4, color='#51cf66', alpha=0.7, 
         edgecolor='black', linewidth=2, label='Improved')

ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Original', 'Improved'])
ax3.set_xlabel('Wind Speed (m/s)', fontsize=11, fontweight='bold')
ax3.set_title('Wind Coverage', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 28)
ax3.grid(axis='x', alpha=0.3)
ax3.text(6, 0, '0-12 m/s', ha='center', va='center', fontsize=9, fontweight='bold')
ax3.text(12.5, 1, '0-25 m/s', ha='center', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# Plot 4: Model Architecture
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
layers_orig = ['Input\n8', 'Dense\n256', 'Dense\n128', 'Dense\n64', 'Output\n1']
layers_new = ['Input\n26', 'Dense+BN+Drop\n512', 'Dense+BN+Drop\n256', 
              'Dense+BN+Drop\n128', 'Dense+BN\n64', 'Output\n1']

y_orig = np.arange(len(layers_orig))
y_new = np.arange(len(layers_new))

ax4.barh(y_orig, [1]*len(layers_orig), height=0.35, left=-0.5, 
         color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=2)
ax4.barh(y_new, [1]*len(layers_new), height=0.35, left=0.5, 
         color='#51cf66', alpha=0.7, edgecolor='black', linewidth=2)

for i, label in enumerate(layers_orig):
    ax4.text(0, i, label, ha='center', va='center', fontsize=8, fontweight='bold')
for i, label in enumerate(layers_new):
    ax4.text(1, i, label, ha='center', va='center', fontsize=8, fontweight='bold')

ax4.set_xlim(-0.75, 1.75)
ax4.set_ylim(-0.5, max(len(layers_orig), len(layers_new)) - 0.5)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Original', 'Improved'])
ax4.set_yticks([])
ax4.set_title('Model Architecture', fontsize=12, fontweight='bold')
ax4.invert_yaxis()

# ============================================================================
# Plot 5: Issue Severity Breakdown
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
issues = ['Critical', 'Major', 'Moderate']
counts = [3, 8, 1]
colors_issues = ['#ff6b6b', '#ffa94d', '#ffd43b']

wedges, texts, autotexts = ax5.pie(counts, labels=issues, colors=colors_issues,
                                    autopct='%d', startangle=90, 
                                    wedgeprops={'edgecolor': 'black', 'linewidth': 2})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax5.set_title('Issues Identified\n(12 total)', fontsize=12, fontweight='bold')

# ============================================================================
# Plot 6: Physics Fidelity
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
physics_aspects = ['Forces', 'Torques', 'Angular\nMomentum', 'TVC\nDynamics', 'CG Shift']
original_scores = [1.0, 0.0, 0.0, 0.0, 0.0]  # Only forces correct
improved_scores = [1.0, 1.0, 1.0, 1.0, 1.0]   # All correct

x = np.arange(len(physics_aspects))
width = 0.35

bars1 = ax6.bar(x - width/2, original_scores, width, label='Original', 
                color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=2)
bars2 = ax6.bar(x + width/2, improved_scores, width, label='Improved', 
                color='#51cf66', alpha=0.7, edgecolor='black', linewidth=2)

ax6.set_ylabel('Correct', fontsize=11, fontweight='bold')
ax6.set_title('Physics Simulation Fidelity', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(physics_aspects, fontsize=9)
ax6.set_ylim(0, 1.2)
ax6.set_yticks([0, 1])
ax6.set_yticklabels(['No', 'Yes'])
ax6.legend(fontsize=9)
ax6.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 7: Data Sampling Density
# ============================================================================
ax7 = plt.subplot(3, 3, 7)
categories_samp = ['Step Interval', 'Points per Flight']
original_samp = [12, 100/12]  # Every 12 steps, ~8 points per flight
improved_samp = [5, 100/5]     # Every 5 steps, ~20 points per flight

x = np.arange(len(categories_samp))
width = 0.35

ax7.bar(x - width/2, original_samp, width, label='Original', 
        color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=2)
ax7.bar(x + width/2, improved_samp, width, label='Improved', 
        color='#51cf66', alpha=0.7, edgecolor='black', linewidth=2)

ax7.set_ylabel('Value', fontsize=11, fontweight='bold')
ax7.set_title('Data Sampling', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(categories_samp, fontsize=10)
ax7.legend(fontsize=9)
ax7.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 8: Rocket Classes
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
classes_orig = ['Small', 'Medium', 'Large']
classes_new = ['Micro', 'Small', 'Medium', 'Large', 'Heavy']

ax8.barh([0], [len(classes_orig)], height=0.4, color='#ff6b6b', alpha=0.7, 
         edgecolor='black', linewidth=2)
ax8.barh([1], [len(classes_new)], height=0.4, color='#51cf66', alpha=0.7, 
         edgecolor='black', linewidth=2)

ax8.set_yticks([0, 1])
ax8.set_yticklabels(['Original', 'Improved'])
ax8.set_xlabel('Number of Classes', fontsize=11, fontweight='bold')
ax8.set_title('Rocket Diversity', fontsize=12, fontweight='bold')
ax8.set_xlim(0, 6)
ax8.grid(axis='x', alpha=0.3)
ax8.text(1.5, 0, '3 classes', ha='center', va='center', fontsize=9, fontweight='bold')
ax8.text(2.5, 1, '5 classes', ha='center', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# Plot 9: Summary Checklist
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

checklist_text = """
✅ FIXED - Critical Issues:
  • Full 6-DOF dynamics
  • TVC control integration
  • 26 features (was 8)
  
✅ IMPROVED - Major Issues:
  • 5 rocket classes (was 3)
  • Mass: 0.2-70kg (was 0.5-18.5kg)
  • Wind: 0-25 m/s (was 0-12 m/s)
  • Fault prob: 15% (was 40%)
  • Oracle: 2-stage (was 1-stage)
  
✅ ENHANCED - Architecture:
  • 5 layers (was 3)
  • Batch normalization
  • Dropout regularization
  • Stratified sampling
"""

ax9.text(0.05, 0.95, checklist_text, transform=ax9.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('ml_improvements_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved to: ml_improvements_comparison.png")
print("\nKey Improvements Summary:")
print("  🔴 3 Critical issues fixed (physics, TVC, features)")
print("  🟠 8 Major issues improved (ranges, conditions, architecture)")
print("  🟡 1 Moderate issue enhanced (oracle search)")
print("  ✅ Total: 12 issues addressed")
print("\n  Features: 8 → 26 (+225%)")
print("  Rocket range: 0.5-18.5kg → 0.2-70kg (+278%)")
print("  Model depth: 3 → 5 layers (+67%)")
print("  Sample density: Every 12 → Every 5 steps (+140%)")

plt.show()
