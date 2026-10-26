import matplotlib.pyplot as plt

# hafta 8 figürleriyle aynı palet
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2,
    'xtick.color': MUTED,
    'ytick.color': INK2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 160,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.25,
})
