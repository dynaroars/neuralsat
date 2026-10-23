'''Colours for the light and the dark version of every figure.

The site shows the version that matches the reader's theme, so each figure
is drawn twice with the same code under a different ``rc`` dictionary.
'''

THEMES = {
    'light': {'bg': '#ffffff', 'fg': '#1b1b1b', 'line': '#c9c9c9',
              'muted': '#6a6a6a', 'blue': '#1f5fbf', 'red': '#c0392b',
              'green': '#2e7d32', 'orange': '#c46a00', 'purple': '#7b3fa0',
              'shade': '#e6edf7'},
    'dark': {'bg': '#1a1a19', 'fg': '#e8e6e3', 'line': '#45443f',
             'muted': '#b3b1ac', 'blue': '#7fb0f5', 'red': '#f08a7e',
             'green': '#7fcf86', 'orange': '#f0a050', 'purple': '#c39be0',
             'shade': '#26323f'},
}


def rc(mode):
    # matplotlib rc parameters of the theme ``mode`` ('light' or 'dark').
    c = THEMES[mode]
    return {'figure.facecolor': c['bg'], 'axes.facecolor': c['bg'],
            'savefig.facecolor': c['bg'], 'axes.edgecolor': c['line'],
            'axes.labelcolor': c['fg'], 'text.color': c['fg'],
            'xtick.color': c['fg'], 'ytick.color': c['fg'],
            'grid.color': c['line'], 'legend.facecolor': c['bg'],
            'legend.edgecolor': c['line'], 'axes.titlecolor': c['fg'],
            'font.size': 10}
