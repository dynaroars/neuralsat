'''Figures of how each non-affine layer is bounded by two lines.

The lines are the ones the code computes (``relax`` of each layer module),
drawn over the interval ``[l, u]`` of the layer input. Writes
``frontend/static/img/{name}-light.png`` and ``-dark.png`` for the site.

Usage (from the project root):
    python -m writeup.backend.plot.relaxation
'''

import os

import matplotlib.pyplot as plt
import torch

from abstraction.lirpa.layer import leakyrelu, maxpool, relu, sigmoid, tanh
from writeup.backend.exp import toy
from writeup.backend.exp.config import IMG_DIR
from writeup.backend.plot.style import THEMES, rc

DTYPE = torch.float64
SLOPE = 0.2  # LeakyReLU negative slope used in the figure


def lines(relax, lower, upper):
    # Lower and upper line of ``relax`` on one interval: ((k, b), (k, b)).
    lo, hi = relax(torch.tensor([lower], dtype=DTYPE),
                   torch.tensor([upper], dtype=DTYPE))
    return tuple((float(k), float(b)) for k, b in (lo, hi))


def panel(ax, colors, func, relax, lower, upper, title):
    # The function, the interval and the two lines on one axis.
    pad = 0.6 * (upper - lower) + 0.3
    grid = torch.linspace(lower - pad, upper + pad, 400, dtype=DTYPE)
    ax.plot(grid, func(grid), color=colors['fg'], linewidth=1.6,
            label='function')
    ax.axvspan(lower, upper, color=colors['shade'], zorder=0)
    (k_lo, b_lo), (k_hi, b_hi) = lines(relax, lower, upper)
    span = torch.linspace(lower - 0.25 * pad, upper + 0.25 * pad, 2,
                          dtype=DTYPE)
    ax.plot(span, k_lo * span + b_lo, color=colors['blue'], linewidth=1.8,
            label='lower line')
    ax.plot(span, k_hi * span + b_hi, color=colors['red'], linewidth=1.8,
            label='upper line')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('z')
    ax.grid(True, alpha=0.4)
    ax.set_ylim(float(func(grid).min()) - 0.35 * pad,
                float(func(grid).max()) + 0.35 * pad)


def activation_figure(name, func, relax, cases, mode):
    # One row of panels, one per ``(lower, upper, title)`` case.
    colors = THEMES[mode]
    with plt.rc_context(rc(mode)):
        fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases),
                                                         3.0))
        for ax, (lower, upper, title) in zip(axes, cases):
            panel(ax, colors, func, relax, lower, upper, title)
        axes[0].set_ylabel('a')
        axes[0].legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f'{name}-{mode}.png'), dpi=160)
        plt.close(fig)


def window(ax, colors, lower, upper, title):
    # Max pooling window: the input intervals and what the lines read.
    layer = {'op': 'maxpool', 'kernel': (2, 2), 'in_shape': (1, 2, 2)}
    index, _, (slope, shift) = maxpool.relaxation(
        layer, torch.tensor(lower, dtype=DTYPE),
        torch.tensor(upper, dtype=DTYPE))
    index, linear = int(index[0]), bool(slope[0] == 1)
    for i, (low, up) in enumerate(zip(lower, upper)):
        ax.plot([i, i], [low, up], linewidth=7, solid_capstyle='butt',
                color=colors['blue'] if i == index else colors['muted'])
    ax.axhline(float(shift[0]) if not linear else max(upper),
               color=colors['red'], linestyle='--', linewidth=1.5)
    ax.annotate('lower line reads the largest l',
                (index, lower[index]), (0.2, min(lower) - 0.9), fontsize=8,
                color=colors['blue'], arrowprops={'arrowstyle': '->',
                                                  'color': colors['blue']})
    label = ('upper line = this input (linear window)' if linear
             else 'upper line = constant max u')
    ax.text(len(lower) - 0.6, max(upper) + 0.12, label, ha='right',
            fontsize=8, color=colors['red'])
    ax.set_xticks(range(len(lower)), [f'x{i + 1}' for i in range(len(lower))])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('value')
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_xlim(-0.6, len(lower) - 0.4)
    ax.set_ylim(min(lower) - 1.3, max(upper) + 0.7)


def maxpool_figure(mode):
    # Two windows of four inputs: a linear one and a non-linear one.
    colors = THEMES[mode]
    cases = [([2.0, -1.0, 0.0, -0.5], [3.0, 0.5, 1.0, 1.2],
              'linear window: l1 >= u2, u3, u4'),
             ([0.0, -0.5, -1.0, 0.2], [2.0, 1.5, 1.0, 1.8],
              'non-linear window: the bounds overlap')]
    with plt.rc_context(rc(mode)):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
        for ax, (lower, upper, title) in zip(axes, cases):
            window(ax, colors, lower, upper, title)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f'maxpool-{mode}.png'), dpi=160)
        plt.close(fig)


def line_values(line, span):
    slope, shift = line
    return slope * span + shift


def relu_panel(ax, colors, info, title):
    # One unstable ReLU of the toy network with its two lines.
    low, up = info['lower'], info['upper']
    pad = 0.35 * (up - low)
    grid = torch.linspace(low - pad, up + pad, 200, dtype=DTYPE)
    ax.plot(grid, torch.relu(grid), color=colors['fg'], linewidth=1.6)
    ax.axvspan(low, up, color=colors['shade'], zorder=0)
    span = torch.linspace(low, up, 2, dtype=DTYPE)
    ax.plot(span, line_values(info['lo'], span), color=colors['blue'],
            linewidth=1.8, label='lower line')
    ax.plot(span, line_values(info['hi'], span), color=colors['red'],
            linewidth=1.8, label='upper line')
    ax.set_title(f'{title}, z in [{low:.2f}, {up:.2f}]', fontsize=10)
    ax.set_xlabel('z')
    ax.grid(True, alpha=0.4)


def toy_relu_figure(trace, j, name, mode):
    # The lines of both neurons of ReLU layer ``j`` of the toy network.
    colors = THEMES[mode]
    infos = trace['relax'][j]
    with plt.rc_context(rc(mode)):
        fig, axes = plt.subplots(1, len(infos), figsize=(7.4, 3.0))
        for k, (ax, info) in enumerate(zip(axes, infos)):
            relu_panel(ax, colors, info, f'neuron {k + 1}')
        axes[0].set_ylabel('a')
        axes[0].legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f'{name}-{mode}.png'), dpi=160)
        plt.close(fig)


def area_figure(trace, mode):
    # The two candidate lower lines of the first unstable neuron.
    colors = THEMES[mode]
    info = trace['relax'][1][0]
    low, up = info['lower'], info['upper']
    chord = info['hi']
    span = torch.linspace(low, up, 200, dtype=DTYPE)
    upper_line = line_values(chord, span)
    with plt.rc_context(rc(mode)):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        for lam, ax in zip((0, 1), axes):
            lower_line = lam * span
            area = 0.5 * (up - low) * (up if lam == 0 else -low)
            ax.plot(span, torch.relu(span), color=colors['fg'],
                    linewidth=1.6)
            ax.fill_between(span, lower_line, upper_line,
                            color=colors['shade'])
            ax.plot(span, upper_line, color=colors['red'], linewidth=1.8)
            ax.plot(span, lower_line, color=colors['blue'], linewidth=1.8)
            chosen = lam == int(info['lo'][0])
            ax.set_title(f'lambda = {lam}: area {area:.2f}'
                         + (' (smaller, kept)' if chosen else ''),
                         fontsize=10)
            ax.set_xlabel('z')
            ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f'relu-area-{mode}.png'), dpi=160)
        plt.close(fig)


def bounds_figure(trace, mode):
    # Interval propagation, CROWN and sampling for every bounded neuron.
    colors = THEMES[mode]
    rows = []
    for step in trace['steps']:
        layer = step['layer']
        for k in range(len(step['lower'])):
            name = f'z{layer // 2 + 1}_{k + 1}' if layer < 4 else 'y'
            rows.append((name, trace['ibp'][layer][0][k],
                         trace['ibp'][layer][1][k], step['lower'][k]['value'],
                         step['upper'][k]['value'],
                         trace['sampled'][layer][0][k],
                         trace['sampled'][layer][1][k]))
    with plt.rc_context(rc(mode)):
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        for position, (name, il, iu, cl, cu, sl, su) in enumerate(rows):
            for offset, (low, up, key) in zip(
                    (0.25, 0.0, -0.25),
                    ((il, iu, 'muted'), (cl, cu, 'blue'),
                     (sl, su, 'green'))):
                ax.plot([low, up], [position + offset] * 2, linewidth=6,
                        solid_capstyle='butt', color=colors[key])
        ax.axvline(0, color=colors['fg'], linewidth=0.8)
        ax.set_yticks(range(len(rows)), [row[0] for row in rows])
        ax.invert_yaxis()
        ax.set_xlabel('value')
        ax.grid(True, axis='x', alpha=0.4)
        handles = [plt.Line2D([0], [0], color=colors[key], linewidth=6)
                   for key in ('muted', 'blue', 'green')]
        ax.legend(handles, ['interval propagation', 'CROWN', 'sampled'],
                  fontsize=8, loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f'toy-bounds-{mode}.png'),
                    dpi=160)
        plt.close(fig)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    leaky = torch.nn.functional.leaky_relu
    trace = toy.trace()
    for mode in THEMES:
        area_figure(trace, mode)
        toy_relu_figure(trace, 1, 'toy-relu1', mode)
        toy_relu_figure(trace, 3, 'toy-relu2', mode)
        bounds_figure(trace, mode)
        activation_figure(
            'relu', torch.relu, relu.relax,
            [(-1.5, 1.0, 'unstable, u < -l: lower slope 0'),
             (-1.0, 1.5, 'unstable, u > -l: lower slope 1'),
             (0.5, 2.0, 'stable active: exact')], mode)
        activation_figure(
            'leakyrelu', lambda z: leaky(z, SLOPE),
            lambda lo, up: leakyrelu.relax(SLOPE, lo, up),
            [(-1.5, 1.0, 'unstable, chord > 0.5: lower slope 1'),
             (-1.0, 0.4, 'unstable, chord < 0.5: lower slope 0.2'),
             (-2.0, -0.5, 'stable inactive: exact')], mode)
        for name, func, module in (('sigmoid', torch.sigmoid, sigmoid),
                                   ('tanh', torch.tanh, tanh)):
            activation_figure(
                name, func, module.relax,
                [(-3.0, -0.5, 'u <= 0: chord above, tangent below'),
                 (0.5, 3.0, 'l >= 0: tangent above, chord below'),
                 (-2.0, 3.0, 'l < 0 < u: tangent from an end')], mode)
        maxpool_figure(mode)
        print(f'[relaxation] {mode} figures written to {IMG_DIR}')


if __name__ == '__main__':
    main()
