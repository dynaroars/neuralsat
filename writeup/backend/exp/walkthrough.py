'''Write ``content/walkthrough.md``: the CROWN bounds of the toy network.

Every number on the page comes from ``toy.trace``, which runs the project's
own layer functions. The page follows the explainer at
https://github.com/hocdot/abstract-domain-explainer: one step per layer, each
bound traced from the output back to the input box.

Usage (from the project root):
    python -m writeup.backend.exp.walkthrough
'''

from writeup.backend.exp import toy
from writeup.backend.exp.config import CONTENT_DIR

OUT = f'{CONTENT_DIR}/walkthrough.md'
# the variable set before layer j: x, z1, a1, z2, a2, y
NAMES = ('x', 'z^{(1)}', 'a^{(1)}', 'z^{(2)}', 'a^{(2)}', 'y')
KEY = {'lower': 'lo', 'upper': 'hi'}
REL = {'lower': r'\ge', 'upper': r'\le'}
EPS = 1e-9


def num(value):
    # A number with at most three decimals and no trailing zeros.
    text = f'{value:.3f}'.rstrip('0').rstrip('.')
    return '0' if text in ('', '-0') else text


def sym(variable, neuron):
    # LaTeX of neuron ``neuron`` (0-based) of a variable set.
    if variable == 5:
        return 'y'
    return f'{NAMES[variable]}_{{{neuron + 1}}}'


def lin(coefs, variable, const):
    # LaTeX of ``sum coef_k * symbol_k + const`` over a variable set.
    parts = []
    for k, coef in enumerate(coefs):
        if abs(coef) >= EPS:
            magnitude = abs(coef)
            body = sym(variable, k) if abs(magnitude - 1) < EPS else (
                f'{num(magnitude)}\\,{sym(variable, k)}')
            parts.append((coef < 0, body))
    if abs(const) >= EPS or not parts:
        parts.append((const < 0, num(abs(const))))
    text = ('-' if parts[0][0] else '') + parts[0][1]
    for negative, body in parts[1:]:
        text += f' {"-" if negative else "+"} {body}'
    return text


def signed(value):
    # A number, in parentheses when negative.
    return f'({num(value)})' if value < 0 else num(value)


def times(coef, text):
    # LaTeX of ``coef * (text)``.
    inner = f'\\left({text}\\right)' if ' ' in text else text
    return f'{num(coef)} \\cdot {inner}'


def total(parts, const):
    # LaTeX sum of the substituted parts and the constant.
    pieces = list(parts)
    if abs(const) >= EPS or not pieces:
        pieces.append(num(const))
    return ' + '.join(pieces).replace('+ -', '- ')


def vector(matrix, k):
    return [float(v) for v in matrix[k]]


def display(equation):
    return ['$$', equation, '$$', '']


def block(name, args, lines):
    # A ``{{% name args %}}`` shortcode around Markdown lines.
    return [f'{{{{% {name} {args} %}}}}', '', *lines,
            f'{{{{% /{name} %}}}}', '']


# -- one back-substitution walk ----------------------------------------------

def use_of(coef, side, relax):
    # The line a coefficient takes on a side: the lower line for a positive
    # coefficient on the lower bound, the opposite line otherwise.
    which = 'lower' if (coef >= 0) == (side == 'lower') else 'upper'
    slope, shift = relax['lo' if which == 'lower' else 'hi']
    return {**relax, 'line': which, 'slope': slope, 'shift': shift,
            'coef': coef}


def why_line(use):
    # Why this ReLU output takes this line, with the numbers.
    low, up = use['lower'], use['upper']
    if use['status'] == 'active':
        return (f'always active ($z \\ge 0$ on $[{num(low)}, {num(up)}]$): '
                'both lines are $z$.')
    if use['status'] == 'inactive':
        return (f'always inactive ($z \\le 0$ on $[{num(low)}, {num(up)}]$): '
                'both lines are $0$.')
    if use['line'] == 'upper':
        slope = up / (up - low)
        return (f'unstable, so the upper line is the chord through '
                f'$({num(low)}, 0)$ and $({num(up)}, {num(up)})$: slope '
                f'$\\tfrac{{u}}{{u-l}} = \\tfrac{{{num(up)}}}'
                f'{{{num(up)} - ({num(low)})}} = {num(slope)}$, intercept '
                f'$-l \\cdot {num(slope)} = {num(-low * slope)}$.')
    verdict = '>' if up > -low else r'\le'
    return (f'unstable, so the lower line is $a \\ge \\lambda z$ with '
            f'$\\lambda = {int(use["slope"])}$, because $u = {num(up)} '
            f'{verdict} -l = {num(-low)}$ (the smaller triangle).')


def expand_card(trace, state, side, k):
    # A linear layer: replace each neuron by its definition (exact).
    j, key = state['layer'], KEY[side]
    weight = trace['layers'][j]['weight']
    bias = trace['layers'][j]['bias']
    coefs, const = state['before'][key]
    body, parts = ['These come straight from the weights and biases, so '
                   'the step is exact.', ''], []
    for n, coef in enumerate(vector(coefs, k)):
        if abs(coef) >= EPS:
            definition = lin(vector(weight, n), j, float(bias[n]))
            body += display(f'{sym(j + 1, n)} = {definition}')
            parts.append(times(coef, definition))
    new_coefs, new_const = state[key]
    return body, parts, lin(vector(new_coefs, k), j, float(new_const[k]))


def relax_card(trace, state, side, k):
    # A ReLU layer: replace each output by one of its two lines.
    j, key = state['layer'], KEY[side]
    coefs, const = state['before'][key]
    same, flip = ('lower', 'upper') if side == 'lower' else (
        'upper', 'lower')
    body = [f'Each ReLU output takes one of its two lines. For the {side} '
            f'bound a positive coefficient takes the {same} line and a '
            f'negative one the {flip} line.', '']
    parts = []
    for n, coef in enumerate(vector(coefs, k)):
        if abs(coef) < EPS:
            continue
        use = use_of(coef, side, trace['relax'][j][n])
        line = lin([use['slope']], j, use['shift']).replace(
            sym(j, 0), sym(j, n))
        body += [f'- ${sym(j + 1, n)}$ has coefficient ${num(coef)}$ '
                 f'({"positive" if coef >= 0 else "negative"}), so it takes '
                 f'the {use["line"]} line '
                 f'${sym(j + 1, n)} {REL[use["line"]]} {line}$: '
                 f'{why_line(use)}']
        parts.append(times(coef, line))
    body.append('')
    new_coefs, new_const = state[key]
    return body, parts, lin(vector(new_coefs, k), j, float(new_const[k]))


def walk_cards(trace, step, k, side):
    # The numbered cards of one walk, then the finished bound over x.
    target = sym(step['layer'] + 1, k)
    lines, number, relaxed = [], 1, False
    lines += block('card', f'"{number}"', [
        f'**Start from the neuron itself.** ${target} = {target}$.', ''])
    for state in step['states'][1:]:
        number += 1
        relu = state['op'] == 'relu'
        card = relax_card if relu else expand_card
        body, parts, simplified = card(trace, state, side, k)
        relaxed = relaxed or relu
        key = KEY[side]
        const = float(state['before'][key][1][k])
        name = NAMES[state['layer'] + 1]
        title = (f'**Relax ${name}$: replace each ReLU output by a line.**'
                 if relu else f'**Expand ${name}$: replace each linear '
                 'neuron by its definition.**')
        relation = REL[side] if relaxed else '='
        body = [title, ''] + body + [
            'Substitute and simplify:', ''] + display(
            f'\\begin{{aligned}} {target} &{relation} '
            f'{total(parts, const)} \\\\ &= {simplified} \\end{{aligned}}')
        lines += block('card', f'"{number}"', body)
    return lines


def plug_line(row, side, target, exact=False):
    # The finished bound and the box plugged in, as one aligned equation.
    first = '=' if exact else REL[side]
    expr = lin([t['coef'] for t in row['terms']], 0, row['const'])
    picks = [f'{num(t["coef"])} \\cdot {signed(t["end"])}'
             for t in row['terms'] if abs(t['coef']) >= EPS]
    return (f'\\begin{{aligned}} {target} &{first} {expr} \\\\ '
            f'&{REL[side]} {total(picks, row["const"])} = '
            f'{num(row["value"])} \\end{{aligned}}')


def concretize_cards(step, k, number):
    # Both bounds over x and the plugged-in box.
    target = sym(step['layer'] + 1, k)
    body = ['Each input takes the end of its interval that makes its term '
            'as small as possible for the lower bound (as large as '
            'possible for the upper bound).', '']
    for side in ('lower', 'upper'):
        body += display(plug_line(step[side][k], side, target))
    return block('card', f'"{number}"', ['**Concretize: plug the box into '
                                         'both bounds.**', ''] + body)


def neuron_walk(trace, index, k):
    # The full walk of one neuron: lower, upper, then concretize.
    step = trace['steps'][index]
    lines = []
    for side in ('lower', 'upper'):
        lines += block('walk', f'"{side.capitalize()} bound"',
                       walk_cards(trace, step, k, side))
    return lines + concretize_cards(step, k, '✓')


# -- the page ----------------------------------------------------------------

def matrix(rows):
    inner = ' \\\\ '.join(' & '.join(num(float(v)) for v in r) for r in rows)
    return f'\\begin{{pmatrix}} {inner} \\end{{pmatrix}}'


def column(values):
    return matrix([[v] for v in values])


def network_section(trace):
    layers = trace['layers']
    weights = [(layers[j]['weight'], layers[j]['bias']) for j in (0, 2, 4)]
    lines = ['## The network', '',
             'Two inputs, two hidden layers of two ReLU neurons and one '
             'output. $z$ is a layer before its ReLU, $a$ after it.', '']
    names = ('z^{(1)} = W_1 x + b_1', 'z^{(2)} = W_2 a^{(1)} + b_2',
             'y = W_3 a^{(2)} + b_3')
    for (weight, bias), name in zip(weights, names):
        lines += display(f'{name},\\quad W = {matrix(weight.tolist())},'
                         f'\\quad b = {column(bias.tolist())}')
    sample = trace['sampled'][4]
    low, high = toy.X0[0] - toy.EPS, toy.X0[0] + toy.EPS
    lines += [f'The input box is $x_1 \\in [{num(low)}, {num(high)}]$, '
              f'$x_2 \\in [{num(toy.X0[1] - toy.EPS)}, '
              f'{num(toy.X0[1] + toy.EPS)}]$. **Question: is $y > 0$ on the '
              'whole box?** Sampling '
              f'{toy.SAMPLES:,} inputs gives $y \\in [{num(sample[0][0])}, '
              f'{num(sample[1][0])}]$, so the answer looks like yes, but '
              'sampling proves nothing. The bounds below do.', '']
    return lines


def first_section(trace):
    step = trace['steps'][0]
    lines = ['## Step 1: the first layer is exact', '',
             'The first layer reads the inputs directly, so there is nothing '
             'to relax: plug the box into the neuron. Each input takes the '
             'end of its interval that makes its term smallest for the lower '
             'bound and largest for the upper bound.', '']
    for k in range(2):
        target = sym(1, k)
        for side in ('lower', 'upper'):
            lines += display(plug_line(step[side][k], side, target, True))
    return lines


def relu_section(trace, j, title, figure, area_figure=False):
    layer_name = NAMES[j].replace('z', 'a')
    lines = [f'## {title}', '',
             f'Each ReLU output ${layer_name}$ gets two lines from the '
             'bounds of its input $z \\in [l, u]$: an upper line and a '
             'lower line.', '']
    for k, info in enumerate(trace['relax'][j]):
        low, up = info['lower'], info['upper']
        lines += [f'- **${sym(j, k)} \\in [{num(low)}, {num(up)}]$**, '
                  f'{info["status"]}.']
        if info['status'] != 'unstable':
            lines += ['']
            continue
        slope, shift = info['hi']
        area0 = 0.5 * (up - low) * up
        area1 = 0.5 * (up - low) * -low
        lam = int(info['lo'][0])
        lines += [
            f'  Upper line (the chord): $a \\le {num(slope)}\\,z + '
            f'{num(shift)}$. Lower line $a \\ge \\lambda z$: '
            f'$\\lambda = 0$ leaves a triangle of area '
            f'$\\tfrac12 (u-l)\\,u = {num(area0)}$, '
            f'$\\lambda = 1$ one of area $\\tfrac12 (u-l)(-l) = '
            f'{num(area1)}$. The smaller one wins: $\\lambda = {lam}$.', '']
    if area_figure:
        lines += ['{{< fig "relu-area" "The two lower lines of the first '
                  'neuron of layer 1 and the area each leaves between the '
                  'lines. CROWN keeps the smaller one (here lambda = 1). '
                  'The code compares u and -l, which is the same test." >}}',
                  '']
    lines += [f'{{{{< fig "{figure}" "The lines of the two neurons. The '
              'shaded band is the interval of the input." >}}', '']
    return lines


def layer2_section(trace):
    lines = ['## Step 3: the second layer by back-substitution', '',
             'The second layer reads the ReLU outputs, which are not the '
             'inputs. Intervals for them would ignore that both come from '
             'the same $x$. CROWN instead rewrites the neuron backward, one '
             'layer at a time, until only $x$ remains. Walk the first '
             'neuron; the walks are folded away, open them to read.', '',
             '### First neuron of layer 2', '',
             'The neuron is $z^{(2)}_1$.', '']
    lines += neuron_walk(trace, 1, 0)
    lines += ['### Second neuron of layer 2', '',
              'The neuron is $z^{(2)}_2$, walked the same way and shown '
              'only by its result:', '']
    lines += concretize_cards(trace['steps'][1], 1, '✓')
    return lines


def output_section(trace):
    return (['## Step 5: the output', '',
             'The output is walked the same way, through three linear '
             'layers and two ReLU layers.', '']
            + neuron_walk(trace, 2, 0))


def compare_section(trace):
    names = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    lines = ['## Step 6: compare with interval propagation', '',
             'Interval propagation replaces each neuron by its interval and '
             'forgets how the neurons depend on each other. CROWN keeps the '
             'dependence in the linear bounds.', '',
             '| neuron | CROWN | interval propagation | sampled |',
             '|---|---|---|---|']
    for step_index, k in names:
        step = trace['steps'][step_index]
        layer = step['layer']
        low, up = step['lower'][k]['value'], step['upper'][k]['value']
        ibp = trace['ibp'][layer]
        sample = trace['sampled'][layer]
        lines.append(
            f'| ${sym(layer + 1, k)}$ | $[{num(low)}, {num(up)}]$ '
            f'| $[{num(ibp[0][k])}, {num(ibp[1][k])}]$ '
            f'| $[{num(sample[0][k])}, {num(sample[1][k])}]$ |')
    out = trace['steps'][2]
    lines += ['', '{{< fig "toy-bounds" "Interval propagation (grey), CROWN '
              '(blue) and the range reached on sampled inputs (green) for '
              'every bounded neuron." >}}', '',
              '**Verdict.** CROWN proves '
              f'$y \\ge {num(out["lower"][0]["value"])} > 0$ on the whole '
              'box; interval propagation only gets '
              f'$y \\ge {num(trace["ibp"][4][0][0])}$ and cannot. The '
              'lower bound equals the smallest sampled value, so it is '
              'tight here. Both bounds are linear in $x$ at every step, so '
              'they hold for every input in the box (not only for the '
              'sampled ones).', '']
    return lines


def page(trace):
    head = ['+++', "title = 'Walkthrough'", 'toc = true',
            "description = 'CROWN on a toy network, every bound traced "
            "from the output back to the input box.'", '+++', '',
            '<!-- Generated by `python -m '
            'writeup.backend.exp.walkthrough`; do not edit. -->', '',
            'This page bounds a tiny network by hand, with the project\'s own '
            'code doing the arithmetic. It follows the style of the '
            '[abstract domain explainer]'
            '(https://github.com/hocdot/abstract-domain-explainer): every '
            'bound is traced from the output back to the input box, with '
            'the numbers plugged in. Numbers are rounded to three '
            'decimals.', '']
    return '\n'.join(
        head + network_section(trace) + first_section(trace)
        + relu_section(trace, 1, 'Step 2: the ReLU lines of layer 1',
                       'toy-relu1', area_figure=True)
        + layer2_section(trace)
        + relu_section(trace, 3, 'Step 4: the ReLU lines of layer 2',
                       'toy-relu2')
        + output_section(trace) + compare_section(trace)) + '\n'


def main():
    with open(OUT, 'w') as handle:
        handle.write(page(toy.trace()))
    print(f'[walkthrough] wrote {OUT}')


if __name__ == '__main__':
    main()
