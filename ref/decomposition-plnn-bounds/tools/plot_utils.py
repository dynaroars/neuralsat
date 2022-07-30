import matplotlib.pyplot as plt
import matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def custom_plot(fignumber, x, y, STD, xlabel, ylabel, title, errorbars=False, labelname="", dotted="-", xlog=False,
                ylog=False, lw=1.15, color=None):
    # Utility function for better (clean, customisable) plotting.

    fontsize = 10
    matplotlib.rcParams.update({'font.size': fontsize})
    plt.figure(fignumber, figsize=(8, 5))
    additional_args = {}
    error_args = {}
    if dotted == "dashed":
        additional_args['dashes'] = (5, 5)
    if color:
        additional_args['color'] = color
        error_args['color'] = color
    if not errorbars:
        additional_args['marker'] = 'x'
    if dotted != "-":
        lw += 0.15
    plt.plot(x, y, linestyle=dotted, label=labelname, ms=4, linewidth=lw, **additional_args)
    if errorbars:
        plt.fill_between(x, y-STD[0], y+STD[1], alpha=0.12, **error_args)
    if xlog:
        plt.xscale('log', nonposx='clip')
    if ylog:
        plt.yscale('log', nonposy='clip')
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title)
    plt.legend(fontsize=fontsize)
