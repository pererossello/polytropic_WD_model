
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def initialize_figure(
    fig_size=20, ratio=1,
    fig_w=512, fig_h=512,
    subplots=(1, 1), grid=True, 
    lw=0.015, ts=1.5, theme="dark",
    pad=0.5,
    color='#222222',
    dpi=300,
    wr=None, hr=None, hmerge=None, wmerge=None,
    ylabel='bottom',
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='out',
    minor=False,
    top_bool=False
):
    """
    Initialize a Matplotlib figure with a specified size, aspect ratio, text size, and theme.

    Parameters:
    fig_size (float): The size of the figure.
    ratio (float): The aspect ratio of the figure.
    text_size (float): The base text size for the figure.
    subplots (tuple): The number of subplots, specified as a tuple (rows, cols).
    grid (bool): Whether to display a grid on the figure.
    theme (str): The theme for the figure ("dark" or any other string for a light theme).

    Returns:
    fig (matplotlib.figure.Figure): The initialized Matplotlib figure.
    ax (list): A 2D list of axes for the subplots.
    fs (float): The scaling factor for the figure size.
    """
    if ratio is not None:
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
            dpi=dpi,
            layout=layout,
        )
    else:
        dpi = dpi
        ratio = fig_w / fig_h
        fig_width = fig_w / dpi
        fig_height = fig_h / dpi
        fig_size = fig_width * fig_height
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=dpi,  # Default dpi, will adjust later for saving
            layout=layout,
        )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr
    

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)


    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    if theme == "dark":
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            
            if hmerge is not None:
                if i in hmerge:
                    ax[i][j] = fig.add_subplot(gs[i, :])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            elif wmerge is not None:
                if j in wmerge:
                    ax[i][j] = fig.add_subplot(gs[:, j])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            else:
                ax[i][j] = fig.add_subplot(gs[i, j])

            if theme == "dark":
                ax[i][j].set_facecolor(color)
                ax[i][j].tick_params(colors="white")
                ax[i][j].spines["bottom"].set_color("white")
                ax[i][j].spines["top"].set_color("white")
                ax[i][j].spines["left"].set_color("white")
                ax[i][j].spines["right"].set_color("white")
                ax[i][j].xaxis.label.set_color("white")
                ax[i][j].yaxis.label.set_color("white")

            #ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)

            if grid:
                ax[i][j].grid(
                    which="major",
                    linewidth=fs * lw,
                    color="white" if theme == "dark" else "black",
                )
            for spine in ax[i][j].spines.values():
                spine.set_linewidth(fs * 0.15)

            if ylabel == 'bottom':
                labeltop_bool = False
                labelbottom_bool = True
            elif ylabel == 'top':
                labeltop_bool = True
                labelbottom_bool = False
                ax[i][j].xaxis.set_label_position('top')

            else:
                labeltop_bool = True
                labelbottom_bool = True
                ax[i][j].xaxis.set_label_position('both')

            
            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=ts * fs,
                size=fs * 0.5,
                width=fs * 0.15,
                pad= pad * fs,
                top=top_bool,
                labelbottom=labelbottom_bool,
                labeltop=labeltop_bool,
                right=top_bool,
                direction=tick_direction
            )

            if minor:
                ax[i][j].minorticks_on()
                ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=ts * fs,
                size=fs * 0.5,
                width=fs * 0.15,
                pad= pad * fs,
                top=top_bool,
                labelbottom=labelbottom_bool,
                labeltop=labeltop_bool,
                right=top_bool,
                direction=tick_direction
                )
                ax[i][j].tick_params(axis='both', which="minor", 
                direction=tick_direction,
                top=top_bool,
                right=top_bool,
                size=fs * 0.25, width=fs * 0.15,)

    if hmerge is not None:
        for k in hmerge:
            for l in range(1, subplots[1]):
                fig.delaxes(ax[k][l])

    if wmerge is not None:
        for k in wmerge:
            for l in range(1, subplots[0]):
                fig.delaxes(ax[l][k])
            
    
    return fig, ax, fs, gs

def plot_2_order_ODE_canonical(
    t,
    xs,
    vs,
    quiver_step=50,
    color="gist_ncar",
    orth_color="w",
    ortho=None,
    ax_equal=False,
    fig_size=20,
    ratio=2,
    labels=["$t$", "$x$", "$\dot{x}$"],
):
    fig, ax, fs, gs = initialize_figure(
        fig_size=fig_size, ratio=ratio, text_size=1, subplots=(1, 2), grid=True
    )

    if color != None:
        cmap = mpl.cm.get_cmap(color)
        cs = [cmap(i) for i in np.linspace(0.1, 0.9, np.shape(xs)[1])]
    else:
        cs = ["k"] * np.shape(xs)[1]

    # dt = t[1] - t[0]
    # N = len(t)
    x_min, x_max = np.nanmin(xs), np.nanmax(xs)
    # x_range = np.linspace(x_min, x_max, N)
    # t_mesh, x_mesh = np.meshgrid(t[::quiver_step], x_range[::quiver_step])
    # f_arr = f(t_mesh, x_mesh)
    # u = np.full((N // quiver_step, N // quiver_step), dt)
    # v = f_arr * dt
    # norm = np.sqrt(u**2 + v**2)
    # u, v = u / norm, v / norm

    # vels = np.diff(xs, axis=0) / dt
    for i in range(np.shape(xs)[1]):
        ax[0][0].plot(t, xs[:, i], color=cs[i], lw=fs * 0.2, zorder=2)
        ax[0][1].plot(xs[:, i], vs[:, i], color=cs[i], alpha=0.7, lw=fs * 0.2, zorder=0)


    # ax[0][0].quiver(
    #     t_mesh,
    #     x_mesh,
    #     u,
    #     v,
    #     angles="xy",
    #     scale=25,
    #     pivot="mid",
    #     color="w",
    #     alpha=0.2,
    # )

    t_lab, x_lab, v_lab = labels

    ax[0][0].set_xlabel(t_lab, fontsize=fs * 1.5)
    ax[0][0].set_ylabel(x_lab, fontsize=fs * 1.5, rotation=0)
    t_margin = (t[-1] - t[0]) * 0.05
    x_margin = (x_max - x_min) * 0.05
    ax[0][0].set_xlim(t[0] - t_margin, t[-1] + t_margin)
    ax[0][0].set_ylim(x_min - x_margin, x_max + x_margin)
    ax[0][0].set_title("Integral curves", fontsize=fs * 1.5)

    ax[0][1].set_xlabel(x_lab, fontsize=fs * 1.5)
    ax[0][1].set_ylabel(v_lab, fontsize=fs * 1.5, rotation=0)
    # ax[0][1].yaxis.tick_right()
    # ax[0][1].yaxis.set_label_position("right")
    ax[0][1].grid(which="major", linewidth=fs * 0.015)
    ax[0][1].set_title("Phase Space", fontsize=fs * 1.5)

    if ax_equal == True:
        print(t[-1])

        ax[0][0].set_aspect("equal")
        ax[0][0].set_xlim(t[0], t[-1])

    #fig.suptitle(function_to_latex(f), fontsize=fs * 2.5)
