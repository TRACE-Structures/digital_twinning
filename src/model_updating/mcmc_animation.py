""" Animation of the MCMC random walk of a DigitalTwin update, in the layout of MCMC_plot """

import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


def animate_MCMC_walk(DigitalTwin, X_train, filename=None, include_burn=True, step=1, max_frames=300,
                      tail=10, extra_point=None, extra_point_name='Extra point', map=False,
                      custom_xlim=None, custom_ylim=None, fps=12, end_pause=3.0, dpi=100, n_grid=150,
                      max_prior_samples=1000, max_kde_samples=3000, running_mean='all', random_seed=0):
    ''' Animate the random walk of the MCMC walkers in a pairplot-like grid.

        Off-diagonal panels: prior samples (grey, static), current walker positions
        (blue) with fading tails of their last steps, and the running mean of the
        walkers (red). Diagonal panels: prior density (grey, static), the density (KDE)
        of the walkers - current positions during burn-in, all samples drawn so far
        during sampling - and the running mean as a vertical red line.

        Parameters
        ----------
        DigitalTwin : DigitalTwin
            Updated DigitalTwin (with sampler). The burn-in is only available if the
            update stored it in DigitalTwin.burn_chain.
        X_train : array_like or DataFrame
            Samples representing the prior (physical parameters).
        filename : str, optional
            Output file, '.gif' (Pillow) or '.mp4' (ffmpeg). If given, the frames are
            rendered with blitting (static background drawn once), which is much
            faster than FuncAnimation.save, and the file name is returned. If None, a
            FuncAnimation is returned for interactive display.
        include_burn : bool, optional
            Start the animation with the burn-in steps, by default True.
        step : int, optional
            Use every step-th MCMC iteration as a frame, by default 1.
        max_frames : int or None, optional
            Upper bound on the number of frames; step is increased if needed, by default 300.
        tail : int, optional
            Number of previous steps drawn as a trail behind each walker, by default 10
            (0 disables the trails).
        extra_point : DataFrame or array_like, optional
            Additional point to show (e.g. the true parameters).
        extra_point_name : str, optional
            Label for the extra point in the legend.
        map : bool, optional
            Show the final MAP estimate (DigitalTwin.get_MAP()) as a static marker,
            by default False.
        custom_xlim, custom_ylim : list or dict, optional
            Axis limits per parameter, as in MCMC_plot.
        fps : int, optional
            Frames per second, by default 12.
        end_pause : float, optional
            Seconds the last frame is held before the animation starts over, by
            default 3. GIF and interactive display loop automatically; an MP4 gets
            the held frame appended (looping is up to the player).
        dpi : int, optional
            Resolution of the frames, by default 100.
        n_grid : int, optional
            Number of grid points of the diagonal densities, by default 150.
        max_prior_samples : int, optional
            Maximum number of prior points drawn, by default 1000.
        max_kde_samples : int, optional
            Maximum number of samples used for a diagonal density (thinned), by default 3000.
        running_mean : {'all', 'sampling'}, optional
            'all' (default): running mean of all steps shown so far, from the first
            burn-in step on. 'sampling': running mean of the sampling steps only,
            shown from the end of the burn-in; its final value is the posterior mean
            of MCMC_plot.
        random_seed : int, optional
            Seed for subsampling the prior.

        Returns
        -------
        str or matplotlib.animation.FuncAnimation
            The file name if filename is given, otherwise a FuncAnimation. In Jupyter,
            show a saved file with IPython.display.Image('walk.gif') or
            IPython.display.Video('walk.mp4'). '''

    Q = DigitalTwin.model.Q
    names = Q.variable_names()
    d = len(names)

    # ------------------------------------------------ chains in physical parameters
    def to_physical(chain):
        T, W, _ = chain.shape
        return np.asarray(Q.germ2variable(chain.reshape(-1, d)), dtype=float).reshape(T, W, d)

    main = to_physical(np.asarray(DigitalTwin.sampler.get_chain()))
    burn = getattr(DigitalTwin, 'burn_chain', None)
    if include_burn and burn is not None and len(burn) > 0:
        burn = to_physical(np.asarray(burn))
    else:
        if include_burn:
            print('No burn-in stored in DigitalTwin.burn_chain; animating the sampling steps only.')
        burn = np.empty((0,) + main.shape[1:])
    chain = np.concatenate([burn, main], axis=0)          # (T, W, d)
    n_burn, n_steps, n_walkers = burn.shape[0], chain.shape[0], chain.shape[1]

    if max_frames is not None and n_steps / step > max_frames:
        step = int(np.ceil(n_steps / max_frames))
    frames = np.arange(0, n_steps, step)
    if frames[-1] != n_steps - 1:
        frames = np.append(frames, n_steps - 1)

    # running mean of the walkers: over all steps, or over the sampling phase only
    if running_mean not in ('all', 'sampling'):
        raise ValueError("running_mean must be 'all' or 'sampling'")
    mean_start = 0 if running_mean == 'all' else n_burn
    mean_src = chain[mean_start:]
    cum_mean = np.cumsum(mean_src.mean(axis=1), axis=0) / np.arange(1, mean_src.shape[0] + 1)[:, None]

    # ------------------------------------------------ static data, limits, densities
    rng = np.random.default_rng(random_seed)
    prior = np.asarray(X_train, dtype=float)
    if prior.shape[0] > max_prior_samples:
        prior = prior[rng.choice(prior.shape[0], max_prior_samples, replace=False)]

    bounds = Q.get_bounds()

    def limits(custom, j):
        if custom is None:
            return tuple(bounds[j])
        value = custom.get(names[j]) if isinstance(custom, dict) else custom[j]
        return tuple(bounds[j]) if value is None else tuple(value)

    xlims = [limits(custom_xlim, j) for j in range(d)]
    ylims = [limits(custom_ylim, i) for i in range(d)]
    grids = [np.linspace(*xlims[j], n_grid) for j in range(d)]

    def density(x, grid):
        x = x[np.isfinite(x)]
        if x.size > max_kde_samples:
            x = x[np.linspace(0, x.size - 1, max_kde_samples).astype(int)]
        if x.size < 2 or np.ptp(x) == 0:
            return np.zeros_like(grid)
        return gaussian_kde(x)(grid)

    def frame_samples(t):
        if t >= n_burn:
            return main[:t - n_burn + 1].reshape(-1, d)    # all samples drawn so far
        return chain[t]                                    # current walker positions

    # densities of all frames up front -> fixed diagonal scale (needed for blitting)
    frame_dens = {t: [density(frame_samples(t)[:, i], grids[i]) for i in range(d)] for t in frames}
    prior_dens = [density(prior[:, i], grids[i]) for i in range(d)]
    diag_max = [1.05 * max(prior_dens[i].max(), max(fd[i].max() for fd in frame_dens.values())) or 1.0
                for i in range(d)]

    extra_p = None
    if extra_point is not None:
        extra_p = (extra_point.iloc[0].to_numpy() if isinstance(extra_point, pd.DataFrame)
                   else np.atleast_2d(np.asarray(extra_point, dtype=float))[0])
    map_p = DigitalTwin.get_MAP().iloc[0].to_numpy() if map else None

    prior_color = '#b8b8b8'
    posterior_color = '#1a80bb'
    MAP_color = '#ea801c'
    tail_rgb = matplotlib.colors.to_rgb(posterior_color)

    # ------------------------------------------------ figure in the pairplot layout
    fig, axes = plt.subplots(d, d, figsize=(2.5 * d, 2.5 * d), squeeze=False, dpi=dpi)
    dynamic = []                                           # artists redrawn every frame
    walkers, tails, means, dens, mean_lines = {}, {}, {}, {}, {}

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            ax.set_xlim(xlims[j])
            if i != j:
                ax.set_ylim(ylims[i])
                ax.scatter(prior[:, j], prior[:, i], s=6, color=prior_color, alpha=0.5, linewidth=0, zorder=1)
                if tail > 0:
                    tails[i, j] = LineCollection([], linewidths=0.8, zorder=2, animated=True)
                    ax.add_collection(tails[i, j])
                    dynamic.append((ax, tails[i, j]))
                walkers[i, j] = ax.scatter(np.full(n_walkers, np.nan), np.full(n_walkers, np.nan),
                                           s=12, color=posterior_color, zorder=3, animated=True)
                means[i, j], = ax.plot([], [], 'o', color='red', ms=4, zorder=5, animated=True)
                dynamic += [(ax, walkers[i, j]), (ax, means[i, j])]
                if map_p is not None:
                    ax.plot(map_p[j], map_p[i], 'o', color=MAP_color, ms=4, zorder=6)
                if extra_p is not None:
                    ax.plot(extra_p[j], extra_p[i], 'o', color='purple', ms=4, zorder=6)
            else:
                g = grids[i]
                ax.set_ylim(0, diag_max[i])
                ax.fill_between(g, prior_dens[i], color=prior_color, alpha=0.3, lw=0, zorder=1)
                ax.plot(g, prior_dens[i], color=prior_color, lw=1, zorder=1)
                fill, = ax.fill(np.r_[g, g[::-1]], np.zeros(2 * g.size), color=posterior_color,
                                alpha=0.2, lw=0, zorder=2, animated=True)
                line, = ax.plot(g, np.zeros_like(g), color=posterior_color, lw=1.2, zorder=2, animated=True)
                dens[i] = (fill, line)
                mean_lines[i] = ax.axvline(np.nan, color='red', lw=1.2, zorder=4, animated=True)
                dynamic += [(ax, fill), (ax, line), (ax, mean_lines[i])]
                if map_p is not None:
                    ax.axvline(map_p[i], color=MAP_color, zorder=3)
                if extra_p is not None:
                    ax.axvline(extra_p[i], color='purple', zorder=3)
                ax.set_yticks([])
            # outer labels only, as in a pairplot
            if i == d - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(names[i])
            elif j != 0 and i != j:
                ax.set_yticklabels([])
    axes[0, 0].set_ylabel(names[0])

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=prior_color, markersize=8, label='Prior'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=posterior_color, markersize=8, label='Walkers'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Running mean'),
    ]
    if map_p is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=MAP_color,
                                      markersize=8, label='MAP Point (final)'))
    if extra_p is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                                      markersize=8, label=extra_point_name))
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.995, 0.995),
               ncol=min(3, len(legend_elements)), fontsize=8, frameon=False)
    title = fig.text(0.01, 0.985, '', ha='left', va='top', fontsize=11, animated=True)
    dynamic.append((None, title))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    tail_alpha = np.linspace(0.05, 0.5, max(tail, 1))

    # ------------------------------------------------ frame update
    def update(t):
        cur = chain[t]
        if t >= n_burn:
            k = t - n_burn
            title.set_text(f'MCMC sampling: step {k + 1}/{main.shape[0]}')
        else:
            title.set_text(f'MCMC burn-in: step {t + 1}/{n_burn}')

        t0 = max(0, t - tail)
        for (i, j), sc in walkers.items():
            sc.set_offsets(cur[:, [j, i]])
            if (i, j) in tails:
                pts = chain[t0:t + 1][:, :, [j, i]]                  # (L, W, 2)
                if pts.shape[0] > 1:
                    segs = np.stack([pts[:-1], pts[1:]], axis=2).reshape(-1, 2, 2)
                    colors = np.empty((len(segs), 4))
                    colors[:, :3] = tail_rgb
                    colors[:, 3] = np.repeat(tail_alpha[-(pts.shape[0] - 1):], n_walkers)
                    tails[i, j].set_segments(segs)
                    tails[i, j].set_color(colors)
                else:
                    tails[i, j].set_segments([])
            if t >= mean_start:
                means[i, j].set_data([cum_mean[t - mean_start, j]], [cum_mean[t - mean_start, i]])
            else:
                means[i, j].set_data([], [])

        for i, vline in mean_lines.items():
            x = cum_mean[t - mean_start, i] if t >= mean_start else np.nan
            vline.set_xdata([x, x])

        for i, (fill, line) in dens.items():
            g, dv = grids[i], frame_dens[t][i]
            line.set_ydata(dv)
            fill.set_xy(np.column_stack([np.r_[g, g[::-1]], np.r_[dv, np.zeros_like(g)]]))
        return [a for _, a in dynamic]

    # ------------------------------------------------ interactive: FuncAnimation with blitting
    if filename is None:
        return FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True,
                             init_func=lambda: [a for _, a in dynamic],
                             repeat=True, repeat_delay=int(1000 * end_pause))

    # ------------------------------------------------ file: fast blitted rendering
    canvas = fig.canvas if hasattr(fig.canvas, 'copy_from_bbox') else FigureCanvasAgg(fig)
    canvas.draw()                                       # static background once
    background = canvas.copy_from_bbox(fig.bbox)
    width, height = canvas.get_width_height()

    def render(t):
        canvas.restore_region(background)
        update(t)
        for ax, artist in dynamic:
            (ax if ax is not None else fig).draw_artist(artist)
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

    if filename.lower().endswith('.gif'):
        from PIL import Image
        # shared palette from the first and last frame (burn-in and sampling colours)
        ref = np.concatenate([render(frames[0]), render(frames[-1])], axis=0)
        palette = Image.fromarray(ref).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        images = [Image.fromarray(render(t)).quantize(palette=palette, dither=Image.Dither.NONE)
                  for t in frames]
        durations = [int(round(1000 / fps))] * len(images)
        durations[-1] += int(round(1000 * end_pause))          # hold the last frame
        images[0].save(filename, save_all=True, append_images=images[1:], duration=durations, loop=0)
    else:
        ffmpeg = matplotlib.rcParams['animation.ffmpeg_path']
        if shutil.which(ffmpeg) is None:
            plt.close(fig)
            raise RuntimeError("ffmpeg not found; save as '.gif' or install ffmpeg")
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', filename]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            for t in frames:
                frame = render(t).tobytes()
                proc.stdin.write(frame)
            for _ in range(int(round(end_pause * fps))):      # hold the last frame
                proc.stdin.write(frame)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('ffmpeg failed to write the video')

    plt.close(fig)
    print(f'Animation saved to {filename} ({len(frames)} frames)')
    return filename
