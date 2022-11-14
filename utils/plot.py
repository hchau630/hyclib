import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import statsmodels.api as sm
import numpy as np

def subplots(rows, cols, plot_size=(6.4,4.8), keep_shape=False, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=(plot_size[0]*cols, plot_size[1]*rows), **kwargs)
    if keep_shape:
        axes = axes.reshape((rows, cols))
    return fig, axes

def scaterr(x, y, yerr, ax=None, cap=False, **kwargs):
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    if 'ls' not in kwargs and 'linestyle' not in kwargs:
        kwargs['ls'] = 'none'
    if 'capsize' not in kwargs and cap:
        kwargs['capsize'] = 2.0
    
    if ax is None:
        ax = plt.gca()
    
    return ax.errorbar(x.tolist(), y.tolist(), yerr=yerr.tolist(), **kwargs)

def hide_unused_axes(axes):
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_axis_off()

def plot_ci(fit_res, alpha=0.05, xlim=None, plot_kwargs=None, fill_kwargs=None, ax=None):
    if ax is None:
        ax = plt.gca()
        
    if plot_kwargs is None:
        plot_kwargs = {}
    
    if fill_kwargs is None:
        fill_kwargs = {}
        
    default_plot_kwargs = {
        'color': 'C0',
    }
    default_plot_kwargs.update(plot_kwargs)
    plot_kwargs = default_plot_kwargs
    
    default_fill_kwargs = {
        'alpha': 0.15,
        'facecolor': 'C0',
        'edgecolor': 'none',
    }
    default_fill_kwargs.update(fill_kwargs)
    fill_kwargs = default_fill_kwargs
    
    if xlim is None:
        xlim = ax.get_xlim()
    x = np.linspace(*xlim)
    pred_res = fit_res.get_prediction(sm.add_constant(x))
    summary_frame = pred_res.summary_frame(alpha=alpha)
    
    artists = {}
    artists['line'] = ax.plot(x, summary_frame['mean'], **plot_kwargs)
    artists['fill'] = ax.fill_between(x, summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], **fill_kwargs)
    
    return artists
    
def regplot(x, y, yerr=None, scatter_kwargs=None, ci_kwargs=None, text_kwargs=None, ax=None):
    if ax is None:
        ax = plt.gca()
        
    if scatter_kwargs is None:
        scatter_kwargs = {}
        
    if ci_kwargs is None:
        ci_kwargs = {}
    
    if text_kwargs is None:
        text_kwargs = {}
        
    default_scatter_kwargs = {
        'marker': 'o',
        'ls': 'none',
    }
    default_scatter_kwargs.update(scatter_kwargs)
    scatter_kwargs = default_scatter_kwargs
    
    default_text_kwargs = {
        'loc': 'upper right',
    }
    default_text_kwargs.update(text_kwargs)
    text_kwargs = default_text_kwargs
    
    if yerr is None:
        model = sm.OLS(y,sm.add_constant(x))
    else:
        model = sm.WLS(y,sm.add_constant(x),weights=1/yerr**2)
        
    fit_res = model.fit()
    
    artists = {}
    artists['scatter'] = ax.errorbar(x, y, yerr=yerr, **scatter_kwargs)
    artists['ci'] = plot_ci(fit_res, ax=ax, **ci_kwargs)
    artists['text'] = ax.add_artist(AnchoredText(
        '\n'.join([
            f'm={fit_res.params[1]:.1e}$\pm${fit_res.bse[1]:.1e}',
            f'b={fit_res.params[0]:.1e}$\pm${fit_res.bse[0]:.1e}',
            f'r={fit_res.rsquared**0.5:.2f}, p={fit_res.pvalues[1]:.1e}',
        ]),
        **text_kwargs
    ))
    
    return artists