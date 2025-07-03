"""Advanced visualization tools for OECT transfer analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    from oect_transfer import Transfer
except ImportError:
    raise ImportError(
        "oect-transfer package is required. Install with: pip install oect-transfer"
    )


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return True
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def _get_transfer_data(transfer_obj: Transfer, data_type: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract voltage and current data from Transfer object."""
    try:
        if data_type == 'raw':
            return transfer_obj.Vg.raw, transfer_obj.I.raw
        elif data_type == 'forward':
            return transfer_obj.Vg.forward, transfer_obj.I.forward
        elif data_type == 'reverse':
            return transfer_obj.Vg.reverse, transfer_obj.I.reverse
        else:
            return None, None
    except AttributeError:
        return None, None


class Visualizer:
    """Advanced visualizer for OECT transfer analysis."""
    
    def __init__(self):
        """Initialize the visualizer."""
        _check_matplotlib()
    
    def plot_evolution(
        self,
        transfer_objects: List[Dict[str, Any]],
        label: str = 'Device',
        data_type: str = 'raw',
        y_scale: str = 'log',
        use_abs_current: bool = True,
        colormap: str = 'black_to_red',
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 150,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot transfer curve evolution with custom colormap and colorbar.
        
        Parameters
        ----------
        transfer_objects : List[Dict[str, Any]]
            List of transfer objects
        label : str, default 'Device'
            Device label for plot title
        data_type : str, default 'raw'
            Data type: 'raw', 'forward', or 'reverse'
        y_scale : str, default 'log'
            Y-axis scale: 'log' or 'linear'
        use_abs_current : bool, default True
            Whether to use absolute current values
        colormap : str, default 'black_to_red'
            Colormap: 'black_to_red', 'Reds', 'viridis', 'plasma', etc.
        figsize : tuple, default (10, 6)
            Figure size in inches
        dpi : int, default 150
            Figure DPI
        save_path : str, optional
            Path to save figure
        **kwargs
            Additional matplotlib plot parameters
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        if not transfer_objects:
            raise ValueError("transfer_objects list is empty")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        n_curves = len(transfer_objects)
        
        # Setup colormap
        if colormap == 'black_to_red':
            colors_list = ['black', 'red']
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'black_to_red', colors_list, N=256
            )
        else:
            try:
                cmap = plt.get_cmap(colormap)
            except ValueError:
                print(f"Warning: Unknown colormap '{colormap}', using 'Reds'")
                cmap = plt.get_cmap('Reds')
        
        # Create normalization
        norm = mcolors.Normalize(vmin=0, vmax=n_curves-1)
        
        # Set default plot parameters
        plot_kwargs = {'alpha': 0.8, 'linewidth': 1.5}
        plot_kwargs.update(kwargs)
        
        successful_plots = 0
        
        # Plot curves
        for i, transfer_item in enumerate(transfer_objects):
            try:
                transfer_obj = transfer_item['transfer']
                voltage, current = _get_transfer_data(transfer_obj, data_type)
                
                if voltage is None or current is None:
                    print(f"Warning: Could not get data for curve {i}, skipping")
                    continue
                
                # Process current data
                current_plot = np.abs(current) if use_abs_current else current
                
                # Get color
                color = cmap(norm(i))
                
                # Plot based on scale
                if y_scale == 'log':
                    ax.semilogy(voltage, current_plot, color=color, **plot_kwargs)
                else:  # linear
                    ax.plot(voltage, current_plot, color=color, **plot_kwargs)
                
                successful_plots += 1
                
            except Exception as e:
                print(f"Warning: Failed to plot curve {i}: {e}")
                continue
        
        if successful_plots == 0:
            raise ValueError("No curves could be plotted successfully")
        
        # Setup plot appearance
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=14)
        
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=14)
        
        scale_text = "Log Scale" if y_scale == 'log' else "Linear Scale"
        ax.set_title(
            f'{label} Transfer Evolution ({scale_text}, {data_type.title()})', 
            fontsize=16
        )
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Measurement Index', fontsize=12, rotation=270, labelpad=20)
        
        # Set colorbar ticks
        if n_curves > 1:
            cbar.set_ticks([0, n_curves//4, n_curves//2, 3*n_curves//4, n_curves-1])
            cbar.set_ticklabels([
                '0', f'{n_curves//4}', f'{n_curves//2}', 
                f'{3*n_curves//4}', f'{n_curves-1}'
            ])
        
        # Add info text
        ax.text(
            0.02, 0.98, f'0 → {n_curves-1:,} Measurements',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = f'{label}_transfer_evolution_{data_type}_{y_scale}.png'
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Evolution plot saved to: {save_path}")
        plt.show()
        
        return fig, ax
    
    def plot_comparison(
        self,
        transfer_objects: List[Dict[str, Any]],
        indices: List[int],
        labels: Optional[List[str]] = None,
        data_type: str = 'raw',
        y_scale: str = 'log',
        use_abs_current: bool = True,
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 150,
        save_path: Optional[str] = None,
        colormap: str = 'Set1',
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Compare transfer curves at specific indices.
        
        Parameters
        ----------
        transfer_objects : List[Dict[str, Any]]
            List of transfer objects
        indices : List[int]
            List of indices to compare
        labels : List[str], optional
            Custom labels for each curve
        data_type : str, default 'raw'
            Data type: 'raw', 'forward', or 'reverse'
        y_scale : str, default 'log'
            Y-axis scale: 'log' or 'linear'
        use_abs_current : bool, default True
            Whether to use absolute current values
        figsize : tuple, default (10, 6)
            Figure size in inches
        dpi : int, default 150
            Figure DPI
        save_path : str, optional
            Path to save figure
        colormap : str, default 'Set1'
            Matplotlib colormap name
        **kwargs
            Additional matplotlib plot parameters
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        # Validate inputs
        if not transfer_objects:
            raise ValueError("transfer_objects list is empty")
        
        if not indices:
            raise ValueError("indices list is empty")
        
        if not all(isinstance(idx, int) for idx in indices):
            raise ValueError("All indices must be integers")
        
        if not all(0 <= idx < len(transfer_objects) for idx in indices):
            raise ValueError(
                f"All indices must be within range [0, {len(transfer_objects)-1}]. "
                f"Got indices: {indices}"
            )
        
        if data_type not in ['raw', 'forward', 'reverse']:
            raise ValueError("data_type must be 'raw', 'forward', or 'reverse'")
        
        if y_scale not in ['log', 'linear']:
            raise ValueError("y_scale must be 'log' or 'linear'")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get colors
        try:
            cmap = plt.get_cmap(colormap)
            colors = [cmap(i / max(1, len(indices) - 1)) for i in range(len(indices))]
        except ValueError:
            print(f"Warning: Unknown colormap '{colormap}', using 'Set1'")
            cmap = plt.get_cmap('Set1')
            colors = [cmap(i / max(1, len(indices) - 1)) for i in range(len(indices))]
        
        # Set default plot parameters
        plot_kwargs = {'linewidth': 2, 'alpha': 0.8}
        plot_kwargs.update(kwargs)
        
        successful_plots = 0
        
        for i, idx in enumerate(indices):
            try:
                transfer_item = transfer_objects[idx]
                transfer_obj = transfer_item['transfer']
                
                voltage, current = _get_transfer_data(transfer_obj, data_type)
                
                if voltage is None or current is None:
                    print(f"Warning: Could not get data for index {idx}, skipping")
                    continue
                
                # Process current data
                current_plot = np.abs(current) if use_abs_current else current
                
                # Set label
                if labels and i < len(labels):
                    label = labels[i]
                else:
                    filename = transfer_item.get('filename', f'Curve_{idx}')
                    label = filename
                
                # Plot
                if y_scale == 'log':
                    ax.semilogy(voltage, current_plot,
                               color=colors[i], label=label, **plot_kwargs)
                else:  # linear
                    ax.plot(voltage, current_plot,
                           color=colors[i], label=label, **plot_kwargs)
                
                successful_plots += 1
                
            except Exception as e:
                print(f"Warning: Failed to plot curve at index {idx}: {e}")
                continue
        
        if successful_plots == 0:
            raise ValueError("No curves could be plotted successfully")
        
        # Setup plot appearance
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=14)
        
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=14)
        
        scale_text = "Log Scale" if y_scale == 'log' else "Linear Scale"
        ax.set_title(
            f'Transfer Curves Comparison ({scale_text}, {data_type.title()})',
            fontsize=16
        )
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add comparison info
        indices_str = ', '.join(map(str, indices))
        ax.text(
            0.02, 0.02, f'Compared indices: {indices_str}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            indices_str_short = '_'.join(map(str, indices[:3]))
            if len(indices) > 3:
                indices_str_short += f'_plus{len(indices)-3}more'
            save_path = f'transfer_comparison_{data_type}_{y_scale}_{indices_str_short}.png'
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        plt.show()
        
        return fig, ax
    
    def generate_animation(self, transfer_objects: List[Dict[str, Any]], *args, **kwargs):
        """
        Generate transfer evolution animation.
        
        This method requires additional dependencies and will import
        from the animation module when called.
        """
        try:
            from .animation import generate_transfer_animation_optimized
            return generate_transfer_animation_optimized(transfer_objects, *args, **kwargs)
        except ImportError:
            raise ImportError(
                "Animation features require additional dependencies. "
                "Install with: pip install oect-transfer-analysis[animation]"
            )


# Convenience functions for direct use
def plot_transfer_evolution(
    transfer_objects: List[Dict[str, Any]],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience function to plot transfer evolution.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    **kwargs
        Arguments passed to Visualizer.plot_evolution()
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    viz = Visualizer()
    return viz.plot_evolution(transfer_objects, **kwargs)


def plot_transfer_comparison(
    transfer_objects: List[Dict[str, Any]],
    indices: List[int],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience function to plot transfer comparison.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    indices : List[int]
        Indices to compare
    **kwargs
        Arguments passed to Visualizer.plot_comparison()
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    viz = Visualizer()
    return viz.plot_comparison(transfer_objects, indices, **kwargs)


# Additional visualization utilities
def create_subplot_grid(
    transfer_objects: List[Dict[str, Any]],
    indices: List[int],
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a grid of individual transfer curve plots.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    indices : List[int]
        Indices to plot
    ncols : int, default 3
        Number of columns in grid
    figsize : tuple, optional
        Figure size. If None, auto-calculated
    **kwargs
        Additional plot parameters
        
    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes]]
        Figure and list of axes objects
    """
    nrows = len(indices) // ncols + (1 if len(indices) % ncols else 0)
    
    if figsize is None:
        figsize = (ncols * 4, nrows * 3)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=kwargs.get('dpi', 150))
    
    if nrows == 1:
        axes = axes if ncols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    plot_kwargs = {'linewidth': 2, 'color': 'blue'}
    plot_kwargs.update(kwargs)
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        try:
            transfer_item = transfer_objects[idx]
            transfer_obj = transfer_item['transfer']
            
            voltage, current = _get_transfer_data(transfer_obj, 'raw')
            current_plot = np.abs(current)
            
            axes[i].semilogy(voltage, current_plot, **plot_kwargs)
            axes[i].set_title(f"Index {idx}: {transfer_item.get('filename', '')}")
            axes[i].set_xlabel('$V_{GS}$ (V)')
            axes[i].set_ylabel('$|I_{DS}|$ (A)')
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", transform=axes[i].transAxes,
                        ha='center', va='center')
    
    # Hide unused axes
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes