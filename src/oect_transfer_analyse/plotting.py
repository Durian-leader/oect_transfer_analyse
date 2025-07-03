# plotting.py
"""
Advanced plotting and visualization for OECT transfer curves

This module provides enhanced plotting capabilities built on top of the core
oect-transfer package, with publication-ready figures and advanced analysis plots.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import os

# Import from core package
try:
    import oect_transfer as ot
except ImportError:
    raise ImportError("oect-transfer is required. Install with: pip install oect-transfer")

# Handle optional matplotlib dependency
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    """Check if matplotlib is available and raise informative error if not."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for plotting functionality. "
            "Install it with: pip install matplotlib"
        )


def plot_transfer_evolution(
    transfer_objects: List[Dict[str, Any]],
    label: str = 'Device',
    data_type: str = 'raw',
    y_scale: str = 'log',
    use_abs_current: bool = True,
    colormap: str = 'viridis',
    style: str = 'standard',
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 150,
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Enhanced transfer curve evolution plot with multiple style options.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    label : str, default 'Device'
        Device label for title and filename
    data_type : str, default 'raw'
        Data type: 'raw', 'forward', or 'reverse'
    y_scale : str, default 'log'
        Y-axis scale: 'log' or 'linear'
    use_abs_current : bool, default True
        Whether to use absolute current values
    colormap : str, default 'viridis'
        Matplotlib colormap name
    style : str, default 'standard'
        Plot style: 'standard', 'publication', 'minimal', or 'detailed'
    figsize : Tuple[float, float], default (12, 8)
        Figure size in inches
    dpi : int, default 150
        Figure resolution
    save_path : str, optional
        Path to save the figure
    **kwargs
        Additional matplotlib parameters
        
    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects
        
    Examples
    --------
    >>> # Publication-ready plot
    >>> fig, ax = plot_transfer_evolution(
    ...     transfer_objects,
    ...     style='publication',
    ...     colormap='plasma',
    ...     save_path='figure_1.png'
    ... )
    """
    _check_matplotlib()
    
    if style == 'publication':
        # Publication style settings
        plt.rcParams.update({
            'font.size': 14,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.minor.width': 1,
            'ytick.minor.width': 1,
            'figure.dpi': dpi
        })
        figsize = (10, 6)
    elif style == 'minimal':
        figsize = (8, 5)
    elif style == 'detailed':
        figsize = (14, 10)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    n_curves = len(transfer_objects)
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=n_curves-1)
    
    # Plot parameters
    plot_kwargs = {
        'linewidth': 2 if style == 'publication' else 1.5,
        'alpha': 0.9 if style == 'publication' else 0.8
    }
    plot_kwargs.update(kwargs)
    
    for i, transfer_item in enumerate(transfer_objects):
        try:
            transfer_obj = transfer_item['transfer']
            
            # Get data based on type
            if data_type == 'raw':
                voltage = transfer_obj.Vg.raw
                current = transfer_obj.I.raw
            elif data_type == 'forward':
                voltage = transfer_obj.Vg.forward
                current = transfer_obj.I.forward
            elif data_type == 'reverse':
                voltage = transfer_obj.Vg.reverse
                current = transfer_obj.I.reverse
            else:
                continue
            
            current_plot = np.abs(current) if use_abs_current else current
            color = cmap(norm(i))
            
            if y_scale == 'log':
                ax.semilogy(voltage, current_plot, color=color, **plot_kwargs)
            else:
                ax.plot(voltage, current_plot, color=color, **plot_kwargs)
                
        except Exception as e:
            print(f"Warning: Failed to plot curve {i}: {e}")
    
    # Styling based on style parameter
    if style == 'publication':
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=16, fontweight='bold')
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=14)
    elif style == 'minimal':
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=12)
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:  # standard or detailed
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=14)
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Measurement Index', fontsize=12, rotation=270, labelpad=20)
    
    # Title
    scale_text = "Log Scale" if y_scale == 'log' else "Linear Scale"
    if style != 'minimal':
        ax.set_title(f'{label} Transfer Evolution ({scale_text})', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig, ax


def plot_transfer_comparison(
    transfer_objects: List[Dict[str, Any]],
    indices: List[int],
    labels: Optional[List[str]] = None,
    data_type: str = 'raw',
    y_scale: str = 'log',
    use_abs_current: bool = True,
    style: str = 'standard',
    colormap: str = 'Set1',
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Enhanced transfer curve comparison with advanced styling options.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    indices : List[int]
        Indices of curves to compare
    labels : List[str], optional
        Custom labels for each curve
    data_type : str, default 'raw'
        Data type: 'raw', 'forward', or 'reverse'
    y_scale : str, default 'log'
        Y-axis scale: 'log' or 'linear'
    use_abs_current : bool, default True
        Whether to use absolute current values
    style : str, default 'standard'
        Plot style: 'standard', 'publication', 'minimal'
    colormap : str, default 'Set1'
        Matplotlib colormap name
    figsize : Tuple[float, float], default (10, 6)
        Figure size
    save_path : str, optional
        Path to save figure
    **kwargs
        Additional matplotlib parameters
        
    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects
    """
    _check_matplotlib()
    
    # Validate inputs
    if not indices:
        raise ValueError("indices list is empty")
    
    if not all(0 <= idx < len(transfer_objects) for idx in indices):
        raise ValueError("All indices must be within valid range")
    
    if style == 'publication':
        plt.rcParams.update({
            'font.size': 14,
            'axes.linewidth': 1.5,
        })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / max(1, len(indices) - 1)) for i in range(len(indices))]
    
    # Plot parameters
    plot_kwargs = {
        'linewidth': 3 if style == 'publication' else 2,
        'alpha': 0.9,
        'marker': 'o' if style == 'publication' else None,
        'markersize': 3 if style == 'publication' else 0,
        'markevery': 10 if style == 'publication' else 1
    }
    plot_kwargs.update(kwargs)
    
    for i, idx in enumerate(indices):
        try:
            transfer_item = transfer_objects[idx]
            transfer_obj = transfer_item['transfer']
            
            # Get data
            if data_type == 'raw':
                voltage = transfer_obj.Vg.raw
                current = transfer_obj.I.raw
            elif data_type == 'forward':
                voltage = transfer_obj.Vg.forward
                current = transfer_obj.I.forward
            elif data_type == 'reverse':
                voltage = transfer_obj.Vg.reverse
                current = transfer_obj.I.reverse
            else:
                continue
            
            current_plot = np.abs(current) if use_abs_current else current
            
            # Set label
            if labels and i < len(labels):
                label = labels[i]
            else:
                label = transfer_item.get('filename', f'Curve {idx}')
            
            # Plot
            if y_scale == 'log':
                ax.semilogy(voltage, current_plot, color=colors[i], 
                           label=label, **plot_kwargs)
            else:
                ax.plot(voltage, current_plot, color=colors[i], 
                       label=label, **plot_kwargs)
                
        except Exception as e:
            print(f"Warning: Failed to plot curve at index {idx}: {e}")
    
    # Styling
    if style == 'publication':
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=16, fontweight='bold')
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=16, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
    elif style == 'minimal':
        ax.set_xlabel('$V_{GS}$ (V)')
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label)
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:  # standard
        ax.set_xlabel('$V_{GS}$ (V)', fontsize=14)
        y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
        ax.set_ylabel(y_label, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    scale_text = "Log Scale" if y_scale == 'log' else "Linear Scale"
    if style != 'minimal':
        ax.set_title(f'Transfer Curves Comparison ({scale_text})', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    return fig, ax


def plot_single_transfer(
    transfer_obj: Any,
    label: str = 'Device',
    data_types: List[str] = ['raw'],
    y_scale: str = 'log',
    use_abs_current: bool = True,
    style: str = 'standard',
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    show_analysis: bool = False
) -> Tuple[Any, Any]:
    """
    Enhanced single transfer curve plot with optional analysis overlay.
    
    Parameters
    ----------
    transfer_obj : Transfer
        Single Transfer object
    label : str, default 'Device'
        Device label
    data_types : List[str], default ['raw']
        Data types to plot
    y_scale : str, default 'log'
        Y-axis scale
    use_abs_current : bool, default True
        Whether to use absolute current
    style : str, default 'standard'
        Plot style
    figsize : Tuple[float, float], default (8, 6)
        Figure size
    save_path : str, optional
        Save path
    show_analysis : bool, default False
        Whether to overlay analysis points (Von, gm_max, etc.)
        
    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'raw': 'blue', 'forward': 'red', 'reverse': 'green'}
    
    for data_type in data_types:
        if data_type == 'raw':
            voltage = transfer_obj.Vg.raw
            current = transfer_obj.I.raw
        elif data_type == 'forward':
            voltage = transfer_obj.Vg.forward
            current = transfer_obj.I.forward
        elif data_type == 'reverse':
            voltage = transfer_obj.Vg.reverse
            current = transfer_obj.I.reverse
        else:
            continue
        
        current_plot = np.abs(current) if use_abs_current else current
        color = colors.get(data_type, 'black')
        
        if y_scale == 'log':
            ax.semilogy(voltage, current_plot, color=color, 
                       label=data_type.capitalize(), linewidth=2, alpha=0.8)
        else:
            ax.plot(voltage, current_plot, color=color, 
                   label=data_type.capitalize(), linewidth=2, alpha=0.8)
    
    # Add analysis points if requested
    if show_analysis:
        try:
            # Mark Von
            von_voltage = transfer_obj.Von.raw
            von_current = np.interp(von_voltage, transfer_obj.Vg.raw, 
                                   np.abs(transfer_obj.I.raw) if use_abs_current else transfer_obj.I.raw)
            
            if y_scale == 'log':
                ax.semilogy(von_voltage, von_current, 'ro', markersize=8, 
                           label=f'$V_{{on}}$ = {von_voltage:.3f} V')
            else:
                ax.plot(von_voltage, von_current, 'ro', markersize=8, 
                       label=f'$V_{{on}}$ = {von_voltage:.3f} V')
            
            # Add text annotation
            ax.annotate(f'$V_{{on}}$', xy=(von_voltage, von_current),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                       
        except Exception as e:
            print(f"Warning: Could not add analysis markers: {e}")
    
    # Styling
    ax.set_xlabel('$V_{GS}$ (V)', fontsize=12)
    y_label = '$|I_{DS}|$ (A)' if use_abs_current else '$I_{DS}$ (A)'
    ax.set_ylabel(y_label, fontsize=12)
    
    scale_text = "Log Scale" if y_scale == 'log' else "Linear Scale"
    ax.set_title(f'{label} Transfer Curve ({scale_text})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if len(data_types) > 1 or show_analysis:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Single transfer plot saved to: {save_path}")
    
    plt.show()
    return fig, ax


def plot_parameter_trends(extractor, 
                         parameters: Optional[List[str]] = None,
                         figsize: Tuple[float, float] = (12, 8),
                         style: str = 'standard',
                         save_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Plot parameter trends over time with trend lines.
    
    Parameters
    ----------
    extractor : TransferTimeSeriesExtractor
        Configured extractor with time-series data
    parameters : List[str], optional
        Parameters to plot
    figsize : Tuple[float, float], default (12, 8)
        Figure size
    style : str, default 'standard'
        Plot style
    save_path : str, optional
        Save path
        
    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects
    """
    _check_matplotlib()
    
    if extractor.time_series_data is None:
        extractor.extract_time_series()
    
    if parameters is None:
        parameters = ['gm_max_raw', 'Von_raw', 'I_max_raw', 'absgm_max_raw']
    
    # Get trend analysis
    trends = extractor.detect_trends(parameters)
    
    fig, axes = plt.subplots(len(parameters), 1, figsize=figsize, sharex=True)
    if len(parameters) == 1:
        axes = [axes]
    
    time_points = extractor.time_series_data.time_points
    
    for i, param in enumerate(parameters):
        param_data = getattr(extractor.time_series_data, param)
        
        # Plot data points
        axes[i].plot(time_points, param_data, 'o-', linewidth=2, markersize=4, 
                    alpha=0.8, label='Data')
        
        # Add trend line if available
        if param in trends and 'error' not in trends[param]:
            trend_data = trends[param]
            trend_line = trend_data['slope'] * time_points + trend_data['intercept']
            
            color = 'red' if trend_data['significant'] else 'gray'
            linestyle = '-' if trend_data['significant'] else '--'
            
            axes[i].plot(time_points, trend_line, color=color, linestyle=linestyle,
                        alpha=0.7, label=f"Trend ({trend_data['relative_slope_percent']:.2f}%/pt)")
        
        # Styling
        axes[i].set_ylabel(_format_parameter_label(param), fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)
        
        # Add trend info as text
        if param in trends and 'error' not in trends[param]:
            trend_info = trends[param]
            status = "📈" if trend_info['trend_direction'] == 'increasing' else "📉"
            strength = trend_info['trend_strength']
            axes[i].text(0.02, 0.98, f"{status} {strength}", 
                        transform=axes[i].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time Point', fontsize=12)
    plt.suptitle('Parameter Trends Over Time', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Parameter trends plot saved to: {save_path}")
    
    plt.show()
    return fig, axes


def plot_drift_analysis(extractor,
                       parameter: str = 'gm_max_raw',
                       window_size: int = 5,
                       figsize: Tuple[float, float] = (12, 6),
                       save_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Plot drift analysis with moving window visualization.
    
    Parameters
    ----------
    extractor : TransferTimeSeriesExtractor
        Configured extractor
    parameter : str, default 'gm_max_raw'
        Parameter to analyze
    window_size : int, default 5
        Moving window size
    figsize : Tuple[float, float], default (12, 6)
        Figure size
    save_path : str, optional
        Save path
        
    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects
    """
    _check_matplotlib()
    
    # Get drift analysis
    drift_result = extractor.detect_drift_advanced(parameter, window_size=window_size)
    
    if 'error' in drift_result:
        print(f"Error in drift analysis: {drift_result['error']}")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    time_points = drift_result['time_points']
    param_data = getattr(extractor.time_series_data, parameter)
    valid_mask = ~np.isnan(param_data)
    param_data = param_data[valid_mask]
    
    # Plot 1: Parameter evolution
    ax1.plot(time_points, param_data, 'o-', linewidth=2, markersize=4, color='blue')
    ax1.set_ylabel(_format_parameter_label(parameter), fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Drift Analysis: {parameter}', fontsize=14)
    
    # Highlight drift events
    for event in drift_result['drift_events']:
        ax1.axvline(event['position'], color='red', linestyle='--', alpha=0.7)
        ax1.text(event['position'], ax1.get_ylim()[1], 
                f"{event['magnitude']:.1%}", rotation=90, 
                verticalalignment='top', color='red')
    
    # Plot 2: Moving window drift
    drift_values = drift_result['drift_values']
    window_positions = np.arange(len(drift_values))
    
    colors = ['red' if abs(d) > drift_result['threshold'] else 'blue' for d in drift_values]
    ax2.bar(window_positions, drift_values * 100, color=colors, alpha=0.7)
    ax2.axhline(drift_result['threshold'] * 100, color='red', linestyle='--', 
               label=f"Threshold ({drift_result['threshold']:.1%})")
    ax2.axhline(-drift_result['threshold'] * 100, color='red', linestyle='--')
    ax2.set_ylabel('Window Drift (%)', fontsize=12)
    ax2.set_xlabel('Window Position', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add summary text
    stability_score = drift_result['stability_score']
    total_drift = drift_result['total_drift'] * 100
    
    summary_text = (f"Total Drift: {total_drift:+.1f}%\n"
                   f"Stability Score: {stability_score:.2f}\n" 
                   f"Drift Events: {len(drift_result['drift_events'])}")
    
    ax1.text(0.02, 0.02, summary_text, transform=ax1.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Drift analysis plot saved to: {save_path}")
    
    plt.show()
    return fig, (ax1, ax2)


def create_publication_plots(transfer_objects: List[Dict[str, Any]],
                           output_dir: str = 'publication_figures',
                           device_label: str = 'Device',
                           formats: List[str] = ['png', 'pdf']) -> List[str]:
    """
    Generate a complete set of publication-ready plots.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        Transfer objects
    output_dir : str, default 'publication_figures'
        Output directory for figures
    device_label : str, default 'Device'
        Device label for figures
    formats : List[str], default ['png', 'pdf']
        File formats to save
        
    Returns
    -------
    List[str]
        List of generated file paths
    """
    _check_matplotlib()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # Set publication style globally
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300
    })
    
    # 1. Transfer evolution
    for fmt in formats:
        save_path = os.path.join(output_dir, f'{device_label}_evolution.{fmt}')
        plot_transfer_evolution(transfer_objects, label=device_label, 
                              style='publication', save_path=save_path)
        generated_files.append(save_path)
    
    # 2. Initial vs final comparison
    if len(transfer_objects) > 1:
        for fmt in formats:
            save_path = os.path.join(output_dir, f'{device_label}_comparison.{fmt}')
            plot_transfer_comparison(transfer_objects, [0, -1], 
                                   labels=['Initial', 'Final'],
                                   style='publication', save_path=save_path)
            generated_files.append(save_path)
    
    # 3. Single curve with analysis
    if transfer_objects:
        for fmt in formats:
            save_path = os.path.join(output_dir, f'{device_label}_single.{fmt}')
            plot_single_transfer(transfer_objects[0]['transfer'], 
                               label=device_label, style='publication',
                               show_analysis=True, save_path=save_path)
            generated_files.append(save_path)
    
    print(f"📊 Generated {len(generated_files)} publication figures in {output_dir}/")
    return generated_files


def _format_parameter_label(param_name: str) -> str:
    """Format parameter name for display in plots."""
    label_map = {
        'gm_max_raw': '$g_m^{max}$ (S)',
        'gm_max_forward': '$g_m^{max}$ Forward (S)',
        'gm_max_reverse': '$g_m^{max}$ Reverse (S)',
        'I_max_raw': '$I_{max}$ (A)',
        'I_max_forward': '$I_{max}$ Forward (A)',
        'I_max_reverse': '$I_{max}$ Reverse (A)',
        'I_min_raw': '$I_{min}$ (A)',
        'I_min_forward': '$I_{min}$ Forward (A)',
        'I_min_reverse': '$I_{min}$ Reverse (A)',
        'Von_raw': '$V_{on}$ (V)',
        'Von_forward': '$V_{on}$ Forward (V)',
        'Von_reverse': '$V_{on}$ Reverse (V)',
        'absgm_max_raw': '$|g_m|^{max}$ (S)',
        'absgm_max_forward': '$|g_m|^{max}$ Forward (S)',
        'absgm_max_reverse': '$|g_m|^{max}$ Reverse (S)',
        'absI_max_raw': '$|I|_{max}$ (A)',
        'absI_max_forward': '$|I|_{max}$ Forward (A)',
        'absI_max_reverse': '$|I|_{max}$ Reverse (A)',
        'absI_min_raw': '$|I|_{min}$ (A)',
        'absI_min_forward': '$|I|_{min}$ Forward (A)',
        'absI_min_reverse': '$|I|_{min}$ Reverse (A)'
    }
    
    return label_map.get(param_name, param_name)