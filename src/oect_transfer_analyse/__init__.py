"""
OECT Transfer Curve Advanced Analysis and Visualization

An advanced analysis and visualization package built on top of oect-transfer.
Provides time-series analysis, animation generation, and enhanced plotting capabilities
for Organic Electrochemical Transistor (OECT) transfer curve data.

Core Dependencies:
    oect-transfer: Core transfer curve analysis
    
Optional Dependencies:
    matplotlib: For plotting and visualization
    opencv-python: For animation generation
    Pillow: For image processing in animations

Basic Usage:
    >>> import oect_transfer_analyse as ota
    >>> import oect_transfer as ot
    >>> 
    >>> # Load data using core package
    >>> transfer_objects = ot.load_all_transfer_files('data/', 'N')
    >>> 
    >>> # Advanced analysis using this package
    >>> extractor = ota.analyze_transfer_stability(transfer_objects)
    >>> ota.plot_transfer_evolution(transfer_objects)
    >>> ota.generate_transfer_animation(transfer_objects, 'evolution.mp4')

Package Structure:
    analysis: Time-series analysis and drift detection
    plotting: Enhanced visualization capabilities  
    animation: Video generation for transfer curve evolution
    workflows: Pre-defined analysis workflows
"""

__version__ = "1.0.0"
__author__ = "lidonghao"
__email__ = "lidonghao100@outlook.com"

# Check core dependency
try:
    import oect_transfer
    _CORE_AVAILABLE = True
    __core_version__ = oect_transfer.__version__
except ImportError:
    _CORE_AVAILABLE = False
    __core_version__ = "not installed"
    raise ImportError(
        "oect-transfer is required but not installed. "
        "Install it with: pip install oect-transfer"
    )

# Import core functionality for convenience
from oect_transfer import (
    Transfer, Sequence, Point,
    load_all_transfer_files, analyze_transfer_batch, show_file_sorting_demo
)

# Analysis functions - always available (depend only on core + pandas/numpy)
from .analysis import (
    TransferTimeSeriesExtractor,
    TimeSeriesData,
    analyze_transfer_stability,
    detect_parameter_trends,
    generate_stability_report
)

# Workflows - always available
from .workflows import (
    complete_analysis_workflow,
    quick_stability_check,
    batch_comparison_workflow
)

# Plotting functions - conditionally available
try:
    from .plotting import (
        plot_transfer_evolution,
        plot_single_transfer,
        plot_transfer_comparison,
        plot_parameter_trends,
        plot_drift_analysis,
        create_publication_plots
    )
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

# Animation functions - conditionally available
try:
    from .animation import (
        generate_transfer_animation,
        create_animation_preview,
        create_parameter_animation,
        batch_animation_generation
    )
    _ANIMATION_AVAILABLE = True
except ImportError:
    _ANIMATION_AVAILABLE = False

# Create placeholder functions for unavailable features
if not _PLOTTING_AVAILABLE:
    def _plotting_not_available(*args, **kwargs):
        raise ImportError(
            "Plotting functionality requires matplotlib. "
            "Install it with: pip install matplotlib"
        )
    
    plot_transfer_evolution = _plotting_not_available
    plot_single_transfer = _plotting_not_available
    plot_transfer_comparison = _plotting_not_available
    plot_parameter_trends = _plotting_not_available
    plot_drift_analysis = _plotting_not_available
    create_publication_plots = _plotting_not_available

if not _ANIMATION_AVAILABLE:
    def _animation_not_available(*args, **kwargs):
        raise ImportError(
            "Animation functionality requires matplotlib and opencv-python. "
            "Install them with: pip install matplotlib opencv-python"
        )
    
    generate_transfer_animation = _animation_not_available
    create_animation_preview = _animation_not_available
    create_parameter_animation = _animation_not_available
    batch_animation_generation = _animation_not_available

# Define what gets imported with "from oect_transfer_analyse import *"
__all__ = [
    # Re-exported from core package
    "Transfer", "Sequence", "Point",
    "load_all_transfer_files", "analyze_transfer_batch", "show_file_sorting_demo",
    
    # Analysis functions
    "TransferTimeSeriesExtractor", "TimeSeriesData", 
    "analyze_transfer_stability", "detect_parameter_trends", "generate_stability_report",
    
    # Workflow functions
    "complete_analysis_workflow", "quick_stability_check", "batch_comparison_workflow",
    
    # Plotting functions (may not be available)
    "plot_transfer_evolution", "plot_single_transfer", "plot_transfer_comparison",
    "plot_parameter_trends", "plot_drift_analysis", "create_publication_plots",
    
    # Animation functions (may not be available)
    "generate_transfer_animation", "create_animation_preview", 
    "create_parameter_animation", "batch_animation_generation",
]


def check_core_available() -> bool:
    """
    Check if core oect-transfer package is available.
    
    Returns
    -------
    bool
        True if oect-transfer is installed and available
    """
    return _CORE_AVAILABLE


def check_plotting_available() -> bool:
    """
    Check if plotting functionality is available.
    
    Returns
    -------
    bool
        True if matplotlib is installed and plotting is available
    """
    return _PLOTTING_AVAILABLE


def check_animation_available() -> bool:
    """
    Check if animation functionality is available.
    
    Returns
    -------
    bool
        True if matplotlib and opencv-python are installed
    """
    return _ANIMATION_AVAILABLE


def get_package_info() -> dict:
    """
    Get comprehensive package information.
    
    Returns
    -------
    dict
        Package information including versions and feature availability
    """
    return {
        "package": "oect-transfer-analyse",
        "version": __version__,
        "author": __author__,
        "core_package": "oect-transfer",
        "core_version": __core_version__,
        "features": {
            "core_analysis": _CORE_AVAILABLE,
            "plotting": _PLOTTING_AVAILABLE,
            "animation": _ANIMATION_AVAILABLE
        }
    }


def print_package_info():
    """Print comprehensive package information."""
    info = get_package_info()
    
    print(f"📊 {info['package']} v{info['version']}")
    print(f"🔧 Built on {info['core_package']} v{info['core_version']}")
    print(f"👨‍💻 Author: {info['author']}")
    print()
    print("🎯 Feature Availability:")
    print(f"   Core Analysis: {'✅' if info['features']['core_analysis'] else '❌'}")
    print(f"   Plotting: {'✅' if info['features']['plotting'] else '❌'}")
    print(f"   Animation: {'✅' if info['features']['animation'] else '❌'}")
    
    if not info['features']['plotting']:
        print("\n💡 Install plotting: pip install matplotlib")
    if not info['features']['animation']:
        print("💡 Install animation: pip install matplotlib opencv-python")


def get_example_usage() -> str:
    """
    Get example usage code for the package.
    
    Returns
    -------
    str
        Example usage code
    """
    example = f"""
# OECT Transfer Analyse v{__version__} - Example Usage

import oect_transfer_analyse as ota

# Basic workflow
transfer_objects = ota.load_all_transfer_files('data/', 'N')
extractor = ota.analyze_transfer_stability(transfer_objects)

# Time-series analysis
trends = ota.detect_parameter_trends(extractor)
report = ota.generate_stability_report(extractor, 'device_report.html')

# Advanced visualization (if matplotlib available)
if ota.check_plotting_available():
    ota.plot_transfer_evolution(transfer_objects, style='publication')
    ota.plot_parameter_trends(extractor, save_path='trends.png')
    ota.create_publication_plots(transfer_objects, output_dir='figures/')

# Animation generation (if dependencies available)
if ota.check_animation_available():
    ota.generate_transfer_animation(transfer_objects, 'evolution.mp4')
    ota.create_parameter_animation(extractor, 'parameters.mp4')

# Pre-defined workflows
results = ota.complete_analysis_workflow(
    'data/', 
    device_type='N',
    output_dir='analysis_results/',
    generate_report=True,
    create_animations=True
)

# Quick stability check
stability_status = ota.quick_stability_check(transfer_objects)
print(f"Device stability: {stability_status}")
"""
    return example


# Package-level convenience functions
def info():
    """Display package information."""
    print_package_info()

def example():
    """Display example usage."""
    print(get_example_usage())