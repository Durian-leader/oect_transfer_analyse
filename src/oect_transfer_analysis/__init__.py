"""OECT Transfer Analysis - Advanced analysis tools for OECT transfer curves"""

__version__ = "1.0.3"
__author__ = "lidonghao"
__email__ = "lidonghao100@outlook.com"

# Core imports
from .data_loader import DataLoader, load_transfer_files
from .time_series import TimeSeriesAnalyzer, TimeSeriesData
from .visualization import Visualizer, plot_transfer_evolution, plot_transfer_comparison
from .utils import check_dependencies, create_example_data, estimate_animation_time, get_system_info, print_system_info, validate_transfer_objects, setup_matplotlib_style, memory_usage_monitor, safe_import, ProgressTracker

# Optional imports with graceful fallback
try:
    from .animation import AnimationGenerator, generate_transfer_animation
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False
    
    # Provide fallback functions
    def generate_transfer_animation(*args, **kwargs):
        raise ImportError(
            "Animation features require additional dependencies. "
            "Install with: pip install oect-transfer-analysis[animation]"
        )
    
    class AnimationGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Animation features require additional dependencies. "
                "Install with: pip install oect-transfer-analysis[animation]"
            )

# Define what gets imported with "from oect_transfer_analysis import *"
__all__ = [
    # Core classes
    "DataLoader",
    "TimeSeriesAnalyzer", 
    "TimeSeriesData",
    "Visualizer",
    
    # Functions
    "load_transfer_files",
    "plot_transfer_evolution",
    "plot_transfer_comparison",
    "generate_transfer_animation",
    "check_dependencies",
    "create_example_data", 
    "estimate_animation_time", 
    "get_system_info", 
    "print_system_info", 
    "validate_transfer_objects", 
    "setup_matplotlib_style", 
    "memory_usage_monitor", 
    "safe_import", 
    "ProgressTracker",
    
    # Optional classes
    "AnimationGenerator",
    
    # Constants
    "ANIMATION_AVAILABLE",
]

def get_version():
    """Return the package version."""
    return __version__

def show_info():
    """Display package information."""
    print(f"OECT Transfer Analysis v{__version__}")
    print(f"Author: {__author__} ({__email__})")
    print(f"Animation support: {'Available' if ANIMATION_AVAILABLE else 'Not available'}")
    print("For more information, visit: https://github.com/yourusername/oect-transfer-analysis")

def quick_start_example():
    """Print a quick start example."""
    example = '''
# OECT Transfer Analysis Quick Start

from oect_transfer_analysis import DataLoader, TimeSeriesAnalyzer, Visualizer

# 1. Load data
loader = DataLoader("path/to/csv/files")
transfer_objects = loader.load_all_files(device_type="N")

# 2. Time series analysis  
analyzer = TimeSeriesAnalyzer(transfer_objects)
time_series = analyzer.extract_time_series()

# 3. Create visualizations
viz = Visualizer()

# Evolution plot with custom colors
viz.plot_evolution(transfer_objects, colormap="black_to_red")

# Compare specific time points
viz.plot_comparison(transfer_objects, indices=[0, 25, 50], 
                   labels=["Initial", "Middle", "Final"])

# 4. Generate animation (if dependencies installed)
if ANIMATION_AVAILABLE:
    viz.generate_animation(transfer_objects, "device_evolution.mp4", fps=30)

# 5. Statistical analysis
stats = analyzer.get_summary_statistics()
drift = analyzer.detect_drift("gm_max_raw", threshold=0.05)
print("Drift detected:", drift["drift_detected"])
'''
    print(example)

# Compatibility with legacy import patterns
# Allow users to import specific modules directly
from . import data_loader
from . import time_series  
from . import visualization
from . import utils

if ANIMATION_AVAILABLE:
    from . import animation