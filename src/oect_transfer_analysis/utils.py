"""Utility functions for OECT transfer analysis package."""

import sys
import importlib
from typing import Dict, List, Tuple, Optional, Any
import warnings


def check_dependencies(verbose: bool = True) -> Dict[str, bool]:
    """
    Check availability of package dependencies.
    
    Parameters
    ----------
    verbose : bool, default True
        Whether to print dependency status
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        # Core dependencies
        'numpy': False,
        'pandas': False,
        'matplotlib': False,
        'oect_transfer': False,
        
        # Optional dependencies for animation
        'cv2': False,
        'PIL': False,
    }
    
    # Check each dependency
    for dep_name in dependencies:
        try:
            if dep_name == 'cv2':
                import cv2
            elif dep_name == 'PIL':
                from PIL import Image
            elif dep_name == 'oect_transfer':
                from oect_transfer import Transfer
            else:
                importlib.import_module(dep_name)
            dependencies[dep_name] = True
        except ImportError:
            dependencies[dep_name] = False
    
    if verbose:
        print("=== Dependency Check ===")
        print("\nCore dependencies:")
        core_deps = ['numpy', 'pandas', 'matplotlib', 'oect_transfer']
        for dep in core_deps:
            status = "✓ Available" if dependencies[dep] else "✗ Missing"
            print(f"  {dep}: {status}")
        
        print("\nOptional dependencies (for animation):")
        optional_deps = ['cv2', 'PIL']
        for dep in optional_deps:
            status = "✓ Available" if dependencies[dep] else "✗ Missing"
            print(f"  {dep}: {status}")
        
        # Check if animation is available
        animation_available = all(dependencies[dep] for dep in optional_deps)
        print(f"\nAnimation support: {'✓ Available' if animation_available else '✗ Not available'}")
        
        if not animation_available:
            print("  To enable animation: pip install oect-transfer-analysis[animation]")
    
    return dependencies


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns
    -------
    Dict[str, Any]
        System information dictionary
    """
    info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'dependencies': check_dependencies(verbose=False)
    }
    
    # Try to get package versions
    version_info = {}
    packages = ['numpy', 'pandas', 'matplotlib', 'oect_transfer']
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            version_info[package] = version
        except ImportError:
            version_info[package] = 'Not installed'
    
    # Special handling for cv2
    try:
        import cv2
        version_info['opencv'] = cv2.__version__
    except ImportError:
        version_info['opencv'] = 'Not installed'
    
    # Special handling for PIL
    try:
        from PIL import Image
        version_info['PIL'] = Image.__version__
    except ImportError:
        version_info['PIL'] = 'Not installed'
    
    info['package_versions'] = version_info
    
    return info


def print_system_info():
    """Print detailed system information."""
    info = get_system_info()
    
    print("=== System Information ===")
    print(f"Python version: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    
    print("\n=== Package Versions ===")
    for package, version in info['package_versions'].items():
        print(f"  {package}: {version}")
    
    print("\n=== Dependency Status ===")
    deps = info['dependencies']
    for dep_name, available in deps.items():
        status = "Available" if available else "Missing"
        print(f"  {dep_name}: {status}")


def validate_transfer_objects(
    transfer_objects: List[Dict[str, Any]],
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate transfer objects list for common issues.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects to validate
    verbose : bool, default True
        Whether to print validation results
        
    Returns
    -------
    Tuple[bool, List[str]]
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if list is empty
    if not transfer_objects:
        issues.append("Transfer objects list is empty")
        if verbose:
            print("❌ Validation failed: Transfer objects list is empty")
        return False, issues
    
    # Check each object
    required_keys = ['transfer', 'filename']
    
    for i, obj in enumerate(transfer_objects):
        # Check if object is dictionary
        if not isinstance(obj, dict):
            issues.append(f"Object {i} is not a dictionary")
            continue
        
        # Check required keys
        for key in required_keys:
            if key not in obj:
                issues.append(f"Object {i} missing required key '{key}'")
        
        # Check if transfer object has required attributes
        if 'transfer' in obj:
            transfer = obj['transfer']
            required_attrs = ['Vg', 'I', 'gm_max', 'I_max', 'I_min', 'Von']
            
            for attr in required_attrs:
                if not hasattr(transfer, attr):
                    issues.append(f"Transfer object {i} missing attribute '{attr}'")
    
    is_valid = len(issues) == 0
    
    if verbose:
        if is_valid:
            print(f"✅ Validation passed: {len(transfer_objects)} transfer objects are valid")
        else:
            print(f"❌ Validation failed with {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
    
    return is_valid, issues


def setup_matplotlib_style(style: str = 'seaborn-v0_8', **kwargs):
    """
    Setup matplotlib style for consistent plotting.
    
    Parameters
    ----------
    style : str, default 'seaborn-v0_8'
        Matplotlib style to use
    **kwargs
        Additional rcParams to set
    """
    try:
        import matplotlib.pyplot as plt
        
        # Try to set style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback styles
            fallback_styles = ['seaborn', 'ggplot', 'default']
            for fallback in fallback_styles:
                try:
                    plt.style.use(fallback)
                    print(f"Using fallback style: {fallback}")
                    break
                except OSError:
                    continue
        
        # Set additional parameters
        default_params = {
            'figure.figsize': (10, 6),
            'figure.dpi': 150,
            'font.size': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
        }
        
        # Merge with user parameters
        default_params.update(kwargs)
        
        plt.rcParams.update(default_params)
        
    except ImportError:
        warnings.warn("Matplotlib not available, cannot set style")


def memory_usage_monitor():
    """
    Monitor memory usage (requires psutil).
    
    Returns
    -------
    Dict[str, float] or None
        Memory usage information or None if psutil not available
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }
    except ImportError:
        return None


def create_example_data(n_points: int = 100, n_files: int = 10) -> List[Dict[str, Any]]:
    """
    Create example transfer curve data for testing.
    
    Parameters
    ----------
    n_points : int, default 100
        Number of data points per curve
    n_files : int, default 10
        Number of files to simulate
        
    Returns
    -------
    List[Dict[str, Any]]
        List of simulated transfer objects
    """
    try:
        import numpy as np
        from oect_transfer import Transfer
    except ImportError:
        raise ImportError("numpy and oect-transfer required for example data generation")
    
    transfer_objects = []
    
    for i in range(n_files):
        # Create synthetic data with some variation
        vg = np.linspace(-0.6, 0.6, n_points)
        
        # Simulate device degradation over time
        degradation_factor = 1 - (i * 0.05)  # 5% degradation per "measurement"
        noise_level = 0.1 + (i * 0.01)  # Increasing noise
        
        # Exponential transfer curve with noise
        id_base = np.exp(vg * 8) * 1e-6 * degradation_factor
        noise = np.random.normal(0, noise_level * np.abs(id_base))
        id_data = id_base + noise
        
        # Ensure positive current
        id_data = np.abs(id_data)
        
        # Create Transfer object
        transfer = Transfer(vg, id_data, device_type="N")
        
        transfer_objects.append({
            'filename': f'transfer_{i+1:03d}.csv',
            'transfer': transfer,
            'data_points': len(vg),
            'file_path': f'/example/data/transfer_{i+1:03d}.csv',
            'vg_column': 'vg',
            'id_column': 'id',
            'device_type': 'N'
        })
    
    return transfer_objects


def estimate_animation_time(
    n_frames: int,
    dpi: int = 100,
    n_workers: Optional[int] = None
) -> Dict[str, float]:
    """
    Estimate animation generation time.
    
    Parameters
    ----------
    n_frames : int
        Number of frames in animation
    dpi : int, default 100
        Figure DPI
    n_workers : int, optional
        Number of parallel workers
        
    Returns
    -------
    Dict[str, float]
        Time estimates in seconds
    """
    if n_workers is None:
        import multiprocessing as mp
        n_workers = mp.cpu_count()
    
    # Empirical estimates (seconds per frame)
    base_time_per_frame = 0.5  # Base time for simple frame
    dpi_factor = (dpi / 100) ** 1.5  # DPI impact
    
    # Sequential time
    sequential_time = n_frames * base_time_per_frame * dpi_factor
    
    # Parallel time (with overhead)
    parallel_efficiency = 0.7  # Account for overhead
    parallel_time = (sequential_time / n_workers) * (1 / parallel_efficiency)
    
    # Video encoding time (approximately 10% of frame generation)
    encoding_time = parallel_time * 0.1
    
    total_time = parallel_time + encoding_time
    
    return {
        'frames': n_frames,
        'sequential_estimate': sequential_time,
        'parallel_estimate': parallel_time,
        'encoding_estimate': encoding_time,
        'total_estimate': total_time,
        'workers': n_workers
    }


def safe_import(module_name: str, package_name: Optional[str] = None):
    """
    Safely import a module with informative error message.
    
    Parameters
    ----------
    module_name : str
        Name of module to import
    package_name : str, optional
        Name of package for installation instructions
        
    Returns
    -------
    module
        Imported module
        
    Raises
    ------
    ImportError
        If module cannot be imported with installation instructions
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pkg_name = package_name or module_name
        raise ImportError(
            f"Module '{module_name}' is required but not installed. "
            f"Install with: pip install {pkg_name}"
        )


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Parameters
        ----------
        total : int
            Total number of items to process
        description : str, default "Processing"
            Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = None
    
    def start(self):
        """Start the progress tracker."""
        import time
        self.start_time = time.time()
        print(f"{self.description}: 0/{self.total} (0.0%)")
    
    def update(self, increment: int = 1):
        """Update progress."""
        import time
        
        self.current += increment
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            percentage = (self.current / self.total) * 100
            
            if self.current > 0:
                estimated_total = elapsed * (self.total / self.current)
                remaining = estimated_total - elapsed
                print(f"{self.description}: {self.current}/{self.total} "
                      f"({percentage:.1f}%) - ETA: {remaining:.1f}s")
            else:
                print(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """Finish progress tracking."""
        import time
        
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"{self.description}: Complete in {elapsed:.2f}s")
        else:
            print(f"{self.description}: Complete")