"""Time series analysis for OECT transfer curve parameters."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple

try:
    from oect_transfer import Transfer
except ImportError:
    raise ImportError(
        "oect-transfer package is required. Install with: pip install oect-transfer"
    )


@dataclass
class TimeSeriesData:
    """Container for time series data extracted from Transfer objects."""
    
    filenames: List[str]
    time_points: np.ndarray
    
    # Core parameters
    gm_max_raw: np.ndarray
    gm_max_forward: np.ndarray
    gm_max_reverse: np.ndarray
    
    I_max_raw: np.ndarray
    I_max_forward: np.ndarray
    I_max_reverse: np.ndarray
    
    I_min_raw: np.ndarray
    I_min_forward: np.ndarray
    I_min_reverse: np.ndarray
    
    Von_raw: np.ndarray
    Von_forward: np.ndarray
    Von_reverse: np.ndarray
    
    # Absolute value parameters
    absgm_max_raw: np.ndarray
    absgm_max_forward: np.ndarray
    absgm_max_reverse: np.ndarray
    
    absI_max_raw: np.ndarray
    absI_max_forward: np.ndarray
    absI_max_reverse: np.ndarray
    
    absI_min_raw: np.ndarray
    absI_min_forward: np.ndarray
    absI_min_reverse: np.ndarray


class TimeSeriesAnalyzer:
    """Analyzer for extracting and analyzing time series data from Transfer objects."""
    
    def __init__(self, transfer_objects: List[Dict[str, Any]]):
        """
        Initialize the time series analyzer.
        
        Parameters
        ----------
        transfer_objects : List[Dict[str, Any]]
            List of transfer objects from DataLoader.load_all_files()
        """
        self.transfer_objects = transfer_objects
        self.time_series_data: Optional[TimeSeriesData] = None
        self._validate_input()
    
    def _validate_input(self) -> None:
        """Validate input transfer objects."""
        if not self.transfer_objects:
            raise ValueError("transfer_objects list is empty")
        
        if not all(isinstance(obj, dict) for obj in self.transfer_objects):
            raise ValueError("All items in transfer_objects must be dictionaries")
        
        required_keys = ['transfer', 'filename']
        for i, obj in enumerate(self.transfer_objects):
            for key in required_keys:
                if key not in obj:
                    raise ValueError(f"Missing key '{key}' in transfer_objects[{i}]")
    
    def extract_time_series(
        self, 
        time_points: Optional[np.ndarray] = None
    ) -> TimeSeriesData:
        """
        Extract time series data from all transfer objects.
        
        Parameters
        ----------
        time_points : np.ndarray, optional
            Custom time points. If None, uses sequential indices
            
        Returns
        -------
        TimeSeriesData
            Container with all extracted time series data
        """
        n_points = len(self.transfer_objects)
        
        # Set up time points
        if time_points is None:
            time_points = np.arange(n_points)
        elif len(time_points) != n_points:
            raise ValueError(
                f"time_points length ({len(time_points)}) must match "
                f"number of transfer objects ({n_points})"
            )
        
        # Initialize arrays
        filenames = []
        
        # Core parameters
        gm_max_raw = np.zeros(n_points)
        gm_max_forward = np.zeros(n_points)
        gm_max_reverse = np.zeros(n_points)
        
        I_max_raw = np.zeros(n_points)
        I_max_forward = np.zeros(n_points)
        I_max_reverse = np.zeros(n_points)
        
        I_min_raw = np.zeros(n_points)
        I_min_forward = np.zeros(n_points)
        I_min_reverse = np.zeros(n_points)
        
        Von_raw = np.zeros(n_points)
        Von_forward = np.zeros(n_points)
        Von_reverse = np.zeros(n_points)
        
        # Absolute value parameters
        absgm_max_raw = np.zeros(n_points)
        absgm_max_forward = np.zeros(n_points)
        absgm_max_reverse = np.zeros(n_points)
        
        absI_max_raw = np.zeros(n_points)
        absI_max_forward = np.zeros(n_points)
        absI_max_reverse = np.zeros(n_points)
        
        absI_min_raw = np.zeros(n_points)
        absI_min_forward = np.zeros(n_points)
        absI_min_reverse = np.zeros(n_points)
        
        # Extract data from each transfer object
        for i, item in enumerate(self.transfer_objects):
            transfer = item['transfer']
            filename = item['filename']
            
            filenames.append(filename)
            
            # Extract gm_max
            gm_max_raw[i] = transfer.gm_max.raw
            gm_max_forward[i] = transfer.gm_max.forward
            gm_max_reverse[i] = transfer.gm_max.reverse
            
            # Extract I_max
            I_max_raw[i] = transfer.I_max.raw
            I_max_forward[i] = transfer.I_max.forward
            I_max_reverse[i] = transfer.I_max.reverse
            
            # Extract I_min
            I_min_raw[i] = transfer.I_min.raw
            I_min_forward[i] = transfer.I_min.forward
            I_min_reverse[i] = transfer.I_min.reverse
            
            # Extract Von
            Von_raw[i] = transfer.Von.raw
            Von_forward[i] = (
                transfer.Von.forward if transfer.Von.forward is not None 
                else np.nan
            )
            Von_reverse[i] = (
                transfer.Von.reverse if transfer.Von.reverse is not None 
                else np.nan
            )
            
            # Extract absolute value parameters
            absgm_max_raw[i] = transfer.absgm_max.raw
            absgm_max_forward[i] = transfer.absgm_max.forward
            absgm_max_reverse[i] = transfer.absgm_max.reverse
            
            absI_max_raw[i] = transfer.absI_max.raw
            absI_max_forward[i] = transfer.absI_max.forward
            absI_max_reverse[i] = transfer.absI_max.reverse
            
            absI_min_raw[i] = transfer.absI_min.raw
            absI_min_forward[i] = transfer.absI_min.forward
            absI_min_reverse[i] = transfer.absI_min.reverse
        
        # Create TimeSeriesData object
        self.time_series_data = TimeSeriesData(
            filenames=filenames,
            time_points=time_points,
            gm_max_raw=gm_max_raw,
            gm_max_forward=gm_max_forward,
            gm_max_reverse=gm_max_reverse,
            I_max_raw=I_max_raw,
            I_max_forward=I_max_forward,
            I_max_reverse=I_max_reverse,
            I_min_raw=I_min_raw,
            I_min_forward=I_min_forward,
            I_min_reverse=I_min_reverse,
            Von_raw=Von_raw,
            Von_forward=Von_forward,
            Von_reverse=Von_reverse,
            absgm_max_raw=absgm_max_raw,
            absgm_max_forward=absgm_max_forward,
            absgm_max_reverse=absgm_max_reverse,
            absI_max_raw=absI_max_raw,
            absI_max_forward=absI_max_forward,
            absI_max_reverse=absI_max_reverse,
            absI_min_raw=absI_min_raw,
            absI_min_forward=absI_min_forward,
            absI_min_reverse=absI_min_reverse
        )
        
        return self.time_series_data
    
    def get_parameter_dict(self) -> Dict[str, np.ndarray]:
        """
        Get parameter dictionary format of time series data.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping parameter names to arrays
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        return {
            'time_points': self.time_series_data.time_points,
            'gm_max_raw': self.time_series_data.gm_max_raw,
            'gm_max_forward': self.time_series_data.gm_max_forward,
            'gm_max_reverse': self.time_series_data.gm_max_reverse,
            'I_max_raw': self.time_series_data.I_max_raw,
            'I_max_forward': self.time_series_data.I_max_forward,
            'I_max_reverse': self.time_series_data.I_max_reverse,
            'I_min_raw': self.time_series_data.I_min_raw,
            'I_min_forward': self.time_series_data.I_min_forward,
            'I_min_reverse': self.time_series_data.I_min_reverse,
            'Von_raw': self.time_series_data.Von_raw,
            'Von_forward': self.time_series_data.Von_forward,
            'Von_reverse': self.time_series_data.Von_reverse,
            'absgm_max_raw': self.time_series_data.absgm_max_raw,
            'absgm_max_forward': self.time_series_data.absgm_max_forward,
            'absgm_max_reverse': self.time_series_data.absgm_max_reverse,
            'absI_max_raw': self.time_series_data.absI_max_raw,
            'absI_max_forward': self.time_series_data.absI_max_forward,
            'absI_max_reverse': self.time_series_data.absI_max_reverse,
            'absI_min_raw': self.time_series_data.absI_min_raw,
            'absI_min_forward': self.time_series_data.absI_min_forward,
            'absI_min_reverse': self.time_series_data.absI_min_reverse
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert time series data to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all time series data
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        data_dict = {
            'filename': self.time_series_data.filenames,
            'time_point': self.time_series_data.time_points,
        }
        
        # Add all parameter data
        param_dict = self.get_parameter_dict()
        for key, values in param_dict.items():
            if key != 'time_points':  # Avoid duplicate
                data_dict[key] = values
        
        return pd.DataFrame(data_dict)
    
    def plot_time_series(
        self,
        parameters: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        ylim: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 150
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot time series of selected parameters.
        
        Parameters
        ----------
        parameters : List[str], optional
            Parameter names to plot. If None, plots key parameters
        save_path : str, optional
            Path to save figure. If None, shows plot
        ylim : tuple or dict, optional
            Y-axis limits. Can be (ymin, ymax) for all plots or 
            dict mapping parameter names to limits
        figsize : tuple, default (12, 8)
            Figure size in inches
        dpi : int, default 150
            Figure DPI
            
        Returns
        -------
        Tuple[plt.Figure, List[plt.Axes]]
            Figure and axes objects
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        if parameters is None:
            parameters = [
                'gm_max_raw', 'I_max_raw', 'I_min_raw', 'Von_raw',
                'absgm_max_raw', 'absI_max_raw', 'absI_min_raw'
            ]
        
        n_params = len(parameters)
        fig, axes = plt.subplots(n_params, 1, figsize=figsize, dpi=dpi)
        
        if n_params == 1:
            axes = [axes]
        
        param_dict = self.get_parameter_dict()
        
        for i, param in enumerate(parameters):
            if param in param_dict:
                axes[i].plot(
                    self.time_series_data.time_points, 
                    param_dict[param], 
                    'o-', linewidth=1.5, markersize=4
                )
                axes[i].set_ylabel(param, fontsize=12)
                axes[i].set_xlabel('Time Point', fontsize=12)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_title(f'{param} vs Time', fontsize=14)
                
                # Set y-axis limits
                if ylim is not None:
                    if isinstance(ylim, dict) and param in ylim:
                        axes[i].set_ylim(ylim[param])
                    elif isinstance(ylim, (tuple, list)) and len(ylim) == 2:
                        axes[i].set_ylim(ylim)
            else:
                print(f"Warning: Parameter '{param}' not found in data")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Time series plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig, axes
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get statistical summary of all parameters.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics table
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        param_dict = self.get_parameter_dict()
        
        stats = {}
        for param_name, values in param_dict.items():
            if param_name != 'time_points':
                # Filter NaN values
                clean_values = values[~np.isnan(values)]
                if len(clean_values) > 0:
                    stats[param_name] = {
                        'count': len(clean_values),
                        'mean': np.mean(clean_values),
                        'std': np.std(clean_values),
                        'min': np.min(clean_values),
                        'max': np.max(clean_values),
                        'range': np.max(clean_values) - np.min(clean_values),
                        'cv_percent': np.std(clean_values) / np.mean(clean_values) * 100
                    }
                else:
                    stats[param_name] = {
                        'count': 0, 'mean': np.nan, 'std': np.nan,
                        'min': np.nan, 'max': np.nan, 'range': np.nan, 
                        'cv_percent': np.nan
                    }
        
        return pd.DataFrame(stats).T
    
    def detect_drift(
        self, 
        parameter: str = 'gm_max_raw', 
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect parameter drift over time.
        
        Parameters
        ----------
        parameter : str, default 'gm_max_raw'
            Parameter name to analyze
        threshold : float, default 0.05
            Drift threshold (relative change percentage)
            
        Returns
        -------
        Dict[str, Any]
            Drift analysis results
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        param_dict = self.get_parameter_dict()
        
        if parameter not in param_dict:
            raise ValueError(f"Parameter '{parameter}' not found in data")
        
        values = param_dict[parameter]
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 2:
            return {"error": "Insufficient data points for drift analysis"}
        
        # Calculate relative change from initial value
        initial_value = clean_values[0]
        if initial_value == 0:
            return {"error": "Initial value is zero, cannot calculate relative change"}
        
        relative_change = (clean_values - initial_value) / initial_value * 100
        
        # Detect drift
        drift_detected = np.any(np.abs(relative_change) > threshold * 100)
        max_drift = np.max(np.abs(relative_change))
        final_drift = relative_change[-1]
        drift_direction = "increase" if final_drift > 0 else "decrease"
        
        return {
            "parameter": parameter,
            "drift_detected": drift_detected,
            "max_drift_percent": max_drift,
            "final_drift_percent": final_drift,
            "drift_direction": drift_direction,
            "threshold_percent": threshold * 100,
            "initial_value": initial_value,
            "final_value": clean_values[-1],
            "n_points": len(clean_values)
        }
    
    def analyze_stability(
        self, 
        parameters: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Analyze stability of multiple parameters.
        
        Parameters
        ----------
        parameters : List[str], optional
            Parameters to analyze. If None, uses key parameters
        threshold : float, default 0.05
            Drift threshold
            
        Returns
        -------
        pd.DataFrame
            Stability analysis results
        """
        if parameters is None:
            parameters = [
                'gm_max_raw', 'I_max_raw', 'Von_raw', 
                'absgm_max_raw', 'absI_max_raw', 'absI_min_raw'
            ]
        
        results = []
        for param in parameters:
            try:
                drift_result = self.detect_drift(param, threshold)
                if 'error' not in drift_result:
                    results.append({
                        'parameter': param,
                        'drift_detected': drift_result['drift_detected'],
                        'max_drift_percent': drift_result['max_drift_percent'],
                        'final_drift_percent': drift_result['final_drift_percent'],
                        'drift_direction': drift_result['drift_direction'],
                        'initial_value': drift_result['initial_value'],
                        'final_value': drift_result['final_value']
                    })
                else:
                    results.append({
                        'parameter': param,
                        'drift_detected': None,
                        'max_drift_percent': np.nan,
                        'final_drift_percent': np.nan,
                        'drift_direction': 'error',
                        'initial_value': np.nan,
                        'final_value': np.nan
                    })
            except Exception as e:
                print(f"Warning: Could not analyze {param}: {e}")
                continue
        
        return pd.DataFrame(results)


def analyze_transfer_stability(
    transfer_objects: List[Dict[str, Any]],
    verbose: bool = True
) -> TimeSeriesAnalyzer:
    """
    Complete transfer stability analysis workflow.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        Transfer objects from DataLoader
    verbose : bool, default True
        Whether to print analysis results
        
    Returns
    -------
    TimeSeriesAnalyzer
        Configured analyzer object
    """
    # Create analyzer
    analyzer = TimeSeriesAnalyzer(transfer_objects)
    
    # Extract time series data
    time_series = analyzer.extract_time_series()
    
    if verbose:
        print(f"Successfully extracted data from {len(time_series.filenames)} files")
        
        # Show data overview
        df = analyzer.to_dataframe()
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Statistical summary
        print("\n=== Statistical Summary ===")
        stats = analyzer.get_summary_statistics()
        print(stats.round(6))
        
        # Drift detection
        print("\n=== Drift Detection ===")
        stability = analyzer.analyze_stability(threshold=0.05)  # 5% threshold
        for _, row in stability.iterrows():
            param = row['parameter']
            if row['drift_detected'] is not None:
                status = 'Drift detected' if row['drift_detected'] else 'Stable'
                print(f"{param}: {status}")
                print(f"  Max drift: {row['max_drift_percent']:.2f}%")
                print(f"  Final drift: {row['final_drift_percent']:.2f}%")
    
    return analyzer