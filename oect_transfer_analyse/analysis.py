# analysis.py
"""
Advanced time-series analysis for OECT transfer curves

This module provides enhanced analysis capabilities built on top of the core
oect-transfer package, including trend detection, stability analysis, and
automated reporting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

# Import from core package
try:
    import oect_transfer as ot
except ImportError:
    raise ImportError("oect-transfer is required. Install with: pip install oect-transfer")

# Handle optional matplotlib dependency  
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass 
class TimeSeriesData:
    """
    Enhanced time-series data container with additional metadata.
    
    This extends the basic time-series functionality with additional
    analysis capabilities and metadata tracking.
    """
    filenames: List[str]
    time_points: np.ndarray
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
    absgm_max_raw: np.ndarray
    absgm_max_forward: np.ndarray
    absgm_max_reverse: np.ndarray
    absI_max_raw: np.ndarray
    absI_max_forward: np.ndarray
    absI_max_reverse: np.ndarray
    absI_min_raw: np.ndarray
    absI_min_forward: np.ndarray
    absI_min_reverse: np.ndarray
    
    # Additional metadata
    device_type: str = "N"
    measurement_conditions: Optional[Dict[str, Any]] = None
    analysis_timestamp: Optional[str] = None


class TransferTimeSeriesExtractor:
    """
    Enhanced time-series extractor with advanced analysis capabilities.
    
    This builds upon the core transfer analysis to provide comprehensive
    time-series extraction and trend analysis for device stability studies.
    """
    
    def __init__(self, transfer_objects: List[Dict[str, Any]], device_type: str = "N"):
        if not transfer_objects:
            raise ValueError("transfer_objects list is empty")
        
        self.transfer_objects = transfer_objects
        self.device_type = device_type
        self.time_series_data: Optional[TimeSeriesData] = None
        self._analysis_cache = {}
        
    def extract_time_series(self, 
                           time_values: Optional[np.ndarray] = None,
                           time_unit: str = "measurement_index") -> TimeSeriesData:
        """
        Extract comprehensive time-series data with enhanced metadata.
        
        Parameters
        ----------
        time_values : np.ndarray, optional
            Actual time values. If None, uses sequential indices
        time_unit : str, default "measurement_index"
            Unit for time values (e.g., "hours", "minutes", "days")
            
        Returns
        -------
        TimeSeriesData
            Enhanced time-series data container
        """
        n_points = len(self.transfer_objects)
        
        # Initialize time points
        if time_values is not None:
            if len(time_values) != n_points:
                raise ValueError(
                    f"Length of time_values ({len(time_values)}) must match "
                    f"number of transfer objects ({n_points})"
                )
            time_points = np.array(time_values)
        else:
            time_points = np.arange(n_points)
        
        # Initialize arrays for all parameters
        arrays = {}
        param_names = [
            'gm_max_raw', 'gm_max_forward', 'gm_max_reverse',
            'I_max_raw', 'I_max_forward', 'I_max_reverse',
            'I_min_raw', 'I_min_forward', 'I_min_reverse',
            'Von_raw', 'Von_forward', 'Von_reverse',
            'absgm_max_raw', 'absgm_max_forward', 'absgm_max_reverse',
            'absI_max_raw', 'absI_max_forward', 'absI_max_reverse',
            'absI_min_raw', 'absI_min_forward', 'absI_min_reverse'
        ]
        
        for param in param_names:
            arrays[param] = np.zeros(n_points)
        
        filenames = []
        
        # Extract data from each transfer object
        for i, item in enumerate(self.transfer_objects):
            try:
                transfer = item['transfer']
                filename = item.get('filename', f'measurement_{i}')
                filenames.append(filename)
                
                # Extract all parameters safely
                arrays['gm_max_raw'][i] = getattr(transfer.gm_max, 'raw', np.nan)
                arrays['gm_max_forward'][i] = getattr(transfer.gm_max, 'forward', np.nan)
                arrays['gm_max_reverse'][i] = getattr(transfer.gm_max, 'reverse', np.nan)
                
                arrays['I_max_raw'][i] = getattr(transfer.I_max, 'raw', np.nan)
                arrays['I_max_forward'][i] = getattr(transfer.I_max, 'forward', np.nan)
                arrays['I_max_reverse'][i] = getattr(transfer.I_max, 'reverse', np.nan)
                
                arrays['I_min_raw'][i] = getattr(transfer.I_min, 'raw', np.nan)
                arrays['I_min_forward'][i] = getattr(transfer.I_min, 'forward', np.nan)
                arrays['I_min_reverse'][i] = getattr(transfer.I_min, 'reverse', np.nan)
                
                arrays['Von_raw'][i] = getattr(transfer.Von, 'raw', np.nan)
                arrays['Von_forward'][i] = getattr(transfer.Von, 'forward', np.nan)
                arrays['Von_reverse'][i] = getattr(transfer.Von, 'reverse', np.nan)
                
                arrays['absgm_max_raw'][i] = getattr(transfer.absgm_max, 'raw', np.nan)
                arrays['absgm_max_forward'][i] = getattr(transfer.absgm_max, 'forward', np.nan)
                arrays['absgm_max_reverse'][i] = getattr(transfer.absgm_max, 'reverse', np.nan)
                
                arrays['absI_max_raw'][i] = getattr(transfer.absI_max, 'raw', np.nan)
                arrays['absI_max_forward'][i] = getattr(transfer.absI_max, 'forward', np.nan)
                arrays['absI_max_reverse'][i] = getattr(transfer.absI_max, 'reverse', np.nan)
                
                arrays['absI_min_raw'][i] = getattr(transfer.absI_min, 'raw', np.nan)
                arrays['absI_min_forward'][i] = getattr(transfer.absI_min, 'forward', np.nan)
                arrays['absI_min_reverse'][i] = getattr(transfer.absI_min, 'reverse', np.nan)
                
            except Exception as e:
                warnings.warn(f"Error extracting data from item {i}: {e}")
                # Fill with NaN for failed extractions
                for param in param_names:
                    arrays[param][i] = np.nan
                filenames.append(f'error_{i}')
        
        # Create enhanced TimeSeriesData object
        import datetime
        self.time_series_data = TimeSeriesData(
            filenames=filenames,
            time_points=time_points,
            device_type=self.device_type,
            analysis_timestamp=datetime.datetime.now().isoformat(),
            **arrays
        )
        
        return self.time_series_data
    
    def detect_trends(self, parameters: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Detect trends in time-series parameters using linear regression.
        
        Parameters
        ----------
        parameters : List[str], optional
            Parameters to analyze. If None, analyzes key parameters
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Trend analysis results for each parameter
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        if parameters is None:
            parameters = ['gm_max_raw', 'Von_raw', 'I_max_raw', 'absgm_max_raw']
        
        trends = {}
        
        for param in parameters:
            param_data = getattr(self.time_series_data, param, None)
            if param_data is None:
                continue
                
            # Remove NaN values
            valid_mask = ~np.isnan(param_data)
            if np.sum(valid_mask) < 3:  # Need at least 3 points for trend
                trends[param] = {'error': 'Insufficient valid data points'}
                continue
            
            x = self.time_series_data.time_points[valid_mask]
            y = param_data[valid_mask]
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1]**2
            
            # Calculate trend significance
            relative_slope = slope / np.mean(y) if np.mean(y) != 0 else 0
            
            trends[param] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'relative_slope_percent': relative_slope * 100,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': 'strong' if abs(r_squared) > 0.7 else 'moderate' if abs(r_squared) > 0.3 else 'weak',
                'significant': abs(relative_slope) > 0.01 and abs(r_squared) > 0.3  # 1% change with moderate correlation
            }
        
        # Cache results
        self._analysis_cache['trends'] = trends
        return trends
    
    def detect_drift_advanced(self, 
                            parameter: str = 'gm_max_raw',
                            window_size: int = 5,
                            threshold: float = 0.05) -> Dict[str, Any]:
        """
        Advanced drift detection using moving window analysis.
        
        Parameters
        ----------
        parameter : str, default 'gm_max_raw'
            Parameter to analyze
        window_size : int, default 5
            Size of moving window for analysis
        threshold : float, default 0.05
            Threshold for drift detection (5%)
            
        Returns
        -------
        Dict[str, Any]
            Advanced drift analysis results
        """
        if self.time_series_data is None:
            self.extract_time_series()
        
        param_data = getattr(self.time_series_data, parameter, None)
        if param_data is None:
            return {'error': f'Parameter {parameter} not found'}
        
        # Remove NaN values
        valid_mask = ~np.isnan(param_data)
        if np.sum(valid_mask) < window_size:
            return {'error': 'Insufficient valid data points'}
        
        x = self.time_series_data.time_points[valid_mask]
        y = param_data[valid_mask]
        
        # Moving window analysis
        window_changes = []
        window_positions = []
        
        for i in range(len(y) - window_size + 1):
            window = y[i:i+window_size]
            initial = window[0]
            final = window[-1]
            
            if initial != 0:
                change = (final - initial) / initial
                window_changes.append(change)
                window_positions.append(x[i + window_size//2])
        
        if not window_changes:
            return {'error': 'Could not calculate window changes'}
        
        window_changes = np.array(window_changes)
        
        # Detect drift events
        drift_events = []
        for i, change in enumerate(window_changes):
            if abs(change) > threshold:
                drift_events.append({
                    'position': window_positions[i],
                    'magnitude': change,
                    'direction': 'increase' if change > 0 else 'decrease'
                })
        
        return {
            'parameter': parameter,
            'window_size': window_size,
            'threshold': threshold,
            'total_drift': (y[-1] - y[0]) / y[0] if y[0] != 0 else 0,
            'max_window_drift': np.max(np.abs(window_changes)),
            'drift_events': drift_events,
            'drift_detected': len(drift_events) > 0,
            'stability_score': 1 - np.std(window_changes)  # Higher is more stable
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert time-series data to pandas DataFrame with enhanced metadata."""
        if self.time_series_data is None:
            self.extract_time_series()
        
        data_dict = {
            'filename': self.time_series_data.filenames,
            'time_point': self.time_series_data.time_points,
            'device_type': [self.time_series_data.device_type] * len(self.time_series_data.filenames)
        }
        
        # Add all parameter data
        param_names = [
            'gm_max_raw', 'gm_max_forward', 'gm_max_reverse',
            'I_max_raw', 'I_max_forward', 'I_max_reverse',
            'I_min_raw', 'I_min_forward', 'I_min_reverse',
            'Von_raw', 'Von_forward', 'Von_reverse',
            'absgm_max_raw', 'absgm_max_forward', 'absgm_max_reverse',
            'absI_max_raw', 'absI_max_forward', 'absI_max_reverse',
            'absI_min_raw', 'absI_min_forward', 'absI_min_reverse'
        ]
        
        for param in param_names:
            data_dict[param] = getattr(self.time_series_data, param)
        
        df = pd.DataFrame(data_dict)
        
        # Add analysis metadata as attributes
        if hasattr(self.time_series_data, 'analysis_timestamp'):
            df.attrs['analysis_timestamp'] = self.time_series_data.analysis_timestamp
        df.attrs['device_type'] = self.time_series_data.device_type
        df.attrs['package_version'] = "1.0.0"
        
        return df


def analyze_transfer_stability(transfer_objects: List[Dict[str, Any]], 
                             device_type: str = "N",
                             time_values: Optional[np.ndarray] = None,
                             drift_threshold: float = 0.05,
                             verbose: bool = True) -> TransferTimeSeriesExtractor:
    """
    Comprehensive transfer curve stability analysis workflow.
    
    This function provides enhanced stability analysis compared to the core package,
    including trend detection and advanced drift analysis.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects from oect_transfer.load_all_transfer_files()
    device_type : str, default "N"
        Device type ('N' or 'P')
    time_values : np.ndarray, optional
        Actual time values. If None, uses sequential indices
    drift_threshold : float, default 0.05
        Threshold for drift detection (5%)
    verbose : bool, default True
        Whether to print analysis results
        
    Returns
    -------
    TransferTimeSeriesExtractor
        Configured analyzer with extracted data and analysis results
    """
    # Create enhanced extractor
    extractor = TransferTimeSeriesExtractor(transfer_objects, device_type)
    
    # Extract time-series data
    time_series = extractor.extract_time_series(time_values)
    
    if verbose:
        print(f"📊 OECT Transfer Stability Analysis")
        print(f"{'='*50}")
        print(f"Device type: {device_type}")
        print(f"Measurements: {len(time_series.filenames)}")
        print(f"Time span: {len(time_series.time_points)} points")
        if time_values is not None:
            print(f"Time range: {time_values[0]:.1f} to {time_values[-1]:.1f}")
    
    # Enhanced trend detection
    if verbose:
        print(f"\n🔍 Trend Analysis")
        print(f"{'-'*30}")
    
    trends = extractor.detect_trends()
    for param, trend_data in trends.items():
        if 'error' not in trend_data:
            if verbose:
                status = "📈" if trend_data['trend_direction'] == 'increasing' else "📉"
                strength = trend_data['trend_strength']
                print(f"{status} {param}: {trend_data['relative_slope_percent']:.2f}%/point ({strength})")
    
    # Advanced drift detection
    if verbose:
        print(f"\n⚠️  Drift Detection (threshold: {drift_threshold*100:.1f}%)")
        print(f"{'-'*40}")
    
    key_params = ['gm_max_raw', 'Von_raw', 'I_max_raw']
    for param in key_params:
        drift_result = extractor.detect_drift_advanced(param, threshold=drift_threshold)
        if 'error' not in drift_result:
            if verbose:
                status = "🚨" if drift_result['drift_detected'] else "✅"
                total_drift = drift_result['total_drift'] * 100
                stability = drift_result['stability_score']
                print(f"{status} {param}: {total_drift:+.1f}% total (stability: {stability:.2f})")
    
    return extractor


def detect_parameter_trends(extractor: TransferTimeSeriesExtractor,
                          parameters: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detect and summarize parameter trends.
    
    Parameters
    ----------
    extractor : TransferTimeSeriesExtractor
        Configured extractor with time-series data
    parameters : List[str], optional
        Parameters to analyze
        
    Returns
    -------
    pd.DataFrame
        Summary of trend analysis results
    """
    trends = extractor.detect_trends(parameters)
    
    trend_summary = []
    for param, trend_data in trends.items():
        if 'error' not in trend_data:
            trend_summary.append({
                'parameter': param,
                'slope': trend_data['slope'],
                'r_squared': trend_data['r_squared'],
                'relative_slope_percent': trend_data['relative_slope_percent'],
                'trend_direction': trend_data['trend_direction'],
                'trend_strength': trend_data['trend_strength'],
                'significant': trend_data['significant']
            })
    
    return pd.DataFrame(trend_summary)


def generate_stability_report(extractor: TransferTimeSeriesExtractor,
                            output_path: str = "stability_report.html",
                            include_plots: bool = True) -> str:
    """
    Generate comprehensive HTML stability report.
    
    Parameters
    ----------
    extractor : TransferTimeSeriesExtractor
        Configured extractor with analysis results
    output_path : str, default "stability_report.html"
        Output file path for the report
    include_plots : bool, default True
        Whether to include plots in the report
        
    Returns
    -------
    str
        Path to generated report
    """
    if extractor.time_series_data is None:
        extractor.extract_time_series()
    
    # Generate report content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OECT Transfer Stability Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .alert {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 OECT Transfer Stability Report</h1>
        <p>Generated: {extractor.time_series_data.analysis_timestamp}</p>
        <p>Device Type: {extractor.time_series_data.device_type}</p>
        <p>Measurements: {len(extractor.time_series_data.filenames)}</p>
    </div>
"""
    
    # Add summary statistics
    df = extractor.to_dataframe()
    html_content += """
    <div class="section">
        <h2>📈 Summary Statistics</h2>
        <table>
            <tr><th>Parameter</th><th>Mean</th><th>Std Dev</th><th>CV (%)</th><th>Range</th></tr>
"""
    
    key_params = ['gm_max_raw', 'Von_raw', 'I_max_raw']
    for param in key_params:
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0
                range_val = values.max() - values.min()
                
                html_content += f"""
            <tr>
                <td>{param}</td>
                <td>{mean_val:.2e}</td>
                <td>{std_val:.2e}</td>
                <td>{cv_val:.1f}</td>
                <td>{range_val:.2e}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
"""
    
    # Add trend analysis
    trends = extractor.detect_trends()
    html_content += """
    <div class="section">
        <h2>🔍 Trend Analysis</h2>
"""
    
    for param, trend_data in trends.items():
        if 'error' not in trend_data:
            alert_class = "alert" if trend_data['significant'] else "success"
            direction_icon = "📈" if trend_data['trend_direction'] == 'increasing' else "📉"
            
            html_content += f"""
        <div class="{alert_class}">
            <strong>{direction_icon} {param}</strong><br>
            Trend: {trend_data['relative_slope_percent']:.2f}%/point ({trend_data['trend_strength']})<br>
            R²: {trend_data['r_squared']:.3f} | Significant: {trend_data['significant']}
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📋 Stability report generated: {output_path}")
    return output_path