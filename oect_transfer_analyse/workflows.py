# workflows.py
"""
Pre-defined analysis workflows for OECT transfer curves

This module provides high-level workflow functions that combine multiple
analysis and visualization steps for common use cases.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import warnings

# Import from core package
try:
    import oect_transfer as ot
except ImportError:
    raise ImportError("oect-transfer is required. Install with: pip install oect-transfer")

# Import from this package
from .analysis import (
    TransferTimeSeriesExtractor, 
    analyze_transfer_stability,
    detect_parameter_trends,
    generate_stability_report
)

# Import optional modules
try:
    from .plotting import (
        plot_transfer_evolution,
        plot_transfer_comparison, 
        plot_parameter_trends,
        create_publication_plots
    )
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

try:
    from .animation import (
        generate_transfer_animation,
        create_animation_preview,
        batch_animation_generation
    )
    _ANIMATION_AVAILABLE = True
except ImportError:
    _ANIMATION_AVAILABLE = False


def complete_analysis_workflow(
    data_folder: str,
    device_type: str = "N",
    output_dir: str = "analysis_results",
    device_label: str = "Device",
    time_values: Optional[List[float]] = None,
    drift_threshold: float = 0.05,
    generate_report: bool = True,
    create_plots: bool = True,
    create_animations: bool = True,
    animation_style: str = 'standard',
    plot_style: str = 'publication',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete end-to-end analysis workflow for OECT transfer data.
    
    This function performs a comprehensive analysis including:
    - Data loading and validation
    - Time-series extraction and analysis
    - Drift detection and trend analysis
    - Visualization generation
    - Animation creation
    - Report generation
    
    Parameters
    ----------
    data_folder : str
        Path to folder containing transfer CSV files
    device_type : str, default "N"
        Device type ('N' or 'P')
    output_dir : str, default "analysis_results"
        Output directory for all results
    device_label : str, default "Device"
        Label for the device (used in plots and reports)
    time_values : List[float], optional
        Actual time values for measurements
    drift_threshold : float, default 0.05
        Threshold for drift detection (5%)
    generate_report : bool, default True
        Whether to generate HTML stability report
    create_plots : bool, default True
        Whether to create plots (requires matplotlib)
    create_animations : bool, default True
        Whether to create animations (requires matplotlib + opencv)
    animation_style : str, default 'standard'
        Animation style: 'standard', 'publication', 'minimal'
    plot_style : str, default 'publication'
        Plot style for figures
    verbose : bool, default True
        Whether to print progress information
        
    Returns
    -------
    Dict[str, Any]
        Results summary including paths to generated files and analysis results
        
    Examples
    --------
    >>> # Complete analysis with all features
    >>> results = complete_analysis_workflow(
    ...     'device_data/',
    ...     device_type='N',
    ...     device_label='Sample_A',
    ...     output_dir='Sample_A_analysis/'
    ... )
    >>> 
    >>> # Check stability status
    >>> print(f"Device stable: {results['stability_summary']['overall_stable']}")
    >>> 
    >>> # Access generated files
    >>> print(f"Report: {results['files']['report']}")
    >>> print(f"Animation: {results['files']['animation']}")
    """
    start_time = time.time()
    
    if verbose:
        print(f"🚀 Starting Complete OECT Analysis Workflow")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Device type: {device_type}")
        print(f"Device label: {device_label}")
        print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results structure
    results = {
        'workflow_info': {
            'device_label': device_label,
            'device_type': device_type,
            'data_folder': data_folder,
            'output_dir': output_dir,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'drift_threshold': drift_threshold,
                'plot_style': plot_style,
                'animation_style': animation_style
            }
        },
        'files': {},
        'analysis_results': {},
        'stability_summary': {},
        'warnings': []
    }
    
    try:
        # Step 1: Load transfer data
        if verbose:
            print(f"\n📁 Step 1: Loading transfer data...")
        
        transfer_objects = ot.load_all_transfer_files(
            data_folder, 
            device_type=device_type,
            sort_numerically=True
        )
        
        if not transfer_objects:
            raise ValueError("No transfer files found in the specified folder")
        
        results['workflow_info']['n_measurements'] = len(transfer_objects)
        
        if verbose:
            print(f"   ✅ Loaded {len(transfer_objects)} measurements")
        
        # Step 2: Time-series analysis
        if verbose:
            print(f"\n🔍 Step 2: Time-series analysis and drift detection...")
        
        import numpy as np
        time_array = np.array(time_values) if time_values else None
        
        extractor = analyze_transfer_stability(
            transfer_objects,
            device_type=device_type,
            time_values=time_array,
            drift_threshold=drift_threshold,
            verbose=verbose
        )
        
        # Export time-series data
        df = extractor.to_dataframe()
        csv_path = os.path.join(output_dir, f'{device_label}_time_series.csv')
        df.to_csv(csv_path, index=False)
        results['files']['time_series_csv'] = csv_path
        
        # Trend detection
        trends = detect_parameter_trends(extractor)
        trends_path = os.path.join(output_dir, f'{device_label}_trends.csv')
        trends.to_csv(trends_path, index=False)
        results['files']['trends_csv'] = trends_path
        
        # Store analysis results
        results['analysis_results']['extractor'] = extractor
        results['analysis_results']['trends'] = trends.to_dict('records')
        
        # Generate stability summary
        stable_params = trends[~trends['significant']]['parameter'].tolist()
        drift_params = trends[trends['significant']]['parameter'].tolist()
        
        results['stability_summary'] = {
            'overall_stable': len(drift_params) == 0,
            'stable_parameters': stable_params,
            'drift_parameters': drift_params,
            'max_drift_percent': trends['relative_slope_percent'].abs().max() if not trends.empty else 0,
            'n_stable': len(stable_params),
            'n_drift': len(drift_params)
        }
        
        if verbose:
            print(f"   📊 Stability Status: {'✅ STABLE' if results['stability_summary']['overall_stable'] else '⚠️ DRIFT DETECTED'}")
            if drift_params:
                print(f"   📈 Parameters showing drift: {', '.join(drift_params)}")
        
        # Step 3: Generate HTML report
        if generate_report:
            if verbose:
                print(f"\n📋 Step 3: Generating stability report...")
            
            report_path = os.path.join(output_dir, f'{device_label}_stability_report.html')
            generate_stability_report(extractor, report_path)
            results['files']['report'] = report_path
        
        # Step 4: Create plots
        if create_plots and _PLOTTING_AVAILABLE:
            if verbose:
                print(f"\n📊 Step 4: Creating visualizations...")
            
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Transfer evolution plot
            evolution_path = os.path.join(plots_dir, f'{device_label}_evolution.png')
            plot_transfer_evolution(
                transfer_objects, 
                label=device_label,
                style=plot_style,
                save_path=evolution_path
            )
            results['files']['evolution_plot'] = evolution_path
            
            # Comparison plot (initial vs final)
            if len(transfer_objects) > 1:
                comparison_path = os.path.join(plots_dir, f'{device_label}_comparison.png')
                plot_transfer_comparison(
                    transfer_objects,
                    indices=[0, -1],
                    labels=['Initial', 'Final'],
                    style=plot_style,
                    save_path=comparison_path
                )
                results['files']['comparison_plot'] = comparison_path
            
            # Parameter trends plot
            trends_plot_path = os.path.join(plots_dir, f'{device_label}_parameter_trends.png')
            plot_parameter_trends(
                extractor,
                style=plot_style,
                save_path=trends_plot_path
            )
            results['files']['trends_plot'] = trends_plot_path
            
            # Publication plots
            pub_plots = create_publication_plots(
                transfer_objects,
                output_dir=plots_dir,
                device_label=device_label
            )
            results['files']['publication_plots'] = pub_plots
            
            if verbose:
                print(f"   📈 Generated {len(results['files']) - 2} plot files")  # -2 for csv files
        
        elif create_plots and not _PLOTTING_AVAILABLE:
            warning_msg = "Plotting requested but matplotlib not available"
            results['warnings'].append(warning_msg)
            if verbose:
                print(f"   ⚠️ {warning_msg}")
        
        # Step 5: Create animations
        if create_animations and _ANIMATION_AVAILABLE:
            if verbose:
                print(f"\n🎬 Step 5: Creating animations...")
            
            # Main evolution animation
            animation_path = os.path.join(output_dir, f'{device_label}_evolution.mp4')
            generate_transfer_animation(
                transfer_objects,
                output_path=animation_path,
                style=animation_style,
                fps=30,
                dpi=120
            )
            results['files']['animation'] = animation_path
            
            # Animation preview
            preview_indices = [0, len(transfer_objects)//4, len(transfer_objects)//2, 
                             3*len(transfer_objects)//4, -1]
            preview_indices = [i for i in preview_indices if i < len(transfer_objects)]
            
            preview_path = os.path.join(output_dir, f'{device_label}_preview.png')
            create_animation_preview(
                transfer_objects,
                indices=preview_indices,
                output_path=preview_path,
                style=animation_style
            )
            results['files']['animation_preview'] = preview_path
            
            if verbose:
                print(f"   🎥 Animation generation completed")
        
        elif create_animations and not _ANIMATION_AVAILABLE:
            warning_msg = "Animation requested but dependencies not available"
            results['warnings'].append(warning_msg)
            if verbose:
                print(f"   ⚠️ {warning_msg}")
        
        # Final summary
        total_time = time.time() - start_time
        results['workflow_info']['total_time_seconds'] = total_time
        results['workflow_info']['completion_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        if verbose:
            print(f"\n🎉 Workflow Completed Successfully!")
            print(f"{'='*60}")
            print(f"⏱️  Total time: {total_time:.1f} seconds")
            print(f"📁 Output directory: {output_dir}")
            print(f"📊 Stability status: {'STABLE' if results['stability_summary']['overall_stable'] else 'DRIFT DETECTED'}")
            print(f"📄 Generated {len(results['files'])} output files")
            
            if results['warnings']:
                print(f"⚠️  {len(results['warnings'])} warnings:")
                for warning in results['warnings']:
                    print(f"   - {warning}")
        
        return results
    
    except Exception as e:
        error_msg = f"Workflow failed: {str(e)}"
        results['error'] = error_msg
        if verbose:
            print(f"\n❌ {error_msg}")
        raise


def quick_stability_check(
    transfer_objects: List[Dict[str, Any]],
    drift_threshold: float = 0.1,
    verbose: bool = True
) -> str:
    """
    Quick stability assessment for device screening.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    drift_threshold : float, default 0.1
        Threshold for drift detection (10% for quick check)
    verbose : bool, default True
        Whether to print results
        
    Returns
    -------
    str
        Stability status: 'STABLE', 'MODERATE_DRIFT', or 'SIGNIFICANT_DRIFT'
    """
    if len(transfer_objects) < 3:
        return 'INSUFFICIENT_DATA'
    
    # Quick analysis using first, middle, and last measurements
    key_indices = [0, len(transfer_objects)//2, -1]
    key_objects = [transfer_objects[i] for i in key_indices]
    
    extractor = TransferTimeSeriesExtractor(key_objects)
    extractor.extract_time_series()
    
    # Check key parameters
    key_params = ['gm_max_raw', 'Von_raw', 'I_max_raw']
    max_drift = 0
    
    for param in key_params:
        try:
            drift_result = extractor.detect_drift_advanced(param, threshold=drift_threshold/2)
            if 'error' not in drift_result:
                total_drift = abs(drift_result['total_drift'])
                max_drift = max(max_drift, total_drift)
        except:
            continue
    
    # Determine status
    if max_drift < drift_threshold/2:
        status = 'STABLE'
    elif max_drift < drift_threshold:
        status = 'MODERATE_DRIFT'
    else:
        status = 'SIGNIFICANT_DRIFT'
    
    if verbose:
        print(f"📊 Quick Stability Check: {status}")
        print(f"   Maximum drift detected: {max_drift*100:.1f}%")
        print(f"   Threshold: {drift_threshold*100:.1f}%")
    
    return status


def batch_comparison_workflow(
    data_folders: List[str],
    device_labels: Optional[List[str]] = None,
    output_dir: str = "batch_comparison",
    device_type: str = "N",
    comparison_indices: Optional[List[int]] = None,
    create_summary_plots: bool = True
) -> Dict[str, Any]:
    """
    Compare multiple devices or conditions in batch.
    
    Parameters
    ----------
    data_folders : List[str]
        List of data folder paths
    device_labels : List[str], optional
        Labels for each device/condition
    output_dir : str, default "batch_comparison"
        Output directory
    device_type : str, default "N"
        Device type
    comparison_indices : List[int], optional
        Specific measurement indices to compare
    create_summary_plots : bool, default True
        Whether to create summary comparison plots
        
    Returns
    -------
    Dict[str, Any]
        Batch comparison results
    """
    print(f"🔄 Starting Batch Comparison Workflow")
    print(f"{'='*50}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if device_labels is None:
        device_labels = [f"Device_{i+1}" for i in range(len(data_folders))]
    
    if len(device_labels) != len(data_folders):
        raise ValueError("Number of device labels must match number of data folders")
    
    results = {
        'devices': {},
        'comparison_summary': {},
        'files': {}
    }
    
    # Process each device
    all_transfer_objects = []
    all_extractors = []
    
    for i, (folder, label) in enumerate(zip(data_folders, device_labels)):
        print(f"\n📁 Processing {label} ({i+1}/{len(data_folders)})...")
        
        try:
            # Load data
            transfer_objects = ot.load_all_transfer_files(folder, device_type)
            
            # Quick analysis
            extractor = analyze_transfer_stability(
                transfer_objects,
                device_type=device_type,
                verbose=False
            )
            
            stability_status = quick_stability_check(transfer_objects, verbose=False)
            
            results['devices'][label] = {
                'folder': folder,
                'n_measurements': len(transfer_objects),
                'stability_status': stability_status,
                'extractor': extractor
            }
            
            all_transfer_objects.append((label, transfer_objects))
            all_extractors.append((label, extractor))
            
            print(f"   ✅ {label}: {len(transfer_objects)} measurements, {stability_status}")
            
        except Exception as e:
            print(f"   ❌ {label}: Error - {str(e)}")
            results['devices'][label] = {'error': str(e)}
    
    # Create comparison plots if requested
    if create_summary_plots and _PLOTTING_AVAILABLE and all_transfer_objects:
        print(f"\n📊 Creating comparison plots...")
        
        # If comparison_indices not specified, use representative measurements
        if comparison_indices is None:
            # Use first measurement from each device
            comparison_indices = [0]
        
        for idx in comparison_indices:
            comparison_objects = []
            comparison_labels = []
            
            for label, transfer_objects in all_transfer_objects:
                if idx < len(transfer_objects):
                    comparison_objects.append(transfer_objects[idx])
                    comparison_labels.append(label)
            
            if comparison_objects:
                plot_path = os.path.join(output_dir, f'device_comparison_idx_{idx}.png')
                plot_transfer_comparison(
                    comparison_objects,
                    indices=list(range(len(comparison_objects))),
                    labels=comparison_labels,
                    style='publication',
                    save_path=plot_path
                )
                
                results['files'][f'comparison_plot_idx_{idx}'] = plot_path
    
    # Generate summary
    stable_devices = [label for label, data in results['devices'].items() 
                     if data.get('stability_status') == 'STABLE']
    drift_devices = [label for label, data in results['devices'].items() 
                    if data.get('stability_status') in ['MODERATE_DRIFT', 'SIGNIFICANT_DRIFT']]
    
    results['comparison_summary'] = {
        'total_devices': len(data_folders),
        'processed_successfully': len([d for d in results['devices'].values() if 'error' not in d]),
        'stable_devices': stable_devices,
        'drift_devices': drift_devices,
        'stability_rate': len(stable_devices) / len(data_folders) * 100
    }
    
    print(f"\n📋 Batch Comparison Summary:")
    print(f"   Total devices: {results['comparison_summary']['total_devices']}")
    print(f"   Stable devices: {len(stable_devices)} ({results['comparison_summary']['stability_rate']:.1f}%)")
    print(f"   Devices with drift: {len(drift_devices)}")
    
    return results