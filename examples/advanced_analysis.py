#!/usr/bin/env python3
"""
Advanced analysis example for OECT Transfer Analysis package.

This example demonstrates:
1. Advanced data loading with custom parameters
2. Comprehensive stability analysis
3. Custom visualization with multiple colormaps
4. High-quality animation generation
5. Statistical analysis and reporting
6. Memory-optimized processing for large datasets

Prerequisites:
- Large dataset of transfer curve CSV files
- oect-transfer-analysis[animation] package installed
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from oect_transfer_analysis import (
    DataLoader,
    TimeSeriesAnalyzer,
    Visualizer,
    AnimationGenerator,
    check_dependencies,
    create_example_data,
    estimate_animation_time,
    ProgressTracker
)


def advanced_data_loading_demo():
    """Demonstrate advanced data loading features."""
    print("=== Advanced Data Loading ===")
    
    # Create larger synthetic dataset for demonstration
    print("Creating large synthetic dataset...")
    transfer_objects = create_example_data(n_points=150, n_files=100)
    
    # Simulate different file naming patterns
    for i, obj in enumerate(transfer_objects):
        if i < 20:
            obj['filename'] = f'device_A_transfer_{i+1}.csv'
        elif i < 40:
            obj['filename'] = f'device_B_transfer_{i-19}.csv'
        else:
            obj['filename'] = f'stability_test_{i-39:03d}_transfer.csv'
    
    print(f"Created {len(transfer_objects)} transfer curves")
    
    # Show file sorting demonstration
    filenames = [obj['filename'] for obj in transfer_objects[:10]]
    print("\nFirst 10 filenames:")
    for i, name in enumerate(filenames):
        print(f"  {i+1:2d}. {name}")
    
    return transfer_objects


def comprehensive_stability_analysis(transfer_objects):
    """Perform comprehensive stability analysis."""
    print("\n=== Comprehensive Stability Analysis ===")
    
    analyzer = TimeSeriesAnalyzer(transfer_objects)
    time_series = analyzer.extract_time_series()
    
    # Multi-parameter stability analysis
    print("\n1. Multi-parameter stability analysis...")
    all_parameters = [
        'gm_max_raw', 'gm_max_forward', 'gm_max_reverse',
        'I_max_raw', 'I_max_forward', 'I_max_reverse',
        'I_min_raw', 'I_min_forward', 'I_min_reverse',
        'Von_raw', 'Von_forward', 'Von_reverse',
        'absgm_max_raw', 'absI_max_raw', 'absI_min_raw'
    ]
    
    stability_results = analyzer.analyze_stability(
        parameters=all_parameters,
        threshold=0.03  # 3% threshold
    )
    
    print("\nStability Analysis Results:")
    print(stability_results.to_string(index=False))
    
    # Detailed drift analysis for key parameters
    print("\n2. Detailed drift analysis...")
    key_params = ['gm_max_raw', 'I_max_raw', 'Von_raw', 'absgm_max_raw']
    
    drift_details = {}
    for param in key_params:
        drift = analyzer.detect_drift(param, threshold=0.02)
        drift_details[param] = drift
        
        print(f"\n{param}:")
        print(f"  Drift detected: {drift['drift_detected']}")
        print(f"  Max drift: {drift['max_drift_percent']:.3f}%")
        print(f"  Final drift: {drift['final_drift_percent']:.3f}%")
        print(f"  Direction: {drift['drift_direction']}")
        print(f"  Initial value: {drift['initial_value']:.3e}")
        print(f"  Final value: {drift['final_value']:.3e}")
    
    # Statistical summary with correlation analysis
    print("\n3. Statistical analysis...")
    stats = analyzer.get_summary_statistics()
    
    # Calculate coefficient of variation (CV) ranking
    cv_ranking = stats['cv_percent'].sort_values(ascending=True)
    print(f"\nMost stable parameters (lowest CV):")
    for i, (param, cv) in enumerate(cv_ranking.head().items()):
        print(f"  {i+1}. {param}: {cv:.2f}% CV")
    
    print(f"\nLeast stable parameters (highest CV):")
    for i, (param, cv) in enumerate(cv_ranking.tail().items()):
        print(f"  {i+1}. {param}: {cv:.2f}% CV")
    
    return analyzer, stability_results, drift_details


def advanced_visualization_demo(transfer_objects, analyzer):
    """Demonstrate advanced visualization capabilities."""
    print("\n=== Advanced Visualization ===")
    
    viz = Visualizer()
    
    # 1. Multiple colormap comparison
    print("\n1. Creating evolution plots with different colormaps...")
    colormaps = ['black_to_red', 'viridis', 'plasma', 'Reds', 'Blues']
    
    for i, cmap in enumerate(colormaps):
        try:
            fig, ax = viz.plot_evolution(
                transfer_objects,
                label=f"Device Analysis",
                colormap=cmap,
                y_scale="log",
                save_path=f"evolution_{cmap}.png",
                figsize=(10, 6),
                dpi=200
            )
            print(f"  ✓ {cmap} colormap plot saved")
        except Exception as e:
            print(f"  ✗ Error with {cmap}: {e}")
    
    # 2. Multi-stage comparison
    print("\n2. Creating multi-stage comparison...")
    n_files = len(transfer_objects)
    
    # Compare multiple stages of degradation
    indices = [0, n_files//5, 2*n_files//5, 3*n_files//5, 4*n_files//5, n_files-1]
    labels = ["Fresh", "20%", "40%", "60%", "80%", "Final"]
    
    try:
        fig, ax = viz.plot_comparison(
            transfer_objects,
            indices=indices,
            labels=labels,
            colormap="viridis",
            save_path="multi_stage_comparison.png",
            linewidth=3,
            alpha=0.8
        )
        print("  ✓ Multi-stage comparison saved")
    except Exception as e:
        print(f"  ✗ Error in multi-stage comparison: {e}")
    
    # 3. Custom time series plots with different scales
    print("\n3. Creating custom time series plots...")
    
    # Linear scale plot
    try:
        fig, axes = analyzer.plot_time_series(
            parameters=['gm_max_raw', 'I_max_raw'],
            save_path="time_series_linear.png",
            figsize=(14, 8),
            dpi=200
        )
        print("  ✓ Linear time series plot saved")
    except Exception as e:
        print(f"  ✗ Error in linear time series: {e}")
    
    # Custom subplot arrangement
    print("\n4. Creating custom subplot grid...")
    try:
        from oect_transfer_analysis.visualization import create_subplot_grid
        
        # Select representative curves
        sample_indices = np.linspace(0, len(transfer_objects)-1, 9, dtype=int)
        
        fig, axes = create_subplot_grid(
            transfer_objects,
            indices=sample_indices,
            ncols=3,
            figsize=(15, 12),
            color='red',
            alpha=0.8
        )
        
        plt.savefig("subplot_grid.png", dpi=150, bbox_inches='tight')
        print("  ✓ Subplot grid saved")
        plt.close()
        
    except Exception as e:
        print(f"  ✗ Error in subplot grid: {e}")


def high_quality_animation_demo(transfer_objects):
    """Demonstrate high-quality animation generation."""
    print("\n=== High-Quality Animation Generation ===")
    
    # Check animation dependencies
    deps = check_dependencies(verbose=False)
    if not (deps.get('cv2', False) and deps.get('PIL', False)):
        print("Animation dependencies not available. Skipping animation demo.")
        return
    
    # Estimate animation time
    print("\n1. Estimating animation generation time...")
    time_est = estimate_animation_time(
        n_frames=len(transfer_objects),
        dpi=150,
        n_workers=None
    )
    
    print(f"Estimated time for {time_est['frames']} frames:")
    print(f"  Sequential: {time_est['sequential_estimate']:.1f}s")
    print(f"  Parallel ({time_est['workers']} workers): {time_est['parallel_estimate']:.1f}s")
    print(f"  Total with encoding: {time_est['total_estimate']:.1f}s")
    
    generator = AnimationGenerator()
    
    # 2. Standard high-quality animation
    print("\n2. Generating standard high-quality animation...")
    try:
        output_path = generator.generate_animation(
            transfer_objects[:50],  # Use subset for demo
            "high_quality_evolution.mp4",
            fps=30,
            dpi=150,
            figsize=(14, 6),
            codec='mp4v',
            verbose=True
        )
        print(f"  ✓ High-quality animation saved: {output_path}")
    except Exception as e:
        print(f"  ✗ Error in standard animation: {e}")
    
    # 3. Memory-optimized animation for large datasets
    if len(transfer_objects) > 30:
        print("\n3. Generating memory-optimized animation...")
        try:
            output_path = generator.generate_memory_optimized(
                transfer_objects,
                "memory_optimized_evolution.mp4",
                batch_size=25,
                fps=60,
                dpi=120,
                verbose=True
            )
            print(f"  ✓ Memory-optimized animation saved: {output_path}")
        except Exception as e:
            print(f"  ✗ Error in memory-optimized animation: {e}")
    
    # 4. Custom animation with specific parameters
    print("\n4. Generating custom animation with specific parameters...")
    try:
        # Custom coordinate ranges for focused view
        xlim = (-0.5, 0.5)
        ylim_log = (1e-10, 1e-5)
        ylim_linear = (0, 2e-6)
        
        output_path = generator.generate_animation(
            transfer_objects[:30],  # Subset for demo
            "custom_animation.mp4",
            fps=45,
            dpi=200,
            xlim=xlim,
            ylim_log=ylim_log,
            ylim_linear=ylim_linear,
            figsize=(16, 7),
            codec='H264',
            verbose=True
        )
        print(f"  ✓ Custom animation saved: {output_path}")
    except Exception as e:
        print(f"  ✗ Error in custom animation: {e}")


def generate_comprehensive_report(analyzer, stability_results, drift_details):
    """Generate a comprehensive analysis report."""
    print("\n=== Generating Comprehensive Report ===")
    
    # Create detailed report
    report_data = []
    
    # Basic statistics
    stats = analyzer.get_summary_statistics()
    df = analyzer.to_dataframe()
    
    print("\n1. Saving analysis data...")
    
    # Export main results
    df.to_csv("comprehensive_analysis_results.csv", index=False)
    stats.to_csv("comprehensive_statistics.csv")
    stability_results.to_csv("stability_analysis.csv", index=False)
    
    # Create summary report
    print("\n2. Creating summary report...")
    
    report_lines = [
        "# OECT Transfer Analysis - Comprehensive Report",
        f"\n## Dataset Overview",
        f"- Total transfer curves: {len(df)}",
        f"- Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Average data points per curve: {df.groupby('filename')['time_point'].count().mean():.0f}",
        
        f"\n## Stability Analysis Summary",
        f"- Parameters analyzed: {len(stability_results)}",
        f"- Stable parameters (drift < 3%): {(~stability_results['drift_detected']).sum()}",
        f"- Unstable parameters: {stability_results['drift_detected'].sum()}",
        
        f"\n## Key Findings",
    ]
    
    # Add key findings from drift analysis
    for param, drift in drift_details.items():
        if drift['drift_detected']:
            report_lines.append(
                f"- {param}: {drift['drift_direction']} by {abs(drift['final_drift_percent']):.2f}%"
            )
    
    # Parameter stability ranking
    cv_ranking = stats['cv_percent'].sort_values(ascending=True)
    report_lines.extend([
        f"\n## Parameter Stability Ranking (by CV%)",
        f"Most stable:",
    ])
    
    for i, (param, cv) in enumerate(cv_ranking.head().items()):
        report_lines.append(f"  {i+1}. {param}: {cv:.2f}%")
    
    report_lines.extend([
        f"\nLeast stable:",
    ])
    
    for i, (param, cv) in enumerate(cv_ranking.tail().items()):
        report_lines.append(f"  {i+1}. {param}: {cv:.2f}%")
    
    # Add recommendations
    report_lines.extend([
        f"\n## Recommendations",
        f"- Monitor parameters with CV > 10%: {(stats['cv_percent'] > 10).sum()} parameters",
        f"- Investigate trends in unstable parameters",
        f"- Consider environmental factors for drift > 5%",
    ])
    
    # Save report
    with open("comprehensive_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print("  ✓ Comprehensive report saved as 'comprehensive_report.md'")
    print("  ✓ All analysis data exported to CSV files")


def main():
    """Run the advanced analysis example."""
    print("=== OECT Transfer Analysis - Advanced Example ===\n")
    
    # Check system
    print("1. System check...")
    check_dependencies(verbose=True)
    
    # Advanced data loading
    transfer_objects = advanced_data_loading_demo()
    
    # Comprehensive stability analysis
    analyzer, stability_results, drift_details = comprehensive_stability_analysis(transfer_objects)
    
    # Advanced visualization
    advanced_visualization_demo(transfer_objects, analyzer)
    
    # High-quality animations
    high_quality_animation_demo(transfer_objects)
    
    # Generate comprehensive report
    generate_comprehensive_report(analyzer, stability_results, drift_details)
    
    print("\n=== Advanced Analysis Completed! ===")
    print("\nGenerated files:")
    print("  Visualizations:")
    print("    - evolution_*.png (various colormaps)")
    print("    - multi_stage_comparison.png")
    print("    - time_series_linear.png")
    print("    - subplot_grid.png")
    print("  Animations:")
    print("    - high_quality_evolution.mp4")
    print("    - memory_optimized_evolution.mp4")
    print("    - custom_animation.mp4")
    print("  Analysis Results:")
    print("    - comprehensive_analysis_results.csv")
    print("    - comprehensive_statistics.csv")
    print("    - stability_analysis.csv")
    print("    - comprehensive_report.md")
    
    print(f"\nProcessed {len(transfer_objects)} transfer curves with advanced analysis")
    print("Results ready for publication and further research!")


if __name__ == "__main__":
    main()