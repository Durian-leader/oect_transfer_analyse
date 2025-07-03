#!/usr/bin/env python3
"""
Basic usage example for OECT Transfer Analysis package.

This example demonstrates:
1. Loading transfer curve data from CSV files
2. Basic time series analysis
3. Creating visualizations
4. Generating simple animations

Prerequisites:
- CSV files with transfer curve data in a folder
- oect-transfer-analysis package installed
"""

import os
from pathlib import Path

# Import the main components
from oect_transfer_analysis import (
    DataLoader, 
    TimeSeriesAnalyzer, 
    Visualizer,
    check_dependencies,
    create_example_data
)


def main():
    """Run the basic usage example."""
    
    print("=== OECT Transfer Analysis - Basic Usage Example ===\n")
    
    # Check if all dependencies are available
    print("1. Checking dependencies...")
    deps = check_dependencies(verbose=True)
    
    # For this example, we'll use synthetic data if no real data is available
    use_real_data = False
    data_folder = "example_data"  # Change this to your data folder path
    
    if os.path.exists(data_folder):
        csv_files = list(Path(data_folder).glob("*.csv"))
        if csv_files:
            use_real_data = True
            print(f"\n2. Found {len(csv_files)} CSV files in {data_folder}")
        else:
            print(f"\n2. No CSV files found in {data_folder}, using synthetic data")
    else:
        print(f"\n2. Folder {data_folder} not found, using synthetic data")
    
    # Load data
    if use_real_data:
        print("\n3. Loading real data...")
        loader = DataLoader(data_folder)
        transfer_objects = loader.load_all_files(
            device_type="N",  # Change to "P" for P-type devices
            file_pattern="transfer",  # Adjust pattern if needed
            sort_numerically=True,
            verbose=True
        )
        
        # Show loading summary
        summary = loader.analyze_batch(show_details=False)
        print(f"\nLoaded {len(transfer_objects)} files successfully")
        print(f"Data points per file: {summary['data_points'].mean():.0f} ± {summary['data_points'].std():.0f}")
    
    else:
        print("\n3. Creating synthetic data for demonstration...")
        transfer_objects = create_example_data(n_points=100, n_files=20)
        print(f"Created {len(transfer_objects)} synthetic transfer curves")
    
    # Time series analysis
    print("\n4. Performing time series analysis...")
    analyzer = TimeSeriesAnalyzer(transfer_objects)
    time_series = analyzer.extract_time_series()
    
    print(f"Extracted time series data with {len(time_series.filenames)} time points")
    
    # Get statistical summary
    print("\n5. Statistical analysis...")
    stats = analyzer.get_summary_statistics()
    print("\nKey statistics (first 5 parameters):")
    print(stats.head().round(6))
    
    # Drift detection
    print("\n6. Drift detection...")
    key_parameters = ['gm_max_raw', 'I_max_raw', 'Von_raw']
    
    for param in key_parameters:
        try:
            drift_result = analyzer.detect_drift(param, threshold=0.05)  # 5% threshold
            status = "DETECTED" if drift_result['drift_detected'] else "Not detected"
            direction = drift_result['drift_direction']
            magnitude = abs(drift_result['final_drift_percent'])
            
            print(f"  {param}: {status}")
            if drift_result['drift_detected']:
                print(f"    Direction: {direction}")
                print(f"    Magnitude: {magnitude:.2f}%")
        except Exception as e:
            print(f"  {param}: Error in analysis - {e}")
    
    # Visualization
    print("\n7. Creating visualizations...")
    viz = Visualizer()
    
    # Evolution plot with black-to-red colormap
    print("  - Creating evolution plot...")
    try:
        fig, ax = viz.plot_evolution(
            transfer_objects,
            label="Example Device",
            colormap="black_to_red",
            y_scale="log",
            save_path="example_evolution.png"
        )
        print("    ✓ Evolution plot saved as 'example_evolution.png'")
    except Exception as e:
        print(f"    ✗ Error creating evolution plot: {e}")
    
    # Comparison plot
    if len(transfer_objects) >= 3:
        print("  - Creating comparison plot...")
        try:
            # Compare first, middle, and last measurements
            n_files = len(transfer_objects)
            indices = [0, n_files//2, n_files-1]
            labels = ["Initial", "Middle", "Final"]
            
            fig, ax = viz.plot_comparison(
                transfer_objects,
                indices=indices,
                labels=labels,
                save_path="example_comparison.png"
            )
            print("    ✓ Comparison plot saved as 'example_comparison.png'")
        except Exception as e:
            print(f"    ✗ Error creating comparison plot: {e}")
    
    # Time series plot
    print("  - Creating time series plot...")
    try:
        fig, axes = analyzer.plot_time_series(
            parameters=['gm_max_raw', 'I_max_raw', 'Von_raw'],
            save_path="example_time_series.png"
        )
        print("    ✓ Time series plot saved as 'example_time_series.png'")
    except Exception as e:
        print(f"    ✗ Error creating time series plot: {e}")
    
    # Animation (if dependencies available)
    animation_available = deps.get('cv2', False) and deps.get('PIL', False)
    
    if animation_available and len(transfer_objects) >= 5:
        print("\n8. Generating animation...")
        try:
            output_path = viz.generate_animation(
                transfer_objects,
                "example_evolution.mp4",
                fps=10,  # Lower fps for faster generation in example
                dpi=100,  # Lower DPI for faster generation
                verbose=True
            )
            print(f"    ✓ Animation saved as '{output_path}'")
        except Exception as e:
            print(f"    ✗ Error generating animation: {e}")
    else:
        if not animation_available:
            print("\n8. Animation skipped (dependencies not available)")
            print("   Install with: pip install oect-transfer-analysis[animation]")
        else:
            print("\n8. Animation skipped (need at least 5 files)")
    
    # Export data
    print("\n9. Exporting results...")
    try:
        # Export to CSV
        df = analyzer.to_dataframe()
        df.to_csv("example_results.csv", index=False)
        print("    ✓ Analysis results saved as 'example_results.csv'")
        
        # Export statistics
        stats.to_csv("example_statistics.csv")
        print("    ✓ Statistics saved as 'example_statistics.csv'")
        
    except Exception as e:
        print(f"    ✗ Error exporting data: {e}")
    
    print("\n=== Example completed successfully! ===")
    print("\nGenerated files:")
    print("  - example_evolution.png (transfer curve evolution)")
    print("  - example_comparison.png (multi-point comparison)")
    print("  - example_time_series.png (parameter evolution)")
    print("  - example_results.csv (complete analysis data)")
    print("  - example_statistics.csv (statistical summary)")
    if animation_available and len(transfer_objects) >= 5:
        print("  - example_evolution.mp4 (evolution animation)")
    
    print(f"\nAnalyzed {len(transfer_objects)} transfer curves")
    print("Ready for further analysis or publication!")


if __name__ == "__main__":
    main()