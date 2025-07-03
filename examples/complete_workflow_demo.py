#!/usr/bin/env python3
"""
OECT Transfer Analyse - Complete Workflow Demo

This script demonstrates the complete workflow capabilities of the
oect-transfer-analyse package, showcasing all major features including
analysis, visualization, and animation generation.
"""

import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from typing import List

# Import the analysis package
import oect_transfer_analyse as ota


def create_demo_data(output_dir: str = "demo_data", n_measurements: int = 15) -> str:
    """
    Create realistic OECT demo data showing device degradation over time.
    
    Parameters
    ----------
    output_dir : str
        Directory to create demo data
    n_measurements : int
        Number of measurements to generate
        
    Returns
    -------
    str
        Path to created data directory
    """
    print(f"🏗️  Creating demo data with {n_measurements} measurements...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulation parameters
    base_von = -0.3
    base_gm = 1e-6
    base_current = 1e-9
    
    for i in range(n_measurements):
        # Simulate gradual degradation over time
        time_factor = i / (n_measurements - 1)
        
        # Parameter evolution (realistic degradation)
        von_shift = time_factor * 0.08  # Threshold shifts positive
        gm_degradation = 1.0 - time_factor * 0.25  # Transconductance decreases
        current_reduction = 1.0 - time_factor * 0.15  # Current decreases
        noise_increase = 1.0 + time_factor * 1.5  # Noise increases
        
        # Create realistic voltage sweep
        vg_forward = np.linspace(-0.6, 0.2, 50)
        vg_reverse = np.linspace(0.2, -0.6, 50)
        vg_full = np.concatenate([vg_forward, vg_reverse])
        
        # Calculate realistic OECT current
        id_full = []
        for vg in vg_full:
            von_effective = base_von + von_shift
            
            if vg < von_effective:
                # Subthreshold region
                current = (base_current * current_reduction * 
                          np.exp((vg + 0.6) * 12) * gm_degradation)
            else:
                # Above threshold (quadratic + exponential)
                current = (base_current * current_reduction * gm_degradation * 
                          (vg - von_effective)**2 + 
                          base_current * current_reduction * 
                          np.exp((vg + 0.6) * 12) * gm_degradation)
            
            # Add realistic noise
            noise = np.random.normal(0, current * 0.03 * noise_increase)
            current = max(current + noise, 1e-12)
            id_full.append(current)
        
        # Save as CSV with timestamp-like naming
        hours = i * 2  # Every 2 hours
        df = pd.DataFrame({
            'Vg (V)': vg_full,
            'Id (A)': id_full
        })
        
        filename = os.path.join(output_dir, f'transfer_t{hours:03d}h.csv')
        df.to_csv(filename, index=False)
    
    print(f"   ✅ Created {n_measurements} measurement files in '{output_dir}/'")
    return output_dir


def demonstrate_core_features():
    """Demonstrate core analysis features."""
    print(f"\n🔬 Demonstrating Core Analysis Features")
    print(f"{'='*50}")
    
    # Create demo data
    data_dir = create_demo_data("demo_device_data", 12)
    
    # Basic loading (using core package functionality)
    print(f"\n📁 Loading transfer data...")
    transfer_objects = ota.load_all_transfer_files(data_dir, 'N')
    print(f"   Loaded {len(transfer_objects)} measurements")
    
    # Time-series analysis
    print(f"\n🔍 Performing time-series analysis...")
    time_hours = np.array([i * 2 for i in range(len(transfer_objects))])
    
    extractor = ota.analyze_transfer_stability(
        transfer_objects,
        time_values=time_hours,
        drift_threshold=0.05,
        verbose=True
    )
    
    # Export data
    print(f"\n💾 Exporting analysis data...")
    df = extractor.to_dataframe()
    df.to_csv('demo_time_series.csv', index=False)
    print(f"   Time-series data saved to: demo_time_series.csv")
    
    # Trend detection
    trends = ota.detect_parameter_trends(extractor)
    trends.to_csv('demo_trends.csv', index=False)
    print(f"   Trend analysis saved to: demo_trends.csv")
    
    # Generate report
    report_path = ota.generate_stability_report(extractor, 'demo_stability_report.html')
    print(f"   HTML report saved to: {report_path}")
    
    return transfer_objects, extractor


def demonstrate_visualization_features(transfer_objects, extractor):
    """Demonstrate enhanced visualization capabilities."""
    print(f"\n📊 Demonstrating Visualization Features")
    print(f"{'='*45}")
    
    if not ota.check_plotting_available():
        print("   ⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return
    
    print(f"   ✅ Matplotlib available - generating plots...")
    
    # 1. Transfer evolution plot
    print(f"\n1️⃣  Transfer curve evolution...")
    ota.plot_transfer_evolution(
        transfer_objects,
        label='Demo_Device',
        style='publication',
        colormap='plasma',
        save_path='demo_evolution.png'
    )
    
    # 2. Comparison plot
    print(f"\n2️⃣  Initial vs final comparison...")
    ota.plot_transfer_comparison(
        transfer_objects,
        indices=[0, -1],
        labels=['Initial (0h)', 'Final (22h)'],
        style='publication',
        colormap='RdBu_r',
        save_path='demo_comparison.png'
    )
    
    # 3. Parameter trends
    print(f"\n3️⃣  Parameter trends over time...")
    ota.plot_parameter_trends(
        extractor,
        parameters=['gm_max_raw', 'Von_raw', 'I_max_raw'],
        style='publication',
        save_path='demo_parameter_trends.png'
    )
    
    # 4. Drift analysis
    print(f"\n4️⃣  Drift analysis visualization...")
    ota.plot_drift_analysis(
        extractor,
        parameter='gm_max_raw',
        save_path='demo_drift_analysis.png'
    )
    
    # 5. Publication plots
    print(f"\n5️⃣  Complete publication figure set...")
    os.makedirs('demo_publication_figures', exist_ok=True)
    pub_files = ota.create_publication_plots(
        transfer_objects,
        output_dir='demo_publication_figures',
        device_label='Demo_Device'
    )
    
    print(f"   📈 Generated {len(pub_files)} publication-ready figures")


def demonstrate_animation_features(transfer_objects, extractor):
    """Demonstrate animation generation capabilities."""
    print(f"\n🎬 Demonstrating Animation Features")
    print(f"{'='*40}")
    
    if not ota.check_animation_available():
        print("   ⚠️  Animation dependencies not available")
        print("   💡 Install with: pip install oect-transfer-analyse[animation]")
        return
    
    print(f"   ✅ Animation dependencies available - generating videos...")
    
    # 1. Animation preview
    print(f"\n1️⃣  Creating animation preview...")
    preview_indices = [0, 3, 6, 9, 11]
    ota.create_animation_preview(
        transfer_objects,
        indices=preview_indices,
        output_path='demo_animation_preview.png',
        style='publication'
    )
    
    # 2. Standard animation
    print(f"\n2️⃣  Generating standard animation...")
    ota.generate_transfer_animation(
        transfer_objects,
        output_path='demo_evolution_standard.mp4',
        style='standard',
        fps=3,  # Slow for better visibility
        dpi=100
    )
    
    # 3. Publication-quality animation
    print(f"\n3️⃣  Generating publication-quality animation...")
    ota.generate_transfer_animation(
        transfer_objects,
        output_path='demo_evolution_publication.mp4',
        style='publication',
        fps=4,
        dpi=120,
        figsize=(14, 6)
    )
    
    # 4. Parameter animation
    print(f"\n4️⃣  Creating parameter evolution animation...")
    ota.create_parameter_animation(
        extractor,
        output_path='demo_parameter_evolution.mp4',
        parameters=['gm_max_raw', 'Von_raw', 'I_max_raw']
    )
    
    print(f"   🎥 Animation generation completed!")


def demonstrate_workflow_features():
    """Demonstrate pre-defined workflow capabilities."""
    print(f"\n🔄 Demonstrating Workflow Features")
    print(f"{'='*40}")
    
    # Create multiple demo datasets
    datasets = []
    for i, stability in enumerate(['stable', 'moderate_drift', 'significant_drift']):
        data_dir = create_demo_data(f"demo_device_{stability}", 8)
        datasets.append((data_dir, f"Device_{stability.replace('_', ' ').title()}"))
    
    # 1. Complete analysis workflow
    print(f"\n1️⃣  Complete analysis workflow...")
    results = ota.complete_analysis_workflow(
        data_folder=datasets[0][0],
        device_type='N',
        device_label='Stable_Device',
        output_dir='demo_complete_analysis/',
        generate_report=True,
        create_plots=ota.check_plotting_available(),
        create_animations=ota.check_animation_available(),
        verbose=True
    )
    
    print(f"   📋 Workflow Results:")
    print(f"   - Stability: {'STABLE' if results['stability_summary']['overall_stable'] else 'DRIFT DETECTED'}")
    print(f"   - Generated files: {len(results['files'])}")
    print(f"   - Processing time: {results['workflow_info']['total_time_seconds']:.1f}s")
    
    # 2. Quick stability checks
    print(f"\n2️⃣  Quick stability screening...")
    for data_dir, label in datasets:
        transfer_objects = ota.load_all_transfer_files(data_dir, 'N')
        status = ota.quick_stability_check(transfer_objects, verbose=False)
        print(f"   {label}: {status}")
    
    # 3. Batch comparison workflow
    print(f"\n3️⃣  Batch comparison workflow...")
    batch_results = ota.batch_comparison_workflow(
        data_folders=[d[0] for d in datasets],
        device_labels=[d[1] for d in datasets],
        output_dir='demo_batch_comparison/',
        create_summary_plots=ota.check_plotting_available()
    )
    
    print(f"   📊 Batch Results:")
    print(f"   - Total devices: {batch_results['comparison_summary']['total_devices']}")
    print(f"   - Stability rate: {batch_results['comparison_summary']['stability_rate']:.1f}%")
    print(f"   - Stable devices: {len(batch_results['comparison_summary']['stable_devices'])}")


def demonstrate_package_info():
    """Demonstrate package information and capabilities."""
    print(f"\n📦 Package Information & Capabilities")
    print(f"{'='*45}")
    
    # Package info
    info = ota.get_package_info()
    print(f"📊 {info['package']} v{info['version']}")
    print(f"🔧 Built on {info['core_package']} v{info['core_version']}")
    print(f"👨‍💻 Author: {info['author']}")
    
    # Feature availability
    print(f"\n🎯 Feature Availability:")
    features = info['features']
    print(f"   Core Analysis: {'✅' if features['core_analysis'] else '❌'}")
    print(f"   Plotting: {'✅' if features['plotting'] else '❌'}")
    print(f"   Animation: {'✅' if features['animation'] else '❌'}")
    
    # Installation suggestions
    if not features['plotting']:
        print(f"\n💡 Enable plotting: pip install matplotlib")
    if not features['animation']:
        print(f"💡 Enable animation: pip install matplotlib opencv-python")
    
    # Show example usage
    print(f"\n📝 Example Usage:")
    print(ota.get_example_usage())


def cleanup_demo_files():
    """Clean up generated demo files."""
    print(f"\n🧹 Cleanup Options")
    print(f"{'='*20}")
    
    response = input("Delete demo files and directories? (y/N): ")
    if response.lower() == 'y':
        # Remove data directories
        for prefix in ['demo_device', 'demo_complete_analysis', 'demo_batch_comparison', 'demo_publication_figures']:
            for item in os.listdir('.'):
                if item.startswith(prefix):
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
        
        # Remove generated files
        demo_files = [
            'demo_time_series.csv', 'demo_trends.csv', 'demo_stability_report.html',
            'demo_evolution.png', 'demo_comparison.png', 'demo_parameter_trends.png',
            'demo_drift_analysis.png', 'demo_animation_preview.png',
            'demo_evolution_standard.mp4', 'demo_evolution_publication.mp4',
            'demo_parameter_evolution.mp4', 'demo_parameter_evolution.png'
        ]
        
        for filename in demo_files:
            if os.path.exists(filename):
                os.remove(filename)
        
        print("   ✅ Demo files cleaned up")
    else:
        print("   📁 Demo files kept for exploration")
        print("\n📋 Generated files:")
        print("   - CSV data: demo_time_series.csv, demo_trends.csv")
        print("   - HTML report: demo_stability_report.html") 
        print("   - Plots: demo_*.png files")
        print("   - Videos: demo_*.mp4 files")
        print("   - Directories: demo_*/ folders")


def main():
    """Main demo function."""
    print(f"🚀 OECT Transfer Analyse - Complete Workflow Demo")
    print(f"{'='*60}")
    print(f"This demo showcases all major features of the package:")
    print(f"• Core analysis and time-series extraction")
    print(f"• Enhanced visualization with multiple styles")
    print(f"• Animation generation for transfer curve evolution")
    print(f"• Pre-defined workflows for complete analysis")
    print(f"{'='*60}")
    
    try:
        # Show package information
        demonstrate_package_info()
        
        # Core analysis features
        transfer_objects, extractor = demonstrate_core_features()
        
        # Visualization features
        demonstrate_visualization_features(transfer_objects, extractor)
        
        # Animation features
        demonstrate_animation_features(transfer_objects, extractor)
        
        # Workflow features
        demonstrate_workflow_features()
        
        # Show summary
        print(f"\n🎉 Demo Completed Successfully!")
        print(f"{'='*35}")
        print(f"✅ All major features demonstrated")
        print(f"📁 Multiple output files generated")
        print(f"📊 Analysis results available for review")
        
        # Optional cleanup
        cleanup_demo_files()
        
        print(f"\n💡 Next Steps:")
        print(f"• Explore the generated files and reports")
        print(f"• Try the package with your own OECT data")
        print(f"• Check the documentation for advanced features")
        print(f"• Consider contributing to the project!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print(f"💡 Make sure all dependencies are installed:")
        print(f"   pip install oect-transfer-analyse[all]")
        raise


if __name__ == "__main__":
    main()