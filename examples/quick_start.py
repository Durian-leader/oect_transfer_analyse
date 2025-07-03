#!/usr/bin/env python3
"""
OECT Transfer Analyse - Quick Start Guide

This script provides a minimal example to get started with the package.
Perfect for first-time users and quick testing.
"""

import oect_transfer_analyse as ota


def quick_start_example():
    """
    Minimal example showing the most common usage patterns.
    
    This function demonstrates:
    1. Package information and feature checking
    2. Loading data (assumes you have transfer CSV files)
    3. Basic analysis
    4. Simple visualization (if matplotlib available)
    5. Quick stability assessment
    """
    
    print("🚀 OECT Transfer Analyse - Quick Start")
    print("=" * 45)
    
    # 1. Check package info and features
    print("\n📦 Package Information:")
    ota.info()  # Built-in function to show package info
    
    # 2. Load your data (replace 'your_data_folder' with actual path)
    data_folder = 'your_data_folder'  # ← Change this to your data path
    
    # Check if the example folder exists
    import os
    if not os.path.exists(data_folder):
        print(f"\n⚠️  Data folder '{data_folder}' not found.")
        print("💡 To use this example:")
        print("   1. Replace 'your_data_folder' with path to your CSV files")
        print("   2. Or run the complete demo: python examples/complete_workflow_demo.py")
        return
    
    try:
        print(f"\n📁 Loading data from '{data_folder}'...")
        
        # Load transfer curve files
        transfer_objects = ota.load_all_transfer_files(
            data_folder, 
            device_type='N'  # Change to 'P' for P-type devices
        )
        
        print(f"   ✅ Loaded {len(transfer_objects)} measurements")
        
        # 3. Basic stability analysis
        print(f"\n🔍 Analyzing device stability...")
        
        extractor = ota.analyze_transfer_stability(
            transfer_objects,
            drift_threshold=0.05,  # 5% drift threshold
            verbose=True
        )
        
        # 4. Quick stability check
        print(f"\n⚡ Quick stability assessment...")
        status = ota.quick_stability_check(transfer_objects)
        
        if status == 'STABLE':
            print(f"   ✅ Device appears STABLE")
        elif status == 'MODERATE_DRIFT':
            print(f"   ⚠️  Device shows MODERATE drift")
        elif status == 'SIGNIFICANT_DRIFT':
            print(f"   🚨 Device shows SIGNIFICANT drift")
        
        # 5. Export basic results
        print(f"\n💾 Exporting results...")
        
        # Export time-series data
        df = extractor.to_dataframe()
        df.to_csv('quick_start_results.csv', index=False)
        print(f"   📊 Time-series data: quick_start_results.csv")
        
        # Generate HTML report
        ota.generate_stability_report(extractor, 'quick_start_report.html')
        print(f"   📋 HTML report: quick_start_report.html")
        
        # 6. Basic visualization (if matplotlib available)
        if ota.check_plotting_available():
            print(f"\n📈 Creating basic plots...")
            
            # Simple evolution plot
            ota.plot_transfer_evolution(
                transfer_objects,
                label='Quick_Start_Device',
                save_path='quick_start_evolution.png'
            )
            print(f"   📊 Evolution plot: quick_start_evolution.png")
            
            # Parameter trends
            ota.plot_parameter_trends(
                extractor,
                save_path='quick_start_trends.png'
            )
            print(f"   📈 Parameter trends: quick_start_trends.png")
            
        else:
            print(f"\n💡 For plots, install: pip install matplotlib")
        
        # 7. Animation (if available)
        if ota.check_animation_available():
            print(f"\n🎬 Creating animation...")
            
            ota.generate_transfer_animation(
                transfer_objects,
                output_path='quick_start_animation.mp4',
                fps=5,  # Slow for visibility
                dpi=100
            )
            print(f"   🎥 Animation: quick_start_animation.mp4")
            
        else:
            print(f"\n💡 For animations, install: pip install matplotlib opencv-python")
        
        print(f"\n🎉 Quick start completed successfully!")
        print(f"\n📋 Generated files:")
        print(f"   • quick_start_results.csv (time-series data)")
        print(f"   • quick_start_report.html (detailed report)")
        if ota.check_plotting_available():
            print(f"   • quick_start_evolution.png (transfer curves)")
            print(f"   • quick_start_trends.png (parameter trends)")
        if ota.check_animation_available():
            print(f"   • quick_start_animation.mp4 (evolution video)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   • Check that your data folder contains CSV files")
        print(f"   • Ensure CSV files have 'Vg' and 'Id' columns")
        print(f"   • Verify the device_type ('N' or 'P') is correct")


def demonstrate_different_workflows():
    """Show different ways to use the package."""
    
    print(f"\n🔄 Different Usage Patterns")
    print(f"=" * 35)
    
    # Method 1: One-line complete analysis
    print(f"\n1️⃣  One-line complete analysis:")
    print(f"""
# Complete analysis with all features
results = ota.complete_analysis_workflow(
    'your_data_folder',
    device_type='N',
    device_label='My_Device'
)
    """)
    
    # Method 2: Step-by-step analysis
    print(f"\n2️⃣  Step-by-step analysis:")
    print(f"""
# Load and analyze step by step
transfer_objects = ota.load_all_transfer_files('data/', 'N')
extractor = ota.analyze_transfer_stability(transfer_objects)
trends = ota.detect_parameter_trends(extractor)
ota.plot_parameter_trends(extractor)
    """)
    
    # Method 3: Quick screening
    print(f"\n3️⃣  Quick device screening:")
    print(f"""
# Fast stability check for multiple devices
for folder in ['device_A/', 'device_B/', 'device_C/']:
    transfer_objects = ota.load_all_transfer_files(folder, 'N')
    status = ota.quick_stability_check(transfer_objects)
    print(f"{folder}: {status}")
    """)
    
    # Method 4: Batch comparison
    print(f"\n4️⃣  Batch comparison:")
    print(f"""
# Compare multiple devices/conditions
results = ota.batch_comparison_workflow(
    data_folders=['condition_A/', 'condition_B/', 'condition_C/'],
    device_labels=['Condition A', 'Condition B', 'Condition C']
)
    """)


def show_installation_options():
    """Show different installation options."""
    
    print(f"\n📦 Installation Options")
    print(f"=" * 25)
    
    print(f"""
# Basic installation (core analysis only)
pip install oect-transfer-analyse

# With plotting support
pip install oect-transfer-analyse[plotting]

# With animation support
pip install oect-transfer-analyse[animation]

# Complete installation (all features)
pip install oect-transfer-analyse[all]

# Development installation
git clone https://github.com/yourusername/oect-transfer-analyse.git
cd oect-transfer-analyse
pip install -e .[all,dev]
    """)


def main():
    """Main function for quick start guide."""
    
    # Show installation options
    show_installation_options()
    
    # Show different usage patterns
    demonstrate_different_workflows()
    
    # Run the actual quick start example
    quick_start_example()
    
    print(f"\n💡 Next steps:")
    print(f"   • Try the complete demo: python examples/complete_workflow_demo.py")
    print(f"   • Read the documentation for advanced features")
    print(f"   • Explore your results in the generated files")
    print(f"   • Use the CLI tool: oect-analyse --help")


if __name__ == "__main__":
    main()