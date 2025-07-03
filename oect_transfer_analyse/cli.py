# cli.py
"""
Command-line interface for oect-transfer-analyse

Provides convenient command-line tools for common OECT analysis tasks.
"""

import argparse
import sys
import os
from typing import Optional

# Import main package
try:
    import oect_transfer_analyse as ota
except ImportError:
    print("Error: oect-transfer-analyse package not found")
    sys.exit(1)


def cmd_info():
    """Display package information."""
    ota.print_package_info()


def cmd_example():
    """Display example usage."""
    print(ota.get_example_usage())


def cmd_analyze(
    data_folder: str,
    device_type: str = "N",
    output_dir: str = "analysis_results",
    device_label: Optional[str] = None,
    no_plots: bool = False,
    no_animations: bool = False,
    no_report: bool = False,
    drift_threshold: float = 0.05,
    style: str = "publication"
):
    """Run complete analysis workflow."""
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' does not exist")
        sys.exit(1)
    
    if device_label is None:
        device_label = os.path.basename(data_folder.rstrip('/'))
    
    print(f"🚀 Starting OECT Analysis")
    print(f"Data folder: {data_folder}")
    print(f"Device type: {device_type}")
    print(f"Device label: {device_label}")
    print(f"Output directory: {output_dir}")
    
    try:
        results = ota.complete_analysis_workflow(
            data_folder=data_folder,
            device_type=device_type,
            output_dir=output_dir,
            device_label=device_label,
            drift_threshold=drift_threshold,
            generate_report=not no_report,
            create_plots=not no_plots and ota.check_plotting_available(),
            create_animations=not no_animations and ota.check_animation_available(),
            plot_style=style,
            animation_style=style,
            verbose=True
        )
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Generated {len(results['files'])} files")
        
        if results['stability_summary']['overall_stable']:
            print(f"✅ Device appears STABLE")
        else:
            print(f"⚠️  Device shows DRIFT in {len(results['stability_summary']['drift_parameters'])} parameters")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def cmd_quick_check(data_folder: str, device_type: str = "N"):
    """Run quick stability check."""
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' does not exist")
        sys.exit(1)
    
    try:
        transfer_objects = ota.load_all_transfer_files(data_folder, device_type)
        status = ota.quick_stability_check(transfer_objects)
        
        print(f"\n📊 Quick Stability Check Result: {status}")
        
        if status == 'STABLE':
            print(f"✅ Device appears stable")
        elif status == 'MODERATE_DRIFT':
            print(f"⚠️  Device shows moderate drift")
        elif status == 'SIGNIFICANT_DRIFT':
            print(f"🚨 Device shows significant drift")
        else:
            print(f"❓ Could not determine stability (insufficient data)")
            
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        sys.exit(1)


def cmd_batch(
    folders: list,
    output_dir: str = "batch_analysis",
    device_type: str = "N",
    labels: Optional[list] = None
):
    """Run batch comparison analysis."""
    
    # Validate folders
    missing_folders = [f for f in folders if not os.path.exists(f)]
    if missing_folders:
        print(f"Error: Missing folders: {', '.join(missing_folders)}")
        sys.exit(1)
    
    try:
        results = ota.batch_comparison_workflow(
            data_folders=folders,
            device_labels=labels,
            output_dir=output_dir,
            device_type=device_type,
            create_summary_plots=ota.check_plotting_available()
        )
        
        print(f"\n🎉 Batch analysis completed!")
        print(f"Processed {results['comparison_summary']['total_devices']} devices")
        print(f"Stability rate: {results['comparison_summary']['stability_rate']:.1f}%")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OECT Transfer Curve Advanced Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oect-analyse info                           # Show package information
  oect-analyse example                        # Show usage examples
  oect-analyse analyze data/                  # Full analysis of data/ folder
  oect-analyse quick data/                    # Quick stability check
  oect-analyse batch data1/ data2/ data3/     # Batch comparison
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Show package information')
    
    # Example command  
    parser_example = subparsers.add_parser('example', help='Show usage examples')
    
    # Analyze command
    parser_analyze = subparsers.add_parser('analyze', help='Run complete analysis')
    parser_analyze.add_argument('data_folder', help='Path to data folder')
    parser_analyze.add_argument('--device-type', '-t', choices=['N', 'P'], default='N',
                               help='Device type (default: N)')
    parser_analyze.add_argument('--output', '-o', default='analysis_results',
                               help='Output directory (default: analysis_results)')
    parser_analyze.add_argument('--label', '-l', help='Device label')
    parser_analyze.add_argument('--no-plots', action='store_true',
                               help='Skip plot generation')
    parser_analyze.add_argument('--no-animations', action='store_true',
                               help='Skip animation generation')
    parser_analyze.add_argument('--no-report', action='store_true',
                               help='Skip HTML report generation')
    parser_analyze.add_argument('--drift-threshold', type=float, default=0.05,
                               help='Drift detection threshold (default: 0.05)')
    parser_analyze.add_argument('--style', choices=['standard', 'publication', 'minimal'],
                               default='publication', help='Plot/animation style')
    
    # Quick check command
    parser_quick = subparsers.add_parser('quick', help='Run quick stability check')
    parser_quick.add_argument('data_folder', help='Path to data folder')
    parser_quick.add_argument('--device-type', '-t', choices=['N', 'P'], default='N',
                             help='Device type (default: N)')
    
    # Batch command
    parser_batch = subparsers.add_parser('batch', help='Run batch comparison')
    parser_batch.add_argument('folders', nargs='+', help='Data folders to compare')
    parser_batch.add_argument('--output', '-o', default='batch_analysis',
                             help='Output directory (default: batch_analysis)')
    parser_batch.add_argument('--device-type', '-t', choices=['N', 'P'], default='N',
                             help='Device type (default: N)')
    parser_batch.add_argument('--labels', nargs='+', help='Device labels')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'info':
        cmd_info()
    elif args.command == 'example':
        cmd_example()
    elif args.command == 'analyze':
        cmd_analyze(
            data_folder=args.data_folder,
            device_type=args.device_type,
            output_dir=args.output,
            device_label=args.label,
            no_plots=args.no_plots,
            no_animations=args.no_animations,
            no_report=args.no_report,
            drift_threshold=args.drift_threshold,
            style=args.style
        )
    elif args.command == 'quick':
        cmd_quick(
            data_folder=args.data_folder,
            device_type=args.device_type
        )
    elif args.command == 'batch':
        cmd_batch(
            folders=args.folders,
            output_dir=args.output,
            device_type=args.device_type,
            labels=args.labels
        )


if __name__ == "__main__":
    main()