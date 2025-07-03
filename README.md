# OECT Transfer Curve Advanced Analysis

[![PyPI version](https://badge.fury.io/py/oect-transfer-analyse.svg)](https://badge.fury.io/py/oect-transfer-analyse)
[![Python Support](https://img.shields.io/pypi/pyversions/oect-transfer-analyse.svg)](https://pypi.org/project/oect-transfer-analyse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[简体中文](README_CN.md)

An advanced analysis and visualization package for Organic Electrochemical Transistor (OECT) transfer curves, built on top of the core `oect-transfer` package.

## 🎯 Features

### 🔬 Advanced Analysis
- **Time-series Analysis**: Extract and analyze parameter evolution over time
- **Drift Detection**: Automatic detection of parameter drift and device degradation
- **Trend Analysis**: Statistical trend detection with significance testing
- **Stability Assessment**: Comprehensive device stability evaluation
- **Batch Processing**: Compare multiple devices or conditions

### 📊 Enhanced Visualization
- **Publication-ready Plots**: High-quality figures with multiple style options
- **Parameter Trends**: Time-series visualization with trend lines
- **Drift Analysis Plots**: Visual representation of drift patterns
- **Comparison Plots**: Side-by-side device/condition comparisons
- **Custom Styling**: Publication, minimal, and standard plot styles

### 🎬 Animation Generation
- **Transfer Evolution Videos**: MP4 animations showing curve evolution
- **Parameter Animations**: Time-series parameter evolution videos
- **High-performance Generation**: Parallel processing for fast video creation
- **Multiple Formats**: Support for different video codecs and quality settings
- **Batch Animation**: Generate animations for multiple datasets

### 🔄 Pre-defined Workflows
- **Complete Analysis**: End-to-end analysis with all features
- **Quick Stability Check**: Fast device screening
- **Batch Comparison**: Multi-device analysis workflows
- **Custom Pipelines**: Flexible workflow building blocks

## 📦 Installation

### Basic Installation (Core + Analysis)
```bash
pip install oect-transfer-analyse
```

### With Plotting Support
```bash
pip install oect-transfer-analyse[plotting]
```

### With Animation Support  
```bash
pip install oect-transfer-analyse[animation]
```

### Complete Installation (All Features)
```bash
pip install oect-transfer-analyse[all]
```

### Development Installation
```bash
git clone https://github.com/yourusername/oect-transfer-analyse.git
cd oect-transfer-analyse
pip install -e .[all,dev]
```

## 🚀 Quick Start

### Complete Analysis Workflow

```python
import oect_transfer_analyse as ota

# Complete end-to-end analysis
results = ota.complete_analysis_workflow(
    data_folder='device_data/',
    device_type='N',
    device_label='Sample_A',
    output_dir='Sample_A_analysis/',
    create_plots=True,
    create_animations=True
)

# Check results
print(f"Device stable: {results['stability_summary']['overall_stable']}")
print(f"Generated files: {len(results['files'])}")
```

### Time-series Analysis

```python
# Load data using core package
transfer_objects = ota.load_all_transfer_files('data/', 'N')

# Advanced stability analysis
extractor = ota.analyze_transfer_stability(transfer_objects)

# Detect trends
trends = ota.detect_parameter_trends(extractor)
print(trends)

# Generate detailed report
ota.generate_stability_report(extractor, 'stability_report.html')
```

### Enhanced Visualization

```python
# Publication-ready plots
ota.plot_transfer_evolution(
    transfer_objects, 
    style='publication',
    colormap='viridis',
    save_path='figure_1.png'
)

# Parameter trends with analysis
ota.plot_parameter_trends(
    extractor,
    parameters=['gm_max_raw', 'Von_raw'],
    style='publication'
)

# Create complete publication figure set
ota.create_publication_plots(
    transfer_objects,
    output_dir='publication_figures/',
    device_label='Device_A'
)
```

### Animation Generation

```python
# Check if animation is available
if ota.check_animation_available():
    # Generate high-quality animation
    ota.generate_transfer_animation(
        transfer_objects,
        output_path='device_evolution.mp4',
        style='publication',
        fps=30,
        dpi=150
    )
    
    # Create preview image
    ota.create_animation_preview(
        transfer_objects,
        indices=[0, 10, 20, 30, 40],
        output_path='evolution_preview.png'
    )
else:
    print("Install dependencies: pip install oect-transfer-analyse[animation]")
```

### Quick Device Screening

```python
# Quick stability check for device screening
status = ota.quick_stability_check(transfer_objects)
print(f"Stability: {status}")  # 'STABLE', 'MODERATE_DRIFT', or 'SIGNIFICANT_DRIFT'

# Batch comparison of multiple devices
results = ota.batch_comparison_workflow(
    data_folders=['device_A/', 'device_B/', 'device_C/'],
    device_labels=['Device A', 'Device B', 'Device C'],
    output_dir='batch_analysis/'
)
```

## 📋 Core Workflow Functions

### `complete_analysis_workflow()`
Comprehensive end-to-end analysis including:
- Data loading and validation
- Time-series extraction and drift detection  
- Visualization generation
- Animation creation
- HTML report generation

### `quick_stability_check()`
Fast device screening for:
- Quality control in manufacturing
- Initial device characterization
- Batch processing workflows

### `batch_comparison_workflow()`
Multi-device analysis for:
- Comparing different fabrication conditions
- Device optimization studies
- Reliability testing across batches

## 🎨 Plot Styles

### Publication Style
- High-quality figures ready for academic papers
- Professional typography and layout
- Optimized for print and digital publication

### Minimal Style  
- Clean, minimalist design
- Reduced visual clutter
- Perfect for presentations

### Standard Style
- Balanced appearance with good readability
- Suitable for reports and documentation

## 🔧 Architecture

This package is built on top of the core `oect-transfer` package:

```
oect-transfer-analyse/
├── analysis.py          # Time-series analysis and drift detection
├── plotting.py          # Enhanced visualization capabilities  
├── animation.py         # Video generation functionality
├── workflows.py         # Pre-defined analysis workflows
└── __init__.py         # Package interface and dependency management
```

### Dependency Management
- **Core Functions**: Only require `oect-transfer` + `numpy` + `pandas`
- **Plotting**: Optional `matplotlib` dependency
- **Animation**: Optional `matplotlib` + `opencv-python` dependencies
- **Graceful Degradation**: Clear error messages when optional dependencies are missing

## 📊 Analysis Capabilities

### Time-series Parameters Extracted
- `gm_max_raw/forward/reverse`: Maximum transconductance
- `I_max_raw/forward/reverse`: Maximum current
- `I_min_raw/forward/reverse`: Minimum current  
- `Von_raw/forward/reverse`: Threshold voltage
- `absgm_max_raw/forward/reverse`: Maximum absolute transconductance
- `absI_max_raw/forward/reverse`: Maximum absolute current
- `absI_min_raw/forward/reverse`: Minimum absolute current

### Drift Detection Methods
- **Relative Drift**: Percentage change from initial value
- **Absolute Drift**: Absolute change in parameter values
- **Moving Window Analysis**: Local drift detection with configurable window size
- **Trend Analysis**: Linear regression-based trend detection

### Stability Metrics
- **Coefficient of Variation**: Parameter variability assessment
- **Stability Score**: Overall device stability rating
- **Drift Events**: Identification of significant drift occurrences
- **Trend Strength**: Statistical significance of parameter trends

## 🎯 Use Cases

### Academic Research
- Device stability studies
- Parameter evolution analysis
- Publication-ready figure generation
- Comparative studies across devices

### Industrial Applications  
- Quality control in manufacturing
- Device reliability assessment
- Batch-to-batch variation analysis
- Process optimization studies

### Educational Applications
- Interactive demonstrations
- Data analysis tutorials
- Visualization examples
- Research training

## 📈 Performance

### Animation Generation
- **Parallel Processing**: Multi-core CPU utilization
- **Memory Optimization**: Efficient handling of large datasets
- **Multiple Codecs**: Support for various video formats
- **Speed**: 3-5x faster than sequential processing

### Analysis Speed
- **Vectorized Operations**: NumPy-optimized calculations
- **Efficient Algorithms**: Optimized for large time-series datasets
- **Caching**: Intelligent result caching for repeated analyses

## 🤝 Integration

### With Core Package
```python
import oect_transfer as ot          # Core functionality
import oect_transfer_analyse as ota # Advanced analysis

# Seamless integration
transfer_objects = ot.load_all_transfer_files('data/', 'N')
results = ota.complete_analysis_workflow(transfer_objects)
```

### With Jupyter Notebooks
Perfect for interactive analysis and visualization in Jupyter environments.

### With Automated Pipelines
Designed for integration into automated analysis pipelines and batch processing systems.

## 📚 Documentation

- **API Reference**: Complete function documentation with examples
- **Tutorials**: Step-by-step guides for common workflows  
- **Examples**: Real-world analysis examples
- **Best Practices**: Guidelines for effective OECT analysis

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=oect_transfer_analyse

# Run specific test categories
pytest tests/test_analysis.py
pytest tests/test_plotting.py
pytest tests/test_animation.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/oect-transfer-analyse.git
cd oect-transfer-analyse
pip install -e .[all,dev]
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the excellent `oect-transfer` core package
- Inspired by the OECT research community
- Thanks to all contributors and users

## 📞 Support

- **Documentation**: [Read the Docs](https://oect-transfer-analyse.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/oect-transfer-analyse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/oect-transfer-analyse/discussions)

## 🔗 Related Projects

- **[oect-transfer](https://github.com/Durian-leader/oect_transfer)**: Core OECT transfer curve analysis

---

*Made with ❤️ for the OECT research community*