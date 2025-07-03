# OECT Transfer Analysis

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](https://github.com/yourusername/oect-transfer-analysis)

[简体中文](https://github.com/Durian-leader/oect_transfer/blob/main/README_CN.md)


Advanced analysis tools for Organic Electrochemical Transistor (OECT) transfer curves, providing comprehensive time series analysis, visualization, and animation capabilities.

## 🚀 Key Features

- **📁 Batch Data Loading**: Automatically load and process multiple transfer curve CSV files
- **📈 Time Series Analysis**: Extract parameter evolution over time with drift detection
- **🎨 Advanced Visualization**: Create publication-ready plots with custom color schemes
- **🎬 Animation Generation**: Generate high-quality videos showing device evolution
- **📊 Statistical Analysis**: Comprehensive statistical summaries and stability analysis
- **⚡ Performance Optimized**: Parallel processing for fast animation generation

## 📦 Installation

### Basic Installation

```bash
pip install oect-transfer-analysis
```

### With Animation Support

```bash
pip install oect-transfer-analysis[animation]
```

### Development Installation

```bash
git clone https://github.com/Durian-leader/oect_transfer_analyse.git
cd oect-transfer-analysis
pip install -e .[dev]
```

## 🔧 Quick Start

### Basic Usage

```python
from oect_transfer_analysis import DataLoader, TimeSeriesAnalyzer, Visualizer

# 1. Load transfer curve data
loader = DataLoader("path/to/csv/files")
transfer_objects = loader.load_all_files(device_type="N")

# 2. Time series analysis
analyzer = TimeSeriesAnalyzer(transfer_objects)
time_series = analyzer.extract_time_series()

# 3. Create visualizations
viz = Visualizer()

# Evolution plot with black-to-red colormap
fig, ax = viz.plot_evolution(transfer_objects, colormap="black_to_red")

# Compare specific time points
fig, ax = viz.plot_comparison(
    transfer_objects, 
    indices=[0, 25, 50],
    labels=["Initial", "Middle", "Final"]
)

# 4. Statistical analysis
stats = analyzer.get_summary_statistics()
print(stats)

# Drift detection
drift = analyzer.detect_drift("gm_max_raw", threshold=0.05)
print(f"Drift detected: {drift['drift_detected']}")
```

### Animation Generation

```python
# Generate animation (requires animation dependencies)
if viz.ANIMATION_AVAILABLE:
    viz.generate_animation(
        transfer_objects,
        "device_evolution.mp4",
        fps=30,
        dpi=150
    )
```

## 📊 Data Format

Your CSV files should contain voltage and current columns. The package automatically detects common column names:

**Voltage columns**: `vg`, `v_g`, `gate`, `vgs`, `v_gs`
**Current columns**: `id`, `i_d`, `drain`, `ids`, `i_ds`, `current`

Example CSV structure:
```csv
Vg,Id
-0.6,-1.23e-11
-0.59,-1.45e-11
...
```

## 🎨 Visualization Examples

### Evolution Plot with Custom Colormap

```python
# Black to red gradient showing time evolution
viz.plot_evolution(
    transfer_objects,
    label="Device A",
    colormap="black_to_red",
    y_scale="log",
    save_path="evolution_plot.png"
)
```

### Multi-point Comparison

```python
# Compare initial, degraded, and recovered states
viz.plot_comparison(
    transfer_objects,
    indices=[0, 50, 100],
    labels=["Fresh", "Degraded", "Recovered"],
    colormap="viridis"
)
```

### Time Series Analysis

```python
# Plot parameter evolution over time
analyzer.plot_time_series(
    parameters=['gm_max_raw', 'Von_raw', 'I_max_raw'],
    save_path="time_series.png"
)
```

## 📈 Advanced Analysis

### Stability Analysis

```python
# Comprehensive stability analysis
stability_results = analyzer.analyze_stability(threshold=0.05)
print(stability_results)

# Custom drift detection
for param in ['gm_max_raw', 'Von_raw', 'I_max_raw']:
    drift = analyzer.detect_drift(param, threshold=0.03)
    print(f"{param}: {drift['drift_direction']} by {drift['final_drift_percent']:.2f}%")
```

### Export Results

```python
# Export to pandas DataFrame
df = analyzer.to_dataframe()
df.to_csv("analysis_results.csv", index=False)

# Get statistical summary
stats = analyzer.get_summary_statistics()
stats.to_csv("statistics_summary.csv")
```

## 🎬 Animation Features

### Standard Animation

```python
from oect_transfer_analysis import generate_transfer_animation

generate_transfer_animation(
    transfer_objects,
    output_path="device_evolution.mp4",
    fps=30,
    dpi=150,
    figsize=(12, 5)
)
```

### Memory-Optimized Animation

```python
# For large datasets
generator = AnimationGenerator()
generator.generate_memory_optimized(
    transfer_objects,
    "large_dataset_evolution.mp4",
    batch_size=50
)
```

### Custom Animation Parameters

```python
generate_transfer_animation(
    transfer_objects,
    "custom_animation.mp4",
    fps=60,
    dpi=200,
    xlim=(-0.6, 0.6),
    ylim_log=(1e-12, 1e-6),
    codec='H264'
)
```

## 🔍 API Reference

### Core Classes

#### `DataLoader`
```python
DataLoader(folder_path)
```
- `load_all_files(device_type, file_pattern, sort_numerically)`: Load transfer files
- `analyze_batch()`: Get summary of loaded files
- `get_metadata()`: Get loading metadata

#### `TimeSeriesAnalyzer`
```python
TimeSeriesAnalyzer(transfer_objects)
```
- `extract_time_series()`: Extract time series data
- `detect_drift(parameter, threshold)`: Detect parameter drift
- `get_summary_statistics()`: Statistical summary
- `analyze_stability()`: Multi-parameter stability analysis

#### `Visualizer`
```python
Visualizer()
```
- `plot_evolution()`: Plot transfer curve evolution
- `plot_comparison()`: Compare curves at specific indices
- `generate_animation()`: Create evolution animation

### Utility Functions

```python
from oect_transfer_analysis import (
    load_transfer_files,
    plot_transfer_evolution,
    plot_transfer_comparison,
    check_dependencies
)
```

## 📋 Requirements

### Core Dependencies
- `oect-transfer>=0.4.2`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `matplotlib>=3.5.0`

### Optional Dependencies (for animation)
- `opencv-python>=4.5.0`
- `Pillow>=8.0.0`

## 🏗️ Architecture

The package is built on top of the `oect-transfer` library and provides:

```
oect-transfer-analysis/
├── DataLoader        # Batch file loading and validation
├── TimeSeriesAnalyzer # Parameter extraction and drift analysis  
├── Visualizer        # Advanced plotting with custom colormaps
├── AnimationGenerator # Video generation with parallel processing
└── Utilities         # Helper functions and system checks
```

## 🎯 Use Cases

- **Device Degradation Studies**: Track parameter changes over operational cycles
- **Environmental Testing**: Analyze stability under different conditions
- **Quality Control**: Automated analysis of production batches
- **Research Publications**: Generate publication-ready figures and animations
- **Real-time Monitoring**: Process data streams from measurement setups

## ⚡ Performance Tips

### For Large Datasets
- Use `generate_memory_optimized()` for animations with >1000 frames
- Set lower DPI (50-100) for faster processing
- Use batch processing for very large file sets

### For High Quality Output
- Use DPI 200-300 for publication figures
- Enable higher frame rates (60+ fps) for smooth animations
- Use 'H264' codec for better compression

### Parallel Processing
```python
# Automatically uses all CPU cores
generate_transfer_animation(transfer_objects, n_workers=None)

# Limit workers for memory constraints
generate_transfer_animation(transfer_objects, n_workers=4)
```

## 🐛 Troubleshooting

### Common Issues

**Import Error**: 
```bash
pip install oect-transfer oect-transfer-analysis
```

**Animation Dependencies Missing**:
```bash
pip install oect-transfer-analysis[animation]
```

**Memory Issues with Large Datasets**:
```python
# Use memory-optimized animation
generator.generate_memory_optimized(data, batch_size=25)
```

**Column Detection Issues**:
```python
# Specify columns explicitly
loader.load_all_files(vg_column="VGate", id_column="IDrain")
```

### Check System Status
```python
from oect_transfer_analysis import check_dependencies, print_system_info

check_dependencies()  # Check what's available
print_system_info()   # Detailed system information
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/yourusername/oect-transfer-analysis.git
cd oect-transfer-analysis
pip install -e .[dev]
```

### Code Style

We use `black` for code formatting and `flake8` for linting:

```bash
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **lidonghao** - *Lead Developer* - [lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)

## 🙏 Acknowledgments

- Built on the excellent `oect-transfer` library
- Thanks to the OECT research community for feedback and testing
- Matplotlib and OpenCV teams for visualization and video capabilities

## 📞 Support

- 📧 Email: [lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)
- 🐛 Issues: [GitHub Issues](https://github.com/Durian-leader/oect_transfer_analyse/issues)

## 🗺️ Roadmap

- [ ] Real-time data streaming support
- [ ] Interactive web dashboard
- [ ] Machine learning-based anomaly detection
- [ ] Integration with measurement equipment APIs
- [ ] Advanced statistical models for degradation prediction

---

**Keywords:** OECT, Organic Electrochemical Transistor, Transfer Curve, Time Series Analysis, Device Characterization, Python, Visualization, Animation