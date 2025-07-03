# OECT Transfer Analysis - Quick Start Guide

## 🚀 Installation

### Option 1: Basic Installation (Recommended)
```bash
pip install oect-transfer-analysis
```

### Option 2: With Animation Support
```bash
pip install oect-transfer-analysis[animation]
```

### Option 3: Development Installation
```bash
git clone https://github.com/yourusername/oect-transfer-analysis.git
cd oect-transfer-analysis
pip install -e .[dev]
```

## 📋 Prerequisites

### Required
- Python 3.8+
- CSV files with transfer curve data (Vg, Id columns)

### Optional (for animations)
- OpenCV (`opencv-python`)
- Pillow (`PIL`)

## 🔧 Basic Usage (5 minutes)

### 1. Prepare Your Data
Ensure your CSV files have voltage and current columns:
```csv
Vg,Id
-0.6,-1.23e-11
-0.59,-1.45e-11
...
```

Column names can be: `vg`, `v_g`, `gate`, `vgs` (voltage) and `id`, `i_d`, `drain`, `ids` (current)

### 2. Basic Analysis Script

```python
from oect_transfer_analysis import DataLoader, TimeSeriesAnalyzer, Visualizer

# Load data
loader = DataLoader("path/to/your/csv/files")
transfer_objects = loader.load_all_files(device_type="N")  # "N" or "P"

# Analyze
analyzer = TimeSeriesAnalyzer(transfer_objects)
time_series = analyzer.extract_time_series()

# Visualize
viz = Visualizer()
viz.plot_evolution(transfer_objects, colormap="black_to_red")
viz.plot_comparison(transfer_objects, indices=[0, 10, 20])

# Check stability
drift = analyzer.detect_drift("gm_max_raw", threshold=0.05)
print(f"Drift detected: {drift['drift_detected']}")
```

### 3. Generate Animation (if dependencies installed)

```python
# Simple animation
viz.generate_animation(transfer_objects, "evolution.mp4", fps=30)
```

## 📊 Quick Data Check

### Check Your Installation
```python
from oect_transfer_analysis import check_dependencies
check_dependencies()
```

### Test with Example Data
```python
from oect_transfer_analysis import create_example_data, TimeSeriesAnalyzer

# Create test data
transfer_objects = create_example_data(n_points=100, n_files=20)

# Quick analysis
analyzer = TimeSeriesAnalyzer(transfer_objects)
stats = analyzer.get_summary_statistics()
print(stats.head())
```

## 🎯 Common Use Cases

### Case 1: Device Degradation Study
```python
# Load sequential measurements
transfer_objects = loader.load_all_files(sort_numerically=True)

# Analyze degradation
stability = analyzer.analyze_stability(threshold=0.03)
print(stability)

# Visualize evolution
viz.plot_evolution(transfer_objects, label="Degradation Study")
```

### Case 2: Multi-Device Comparison
```python
# Compare initial vs final states
n_devices = len(transfer_objects)
indices = [0, n_devices//2, n_devices-1]
labels = ["Initial", "Mid-life", "End-of-life"]

viz.plot_comparison(transfer_objects, indices, labels)
```

### Case 3: Publication Figure
```python
# High-quality evolution plot
viz.plot_evolution(
    transfer_objects,
    colormap="black_to_red",
    figsize=(12, 8),
    dpi=300,
    save_path="publication_figure.png"
)
```

## 🔍 Troubleshooting

### Problem: "No CSV files found"
- Check file pattern: `loader.load_all_files(file_pattern="your_pattern")`
- Verify folder path exists

### Problem: "Could not find Vg or Id columns"  
- Specify columns: `loader.load_all_files(vg_column="VGate", id_column="IDrain")`
- Check column names in your CSV

### Problem: Animation fails
- Install dependencies: `pip install oect-transfer-analysis[animation]`
- Check: `check_dependencies()`

### Problem: Memory issues with large datasets
- Use memory-optimized animation: `generator.generate_memory_optimized()`
- Process in smaller batches

## 📁 File Structure

Organize your data like this:
```
your_project/
├── data/
│   ├── transfer_001.csv
│   ├── transfer_002.csv
│   └── ...
├── analysis.py
└── results/
    ├── plots/
    └── animations/
```

## 🎨 Customization Examples

### Custom Colors
```python
# Different colormaps
viz.plot_evolution(transfer_objects, colormap="viridis")
viz.plot_evolution(transfer_objects, colormap="plasma")
viz.plot_evolution(transfer_objects, colormap="Reds")
```

### Custom Analysis
```python
# Specific parameters
analyzer.plot_time_series(['gm_max_raw', 'Von_raw'])

# Custom thresholds
drift = analyzer.detect_drift("I_max_raw", threshold=0.02)  # 2%
```

### Export Results
```python
# Save to CSV
df = analyzer.to_dataframe()
df.to_csv("analysis_results.csv")

# Statistics
stats = analyzer.get_summary_statistics()
stats.to_csv("statistics.csv")
```

## 📖 Next Steps

1. **Run examples**: Check `examples/basic_usage.py`
2. **Read full documentation**: [Link to docs]
3. **Advanced features**: See `examples/advanced_analysis.py`
4. **Customize**: Modify parameters for your specific needs

## 🆘 Getting Help

- **Documentation**: [Read the Docs link]
- **Issues**: [GitHub Issues](https://github.com/yourusername/oect-transfer-analysis/issues)
- **Email**: lidonghao100@outlook.com

## ⚡ Performance Tips

- **Large datasets**: Use `generate_memory_optimized()` for >100 files
- **Speed**: Lower DPI (50-100) for faster processing
- **Quality**: Higher DPI (200-300) for publications
- **Memory**: Process in batches if memory is limited

---

**Ready to analyze your OECT data? Start with the basic script above! 🚀**