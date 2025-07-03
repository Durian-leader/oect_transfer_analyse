以下是该英文文档的完整中文翻译：

---

# OECT 转移特性分析工具

[![Python版本](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![版本](https://img.shields.io/badge/version-1.0.0-orange.svg)](https://github.com/yourusername/oect-transfer-analysis)

用于有机电化学晶体管（OECT）转移曲线的高级分析工具，提供全面的时间序列分析、可视化和动画生成能力。

---

## 🚀 核心功能

* **📁 批量数据加载**：自动加载并处理多个转移曲线 CSV 文件
* **📈 时间序列分析**：提取参数随时间变化的趋势，并检测漂移
* **🎨 高级可视化**：生成可用于发表的图像，支持自定义配色方案
* **🎬 动画生成**：制作设备演化过程的高质量视频
* **📊 统计分析**：提供详尽的统计摘要与稳定性分析
* **⚡ 性能优化**：支持并行处理以加速动画生成

---

## 📦 安装方法

### 基础安装

```bash
pip install oect-transfer-analysis
```

### 含动画支持

```bash
pip install oect-transfer-analysis[animation]
```

### 开发安装

```bash
git clone https://github.com/Durian-leader/oect_transfer_analyse.git
cd oect-transfer-analysis
pip install -e .[dev]
```

---

## 🔧 快速开始

### 基本用法

```python
from oect_transfer_analysis import DataLoader, TimeSeriesAnalyzer, Visualizer

# 1. 加载转移曲线数据
loader = DataLoader("path/to/csv/files")
transfer_objects = loader.load_all_files(device_type="N")

# 2. 时间序列分析
analyzer = TimeSeriesAnalyzer(transfer_objects)
time_series = analyzer.extract_time_series()

# 3. 创建可视化图像
viz = Visualizer()

# 使用黑到红的渐变色绘制演化图
fig, ax = viz.plot_evolution(transfer_objects, colormap="black_to_red")

# 对特定时间点进行对比
fig, ax = viz.plot_comparison(
    transfer_objects, 
    indices=[0, 25, 50],
    labels=["初始", "中期", "最终"]
)

# 4. 统计分析
stats = analyzer.get_summary_statistics()
print(stats)

# 漂移检测
drift = analyzer.detect_drift("gm_max_raw", threshold=0.05)
print(f"检测到漂移: {drift['drift_detected']}")
```

### 动画生成

```python
# 生成动画（需要安装动画相关依赖）
if viz.ANIMATION_AVAILABLE:
    viz.generate_animation(
        transfer_objects,
        "device_evolution.mp4",
        fps=30,
        dpi=150
    )
```

---

## 📊 数据格式

CSV 文件应包含电压和电流两列，工具会自动识别常见列名：

* **电压列**：`vg`, `v_g`, `gate`, `vgs`, `v_gs`
* **电流列**：`id`, `i_d`, `drain`, `ids`, `i_ds`, `current`

示例 CSV 格式：

```csv
Vg,Id
-0.6,-1.23e-11
-0.59,-1.45e-11
...
```

---

## 🎨 可视化示例

### 演化图

```python
viz.plot_evolution(
    transfer_objects,
    label="器件 A",
    colormap="black_to_red",
    y_scale="log",
    save_path="evolution_plot.png"
)
```

### 多点对比

```python
viz.plot_comparison(
    transfer_objects,
    indices=[0, 50, 100],
    labels=["初始", "退化", "恢复"],
    colormap="viridis"
)
```

### 参数随时间变化图

```python
analyzer.plot_time_series(
    parameters=['gm_max_raw', 'Von_raw', 'I_max_raw'],
    save_path="time_series.png"
)
```

---

## 📈 高级分析

### 稳定性分析

```python
stability_results = analyzer.analyze_stability(threshold=0.05)
print(stability_results)

# 自定义参数漂移检测
for param in ['gm_max_raw', 'Von_raw', 'I_max_raw']:
    drift = analyzer.detect_drift(param, threshold=0.03)
    print(f"{param}: {drift['drift_direction']}，变化 {drift['final_drift_percent']:.2f}%")
```

### 导出结果

```python
df = analyzer.to_dataframe()
df.to_csv("analysis_results.csv", index=False)

stats = analyzer.get_summary_statistics()
stats.to_csv("statistics_summary.csv")
```

---

## 🎬 动画功能

### 标准动画

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

### 内存优化动画

```python
generator = AnimationGenerator()
generator.generate_memory_optimized(
    transfer_objects,
    "large_dataset_evolution.mp4",
    batch_size=50
)
```

### 自定义参数动画

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

---

## 🔍 API 参考

### 核心类

#### `DataLoader`

```python
DataLoader(folder_path)
```

* `load_all_files(device_type, file_pattern, sort_numerically)`：加载文件
* `analyze_batch()`：分析批量数据
* `get_metadata()`：获取加载元信息

#### `TimeSeriesAnalyzer`

```python
TimeSeriesAnalyzer(transfer_objects)
```

* `extract_time_series()`：提取时间序列
* `detect_drift(parameter, threshold)`：检测漂移
* `get_summary_statistics()`：统计摘要
* `analyze_stability()`：稳定性分析

#### `Visualizer`

```python
Visualizer()
```

* `plot_evolution()`：绘制演化图
* `plot_comparison()`：绘制对比图
* `generate_animation()`：生成动画

### 实用函数

```python
from oect_transfer_analysis import (
    load_transfer_files,
    plot_transfer_evolution,
    plot_transfer_comparison,
    check_dependencies
)
```

---

## 📋 系统要求

### 必要依赖

* `oect-transfer>=0.4.2`
* `numpy>=1.20.0`
* `pandas>=1.3.0`
* `matplotlib>=3.5.0`

### 可选依赖（用于动画）

* `opencv-python>=4.5.0`
* `Pillow>=8.0.0`

---

## 🏗️ 架构概览

该工具构建于 `oect-transfer` 库之上：

```
oect-transfer-analysis/
├── DataLoader          # 批量加载与验证
├── TimeSeriesAnalyzer  # 参数提取与漂移分析
├── Visualizer          # 高级可视化
├── AnimationGenerator  # 并行处理的视频生成
└── Utilities           # 辅助函数与系统检查
```

---

## 🎯 应用场景

* **器件老化研究**：跟踪参数随循环次数变化
* **环境稳定性测试**：在不同环境下分析稳定性
* **质量控制**：自动分析生产批次数据
* **科研发表**：生成可发表图表与动画
* **实时监测**：处理测试系统的数据流

---

## ⚡ 性能优化建议

### 针对大数据集

* 使用 `generate_memory_optimized()` 生成动画
* 将 DPI 设置为 50–100 加速渲染
* 使用批处理分析大量文件

### 高质量输出

* 使用 DPI 200–300 生成发表图像
* 设置高帧率（60+ fps）获得流畅动画
* 使用 'H264' 编码提高压缩效率

### 并行处理

```python
# 使用所有 CPU 核心
generate_transfer_animation(transfer_objects, n_workers=None)

# 限制并行数以节省内存
generate_transfer_animation(transfer_objects, n_workers=4)
```

---

## 🐛 故障排查

### 常见问题

**导入错误**：

```bash
pip install oect-transfer oect-transfer-analysis
```

**缺少动画依赖**：

```bash
pip install oect-transfer-analysis[animation]
```

**大数据内存问题**：

```python
generator.generate_memory_optimized(data, batch_size=25)
```

**列名识别失败**：

```python
loader.load_all_files(vg_column="VGate", id_column="IDrain")
```

**系统状态检查**：

```python
from oect_transfer_analysis import check_dependencies, print_system_info

check_dependencies()
print_system_info()
```

---

## 🤝 贡献指南

欢迎贡献！请阅读 [贡献指南](CONTRIBUTING.md)。

### 开发环境设置

```bash
git clone https://github.com/yourusername/oect-transfer-analysis.git
cd oect-transfer-analysis
pip install -e .[dev]
```

### 代码风格

使用 `black` 格式化代码，使用 `flake8` 检查风格：

```bash
black src/
flake8 src/
```

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 👥 作者

* **李东昊** - *项目负责人* - [lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)

---

## 🙏 鸣谢

* 构建于优秀的 `oect-transfer` 库之上
* 感谢 OECT 研究社区提供反馈与测试
* 感谢 Matplotlib 与 OpenCV 团队提供可视化与视频支持

---

## 📞 支持与联系

* 📧 邮箱：[lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)
* 🐛 问题反馈：[GitHub Issues](https://github.com/Durian-leader/oect_transfer_analyse/issues)

---

## 🗺️ 项目路线图

* [ ] 实时数据流支持
* [ ] 交互式 Web 可视化仪表板
* [ ] 基于机器学习的异常检测
* [ ] 与仪器设备 API 集成
* [ ] 更高级的退化预测统计模型

---

**关键词**：OECT、有机电化学晶体管、转移曲线、时间序列分析、器件表征、Python、可视化、动画生成
