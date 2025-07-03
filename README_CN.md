以下是《OECT Transfer Curve Advanced Analysis》文档的完整中文翻译：

---

# OECT 转移曲线高级分析工具

[![PyPI 版本](https://badge.fury.io/py/oect-transfer-analyse.svg)](https://badge.fury.io/py/oect-transfer-analyse)
[![Python 支持](https://img.shields.io/pypi/pyversions/oect-transfer-analyse.svg)](https://pypi.org/project/oect-transfer-analyse/)
[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个用于分析和可视化有机电化学晶体管（OECT）转移曲线的高级 Python 工具包，构建于核心包 `oect-transfer` 之上。

---

## 🎯 功能亮点

### 🔬 高级分析功能

* **时间序列分析**：提取并分析器件参数随时间演化的过程
* **漂移检测**：自动检测参数漂移和器件退化现象
* **趋势分析**：统计趋势检测，包含显著性检验
* **稳定性评估**：对器件稳定性进行全面评估
* **批量处理**：同时比较多个器件或测试条件

### 📊 增强型可视化

* **可用于发表的图像**：高质量图像，支持多种风格
* **参数趋势图**：时间序列变化趋势展示
* **漂移分析图**：直观展示漂移模式
* **对比图**：并排展示不同器件或条件的曲线
* **自定义样式**：支持发表、简约、标准风格

### 🎬 动画生成

* **转移曲线演化动画**：MP4 视频展示曲线随时间演化
* **参数变化动画**：展示参数随时间变化的视频
* **高性能生成**：支持并行处理加快视频生成速度
* **多种格式**：支持不同编码格式和视频质量
* **批量动画**：支持多数据集批量动画生成

### 🔄 预设工作流

* **完整分析流程**：一键执行完整分析
* **快速稳定性筛查**：快速评估器件稳定性
* **批量对比分析**：多个器件的统一分析流程
* **自定义流程**：灵活拼装分析模块构建自定义工作流

---

## 📦 安装方式

### 基础安装（核心功能 + 分析功能）

```bash
pip install oect-transfer-analyse
```

### 加上绘图支持

```bash
pip install oect-transfer-analyse[plotting]
```

### 加上动画支持

```bash
pip install oect-transfer-analyse[animation]
```

### 完整安装（所有功能）

```bash
pip install oect-transfer-analyse[all]
```

### 开发者安装

```bash
git clone https://github.com/yourusername/oect-transfer-analyse.git
cd oect-transfer-analyse
pip install -e .[all,dev]
```

---

## 🚀 快速开始

### 完整分析流程示例

```python
import oect_transfer_analyse as ota

# 执行完整分析流程
results = ota.complete_analysis_workflow(
    data_folder='device_data/',
    device_type='N',
    device_label='Sample_A',
    output_dir='Sample_A_analysis/',
    create_plots=True,
    create_animations=True
)

# 输出结果
print(f"器件是否稳定: {results['stability_summary']['overall_stable']}")
print(f"生成的文件数量: {len(results['files'])}")
```

### 时间序列分析

```python
# 加载数据
transfer_objects = ota.load_all_transfer_files('data/', 'N')

# 稳定性分析
extractor = ota.analyze_transfer_stability(transfer_objects)

# 参数趋势检测
trends = ota.detect_parameter_trends(extractor)
print(trends)

# 生成 HTML 报告
ota.generate_stability_report(extractor, 'stability_report.html')
```

### 高质量可视化

```python
# 生成曲线演化图
ota.plot_transfer_evolution(
    transfer_objects, 
    style='publication',
    colormap='viridis',
    save_path='figure_1.png'
)

# 生成参数趋势图
ota.plot_parameter_trends(
    extractor,
    parameters=['gm_max_raw', 'Von_raw'],
    style='publication'
)

# 批量生成图集
ota.create_publication_plots(
    transfer_objects,
    output_dir='publication_figures/',
    device_label='Device_A'
)
```

### 动画生成

```python
# 检查动画依赖是否可用
if ota.check_animation_available():
    ota.generate_transfer_animation(
        transfer_objects,
        output_path='device_evolution.mp4',
        style='publication',
        fps=30,
        dpi=150
    )

    ota.create_animation_preview(
        transfer_objects,
        indices=[0, 10, 20, 30, 40],
        output_path='evolution_preview.png'
    )
else:
    print("请先安装动画依赖: pip install oect-transfer-analyse[animation]")
```

### 快速筛查器件

```python
# 快速稳定性检查
status = ota.quick_stability_check(transfer_objects)
print(f"稳定性状态: {status}")  # 'STABLE', 'MODERATE_DRIFT', 'SIGNIFICANT_DRIFT'

# 批量对比分析
results = ota.batch_comparison_workflow(
    data_folders=['device_A/', 'device_B/', 'device_C/'],
    device_labels=['Device A', 'Device B', 'Device C'],
    output_dir='batch_analysis/'
)
```

---

## 📋 核心函数

### `complete_analysis_workflow()`

端到端完整分析，包括：

* 数据加载与验证
* 时间序列提取与漂移检测
* 可视化图表生成
* 动画制作
* HTML 报告输出

### `quick_stability_check()`

用于快速筛查器件稳定性，适用于：

* 工业生产质检
* 初期性能评估
* 批量数据分析

### `batch_comparison_workflow()`

用于比较多个器件在不同工艺或条件下的表现：

---

## 🎨 绘图样式

### 发布风格（publication）

* 面向论文投稿优化的图表
* 专业排版与色彩
* 适用于数字与印刷出版

### 极简风格（minimal）

* 简洁干净
* 无干扰视觉元素
* 适合演示文稿

### 标准风格（standard）

* 平衡美观与可读性
* 适合报告或说明文档

---

## 🔧 软件结构

此工具包构建在 `oect-transfer` 核心包之上：

```
oect-transfer-analyse/
├── analysis.py        # 时间序列分析与漂移检测
├── plotting.py        # 增强可视化模块
├── animation.py       # 视频生成模块
├── workflows.py       # 预定义分析工作流
└── __init__.py        # 包接口与依赖管理
```

### 依赖管理

* **核心功能**：仅依赖 `oect-transfer`, `numpy`, `pandas`
* **绘图功能**：可选依赖 `matplotlib`
* **动画功能**：可选依赖 `opencv-python`
* **降级支持**：缺少依赖时提示清晰的错误信息

---

## 📊 分析能力详解

### 可提取的参数（支持正/反向扫）

* `gm_max_raw/forward/reverse`: 最大跨导
* `I_max_raw/forward/reverse`: 最大电流
* `I_min_raw/forward/reverse`: 最小电流
* `Von_raw/forward/reverse`: 阈值电压
* `absgm_max_raw/forward/reverse`: 跨导绝对值最大值
* `absI_max_raw/forward/reverse`: 电流绝对值最大值
* `absI_min_raw/forward/reverse`: 电流绝对值最小值

### 漂移检测方法

* **相对漂移**：相较初始值的百分比变化
* **绝对漂移**：数值上的绝对变化
* **滑动窗口分析**：局部漂移分析
* **趋势分析**：基于线性回归的趋势检测

### 稳定性指标

* **变异系数**：评估参数波动性
* **稳定性评分**：整体稳定性量化
* **漂移事件**：识别关键漂移节点
* **趋势强度**：趋势的统计显著性分析

---

## 🎯 应用场景

### 学术研究

* 器件稳定性研究
* 参数演化分析
* 支持发表的图表生成
* 多器件比较分析

### 工业应用

* 生产质控
* 器件可靠性评估
* 批次一致性分析
* 工艺优化指导

### 教育用途

* 交互式演示
* 数据分析教学
* 可视化教学示例
* 科研培训辅助

---

## 📈 性能优化

### 动画生成

* **并行处理**：多核 CPU 高效利用
* **内存优化**：可处理大体积数据
* **多种编码器支持**：适配不同平台
* **速度**：比串行处理快 3-5 倍

### 分析效率

* **向量化计算**：基于 NumPy 优化性能
* **高效算法**：适配大规模时间序列
* **缓存机制**：智能缓存避免重复计算

---

## 🤝 系统集成支持

### 与核心包联用

```python
import oect_transfer as ot
import oect_transfer_analyse as ota

transfer_objects = ot.load_all_transfer_files('data/', 'N')
results = ota.complete_analysis_workflow(transfer_objects)
```

### 与 Jupyter 配合

* 适用于交互式分析与教学演示

### 与自动化流程集成

* 可嵌入批处理与自动化测试系统中使用

---

## 📚 文档资源

* **API 参考**：详细函数文档和使用示例
* **教学教程**：逐步讲解典型工作流
* **案例演示**：真实分析案例
* **最佳实践**：OECT 分析的推荐方法

---

## 🧪 测试方法

```bash
# 运行所有测试
pytest

# 带覆盖率的测试
pytest --cov=oect_transfer_analyse

# 运行特定模块测试
pytest tests/test_analysis.py
pytest tests/test_plotting.py
pytest tests/test_animation.py
```

---

## 🤝 贡献指南

欢迎贡献！请参见我们的[贡献指南](CONTRIBUTING.md)。

### 开发环境配置

```bash
git clone https://github.com/yourusername/oect-transfer-analyse.git
cd oect-transfer-analyse
pip install -e .[all,dev]
pre-commit install
```

---

## 📄 开源协议

本项目采用 MIT 协议开源，详情见 [LICENSE](LICENSE)。

---

## 🙏 鸣谢

* 基于 `oect-transfer` 核心包构建
* 感谢 OECT 研究社区的启发与支持
* 感谢所有的贡献者与用户

---

## 📞 支持

* **文档网站**：[Read the Docs](https://oect-transfer-analyse.readthedocs.io/)
* **问题反馈**：[GitHub Issues](https://github.com/yourusername/oect-transfer-analyse/issues)
* **讨论区**：[GitHub Discussions](https://github.com/yourusername/oect-transfer-analyse/discussions)

---

## 🔗 相关项目

* [`oect-transfer`](https://github.com/yourusername/oect-transfer)：核心转移曲线分析包
* [`oect-data`](https://github.com/yourusername/oect-data)：OECT 数据集收集
* [`oect-modeling`](https://github.com/yourusername/oect-modeling)：OECT 模型工具包

---

*本项目致力于服务 OECT 研究社区 ❤️*
