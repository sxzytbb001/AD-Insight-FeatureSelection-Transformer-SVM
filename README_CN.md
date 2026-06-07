# 基因表达矩阵二分类流水线

这是一个用于基因表达矩阵二分类的可复现实验流水线。当前仓库的案例研究面向阿尔茨海默病 AD 的病例/对照队列，但代码接口是通用的：只要提供表达矩阵和样本标签，就可以完成特征选择、Transformer 训练、SVM 基线、外部验证和统计比较。

## 核心功能

- 自动识别表达矩阵方向：基因为行或样本为行均可。
- 支持常见二分类标签归一化，例如 `control`、`normal`、`positive`、`AD`。
- 使用 7 类方法做集成特征选择，输出候选基因面板。
- 训练轻量级 `TransformerV3`，输出注意力图、gate 权重和基因交互矩阵。
- 训练单 SVM、Voting SVM、Bagging SVM 作为强基线。
- 支持外部队列验证，并可在配置文件中显式声明标签方向翻转。
- 提供 nested internal validation、leave-one-cohort-out validation 和统计检验。

## 目录结构

```text
.
|-- apps/                         # 主流程编排和可视化工具
|-- data/                         # 案例数据矩阵和数据集清单
|-- docs/                         # 复现说明、数据划分和结果快照
|-- results/                      # 可选的运行结果快照
|-- scripts/
|   |-- analysis/                 # 统计分析
|   |-- data/                     # GEO 数据准备工具
|   |-- evaluation/               # 外部验证、嵌套验证、LOCO 验证
|   |-- preprocessing/            # 数据预处理和特征选择
|   `-- training/                 # Transformer 与 SVM 训练
|-- tests/                        # 轻量回归测试
|-- main.py                       # 兼容入口
|-- config.py                     # 路径、标签和产物配置
|-- requirements.txt              # 运行依赖
`-- requirements-dev.txt          # 测试依赖
```

## 安装

建议使用 Python 3.10 或 3.11。完整训练建议使用支持 CUDA 的 PyTorch 环境；轻量测试和多数数据工具可在 CPU 上运行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

如果使用仓库中通过 Git LFS 管理的数据矩阵、模型权重或结果产物，请先安装并拉取 LFS 文件：

```bash
git lfs install
git lfs pull
```

## 数据格式

默认训练数据路径：

```text
data/train/cleaned_gene_matrix.csv
data/train/sample_labels.csv
```

表达矩阵可以是：

- 行为基因、列为样本；
- 或行为样本、列为基因。

标签表至少包含：

```csv
sample_id,label
sample_001,control
sample_002,positive
```

如果你的标签名称不同，可以设置：

```powershell
$env:GENE_EXPR_POSITIVE_LABEL="tumor"
$env:GENE_EXPR_NEGATIVE_LABEL="normal"
```

## 外部验证配置

外部队列通过根目录的 `external_datasets.json` 配置：

```json
{
  "cohort_a": {
    "path": "data/external/cohort_a"
  },
  "cohort_b": {
    "path": "D:/datasets/cohort_b",
    "label_flip": true,
    "label_flip_reason": "Use only after confirming reversed label polarity."
  }
}
```

只有在已经确认某个外部队列的标签方向与本项目的 `positive/control` 定义相反时，才应使用 `label_flip: true`。主流程会在计算指标前翻转标签，并在结果 CSV 中写入 `ConfiguredLabelFlip` 和 `LabelPolarity`。

当前案例中，`GSE109887` 已在 `external_datasets.json` 中显式配置为标签翻转，因为其 s1/s2 标签方向已确认与本项目约定相反。

## 运行

预处理数据：

```bash
python -m scripts.preprocessing.preprocess --matrix raw_matrix.csv --labels labels.csv
```

运行完整流水线：

```bash
python main.py
```

分阶段运行：

```bash
python -m scripts.preprocessing.feature_selection
python -m scripts.training.train_transformer
python -m scripts.training.train_svm
python -m scripts.evaluation.external_validation
python -m scripts.analysis.statistical_analysis
```

严格验证：

```bash
python -m scripts.evaluation.nested_internal_validation
python -m scripts.evaluation.loco_validation
```

## 结果与复现

主要输出写入 `results/`：

```text
results/
|-- feature_selection/
|-- transformer/
|-- svm/
|-- external_validation/
|-- nested_internal_validation/
|-- loco_validation/
`-- statistics/
```

当前 AD 案例的结果快照见 [docs/latest_results.md](docs/latest_results.md)。推荐结论口径是：轻量 Transformer 在小样本转录组 AD 分类中达到与强 SVM 基线相近的水平，并提供额外的解释性输出；外部验证受平台差异和队列注释质量影响，不应声称 Transformer 在所有外部数据集上都稳定优于 SVM。

## 测试

```bash
python -m pytest -q
```

GitHub Actions 配置位于 [.github/workflows/tests.yml](.github/workflows/tests.yml)，会在 Python 3.10 和 3.11 上运行轻量测试。

## 开源注意事项

- 发布真实矩阵、模型权重或结果快照前，确认数据源许可证和隐私规则允许再分发。
- 大文件应通过 Git LFS 或 GitHub Releases 发布。
- 不要把本地路径、未公开临床信息、论文草稿或临时日志提交到仓库。
- 贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 引用信息见 [CITATION.cff](CITATION.cff)。

## 许可证

本项目使用 [MIT License](LICENSE)。
