# AD-Insight：阿尔茨海默病基因表达诊断研究

AD-Insight 是一个面向阿尔茨海默病 AD 诊断研究的可复现机器学习流水线，基于基因表达矩阵、集成特征选择、Transformer 建模与 SVM 基线对比。代码接口也适用于其他二分类转录组数据集，覆盖外部队列验证和统计比较。

[English](README_EN.md)

## 授权与使用限制

本项目为公开源码的研究学习项目，按照 [PolyForm Noncommercial License 1.0.0](LICENSE) 提供，仅供学习、复现和非商业评估使用。

未经作者或维护者书面许可，不得将本项目代码、模型产物、候选基因列表、图表、表格、实验输出或直接派生结论用于论文发表、学位论文、预印本、竞赛提交、奖项申报、专利、商业产品、公开 benchmark 或其他公开研究成果，并不得将其声称为独立原创成果。

使用本项目时必须保留作者署名、版权声明、[NOTICE](NOTICE)、[LICENSE](LICENSE) 和 [CITATION.cff](CITATION.cff)。如需基于本项目开展论文、竞赛、商业应用或扩展研究，请先联系作者或维护者沟通合作授权。详细规则见 [Publication and Competition Policy](PUBLICATION_AND_COMPETITION_POLICY.md)。

## 核心功能

- 自动识别表达矩阵方向：基因为行或样本为行均可。
- 支持常见二分类标签归一化，例如 `control`、`normal`、`positive`、`AD`。
- 使用 Welch t-test、Mutual Information、XGBoost、Random Forest、ElasticNet、mRMR、Stability Selection 做集成特征选择。
- 训练轻量级 `TransformerV3`，输出注意力图、gate 权重和基因交互矩阵。
- 训练单 SVM、Voting SVM、Bagging SVM 作为强基线。
- 支持外部队列验证，并可在配置文件中显式声明标签方向翻转。
- 提供 nested internal validation、leave-one-cohort-out validation 和统计检验。

## 方法流程

- **数据准备**：GEO 表达矩阵通过平台注释映射到基因符号，重复基因按均值聚合；表达值在高动态范围时转换为 `log2(x + 1)`，随后按基因在单个数据集内做 z-score 标准化。
- **训练集构建**：训练队列按公共基因交集合并，样本 ID 增加数据集前缀，并在标签表中保留 `dataset` 和 `source_sample_id` 元数据。
- **矩阵读取**：训练和验证阶段根据 `sample_id` 自动判断矩阵方向，统一转换为“样本 x 基因”的建模矩阵；非数值表达值转换为缺失后填充为 `0.0`。
- **标签处理**：常见 AD / control 标签归一化为二分类标签，`positive` 为 1，`control` 为 0。
- **特征选择**：先通过 Welch t-test、FDR、效应量等统计量预筛选基因，再结合 Mutual Information、XGBoost gain、Random Forest importance、ElasticNet Logistic Regression、mRMR 和 Stability Selection 的排名，以验证 AUC 加权投票形成 30 个候选基因。
- **Transformer 建模**：候选基因表达经过 rank-gauss 标准化后输入轻量级 `TransformerV3`。模型融合 CLS token、gene gate pooling、原始线性投影和二阶 interaction factor，并使用类别权重、label smoothing、辅助 focal loss、数据增强、多随机种子 OOF 集成训练。
- **SVM 基线**：使用网格搜索选择 SVM 参数，并训练单 SVM、Voting SVM 和 Bagging SVM。
- **验证与统计**：内部 OOF、nested internal validation、leave-one-cohort-out validation 和外部队列验证共同输出 AUC、Accuracy、混淆矩阵、ROC、DeLong 检验、McNemar 检验和 bootstrap 置信区间。

## 目录结构

```text
.
|-- apps/                         # 主流程编排、可视化和分阶段命令模块
|   |-- main.py                   # 流水线命令入口
|   |-- pipeline.py               # 主流程编排
|   |-- config.py                 # 路径、标签和产物配置
|   |-- common.py                 # 数据读取、标签处理和通用建模工具
|   |-- plot_style.py             # 绘图主题和通用图表样式
|   |-- visualization.py          # 可视化工具
|   |-- analysis/                 # 统计分析
|   |-- data/                     # GEO 数据准备工具
|   |-- evaluation/               # 外部验证、嵌套验证、LOCO 验证
|   |-- preprocessing/            # 数据预处理和特征选择
|   `-- training/                 # Transformer 与 SVM 训练
|-- data/                         # 案例数据矩阵和数据集清单
|-- docs/                         # 复现说明、数据划分和结果快照
|-- results/                      # 可选的运行结果快照
|-- tests/                        # 轻量回归测试
|-- requirements.txt              # 运行依赖
`-- requirements-dev.txt          # 测试依赖
```

## 安装

支持 Python 3.10 或 3.11。完整 Transformer 训练可使用 CUDA 版 PyTorch 环境；轻量测试和多数数据工具可在 CPU 上运行。

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

仓库中的大体积数据矩阵、模型权重或结果产物可通过 Git LFS 管理：

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

标签名称可通过环境变量配置：

```powershell
$env:GENE_EXPR_POSITIVE_LABEL="tumor"
$env:GENE_EXPR_NEGATIVE_LABEL="normal"
```

## 案例数据集

当前 AD 案例使用 GEO 公开数据集构建训练集和外部验证集。

| 数据集 | 角色 | 平台 | 样本数 | Control | Positive | 基因数 |
|---|---|---:|---:|---:|---:|---:|
| GSE1297 | 训练 | GPL96 | 31 | 9 | 22 | 13100 |
| GSE33000 | 训练 | preprocessed | 467 | 157 | 310 | 17402 |
| GSE36980 | 训练 | GPL6244 | 80 | 47 | 33 | 20003 |
| GSE5281 | 训练 | GPL570 | 161 | 74 | 87 | 21753 |
| GSE109887 | 外部验证 | preprocessed | 78 | 46 | 32 | 31682 |
| GSE118553 | 外部验证 | GPL10558 | 267 | 100 | 167 | 20759 |
| GSE122063 | 外部验证 | preprocessed | 100 | 44 | 56 | 32074 |
| GSE48350 | 外部验证 | GPL570 | 220 | 140 | 80 | 21753 |

训练集合并后包含 739 个样本，其中 control 287 个、positive 452 个，公共基因为 9981 个。`GSE29378` 保留为 exploratory 数据集，不在默认外部验证配置中使用。

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
    "label_flip_reason": "Confirmed reversed label polarity."
  }
}
```

`label_flip: true` 表示该外部队列的标签方向与本项目的 `positive/control` 定义相反。主流程会在计算指标前翻转标签，并在结果 CSV 中写入 `ConfiguredLabelFlip` 和 `LabelPolarity`。

当前案例中，`GSE109887` 已在 `external_datasets.json` 中显式配置为标签翻转，因为其 s1/s2 标签方向已确认与本项目约定相反。

## 运行

预处理数据：

```bash
python -m apps.preprocessing.preprocess --matrix raw_matrix.csv --labels labels.csv
```

运行完整流水线：

```bash
python -m apps.main
```

分阶段运行：

```bash
python -m apps.preprocessing.feature_selection
python -m apps.training.train_transformer
python -m apps.training.train_svm
python -m apps.evaluation.external_validation
python -m apps.analysis.statistical_analysis
```

严格验证：

```bash
python -m apps.evaluation.nested_internal_validation
python -m apps.evaluation.loco_validation
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

当前 AD 案例的结果快照见 [docs/latest_results.md](docs/latest_results.md)。结果显示，轻量 Transformer 在小样本转录组 AD 分类中达到与强 SVM 基线相近的水平，并输出注意力图、gate 权重和基因交互矩阵等解释性结果。

当前候选基因面板包含 30 个基因，见 [results/feature_selection/candidate_genes.txt](results/feature_selection/candidate_genes.txt)，包括 `ITPKB`、`NRN1`、`PPP1R7`、`NEUROD6`、`GFAP`、`SST`、`CD200`、`NRXN3`、`VGF`、`PTPRN2` 等。

内部 OOF 验证结果：

| 模型 | AUC | Accuracy |
|---|---:|---:|
| Transformer | 0.9322 | 0.8769 |
| Logistic Regression | 0.9199 | 0.8444 |
| SVM | 0.9256 | 0.8498 |
| Voting SVM | 0.9316 | 0.8687 |
| Bagging SVM | 0.9269 | 0.8566 |

严格泛化验证结果：

| 协议 | 模型 | Mean AUC | Mean Accuracy |
|---|---|---:|---:|
| Nested internal validation | Transformer | 0.9250 | 0.8468 |
| Nested internal validation | SVM | 0.9188 | 0.8366 |
| Leave-one-cohort-out validation | Transformer | 0.8000 | 0.7143 |
| Leave-one-cohort-out validation | SVM | 0.7950 | 0.6756 |

外部验证主结果采用 `train_prior_quantile` 阈值策略：

| 数据集 | Transformer AUC | Transformer Accuracy | SVM AUC | SVM Accuracy |
|---|---:|---:|---:|---:|
| GSE109887 | 0.8770 | 0.7949 | 0.8601 | 0.7692 |
| GSE118553 | 0.6914 | 0.6854 | 0.6778 | 0.6554 |
| GSE122063 | 0.8482 | 0.6900 | 0.8369 | 0.7300 |
| GSE48350 | 0.6691 | 0.5636 | 0.6824 | 0.5636 |

## 测试

```bash
python -m pytest -q
```

GitHub Actions 配置位于 [.github/workflows/tests.yml](.github/workflows/tests.yml)，会在 Python 3.10 和 3.11 上运行轻量测试。

## 引用

引用信息见 [CITATION.cff](CITATION.cff)。

## 许可证

本项目按照 [PolyForm Noncommercial License 1.0.0](LICENSE) 公开源码，并受 [NOTICE](NOTICE) 与 [Publication and Competition Policy](PUBLICATION_AND_COMPETITION_POLICY.md) 约束。论文、竞赛、获奖、商业使用或直接复用研究成果前需获得作者或维护者书面许可。
