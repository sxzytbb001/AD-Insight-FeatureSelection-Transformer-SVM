# Transformer 与 SVM 统计学分析

## 结果汇总

| 数据集 | 样本量 | Transformer AUC（95% CI） | SVM AUC（95% CI） | AUC差值（Transformer-SVM） | DeLong P值 | Transformer 准确率（95% CI） | SVM 准确率（95% CI） | 准确率差值（Transformer-SVM） | McNemar P值 |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| Internal OOF | 739 | 0.9322 (0.9131, 0.9491) | 0.9256 (0.9059, 0.9427) | 0.0066 | 0.1391 | 0.8755 (0.8525, 0.8985) | 0.8498 (0.8227, 0.8742) | 0.0257 | 0.0351 |
| GSE109887 | 78 | 0.8770 (0.7826, 0.9506) | 0.8601 (0.7636, 0.9368) | 0.0170 | 0.2347 | 0.7949 (0.7051, 0.8721) | 0.7692 (0.6667, 0.8590) | 0.0256 | 0.6875 |
| GSE118553 | 267 | 0.6914 (0.6222, 0.7558) | 0.6778 (0.6123, 0.7371) | 0.0137 | 0.5193 | 0.6854 (0.6292, 0.7379) | 0.6554 (0.5992, 0.7116) | 0.0300 | 0.3662 |
| GSE122063 | 100 | 0.8482 (0.7644, 0.9140) | 0.8369 (0.7551, 0.9095) | 0.0114 | 0.4256 | 0.6900 (0.6000, 0.7800) | 0.7300 (0.6400, 0.8100) | -0.0400 | 0.1250 |
| GSE48350 | 220 | 0.6691 (0.5954, 0.7384) | 0.6824 (0.6071, 0.7509) | -0.0133 | 0.3061 | 0.5636 (0.4955, 0.6273) | 0.5636 (0.5000, 0.6273) | 0.0000 | 1.0000 |

## 判别分歧与阈值

| 数据集 | 仅 Transformer 判对样本数 | 仅 SVM 判对样本数 | Transformer 阈值 | SVM 阈值 |
| --- | ---: | ---: | ---: | ---: |
| Internal OOF | 46 | 27 | 0.2419 | 0.6621 |
| GSE109887 | 4 | 2 | 0.2724 | 0.3502 |
| GSE118553 | 34 | 26 | 0.3874 | 0.4577 |
| GSE122063 | 0 | 4 | 0.1576 | 0.3000 |
| GSE48350 | 8 | 8 | 0.5975 | 0.6268 |

## 简要结论

- Internal OOF: Transformer AUC=0.9322, SVM AUC=0.9256, DeLong P=0.1391; Transformer Accuracy=0.8755, SVM Accuracy=0.8498, McNemar P=0.0351.
- GSE109887: Transformer AUC=0.8770, SVM AUC=0.8601, DeLong P=0.2347; Transformer Accuracy=0.7949, SVM Accuracy=0.7692, McNemar P=0.6875.
- GSE118553: Transformer AUC=0.6914, SVM AUC=0.6778, DeLong P=0.5193; Transformer Accuracy=0.6854, SVM Accuracy=0.6554, McNemar P=0.3662.
- GSE122063: Transformer AUC=0.8482, SVM AUC=0.8369, DeLong P=0.4256; Transformer Accuracy=0.6900, SVM Accuracy=0.7300, McNemar P=0.1250.
- GSE48350: Transformer AUC=0.6691, SVM AUC=0.6824, DeLong P=0.3061; Transformer Accuracy=0.5636, SVM Accuracy=0.5636, McNemar P=1.0000.
