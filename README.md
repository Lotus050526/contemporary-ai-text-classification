# 基于 TF-IDF 的英文文本分类：模型比较与特征分析

本项目面向英文文本十分类任务，比较朴素贝叶斯、线性支持向量机和逻辑回归，并通过特征对照、邮件头部处理、训练曲线与错误分析，研究分类性能及其局限。

## 1. 项目概述与研究问题

项目以稀疏TF-IDF为基础，围绕以下问题开展实验：

- 相同特征与数据划分下，不同分类器及正则化参数如何影响性能？
- 词频压缩、短语特征和归一化能否改善分类？
- 分类器是否依赖邮件头部的来源信息？
- 训练损失与macro-F1是否同步变化，错分样本反映了哪些局限？

模型选择以主验证集macro-F1为依据，同时记录accuracy、训练表现、特征数与耗时。

## 2. 数据格式与划分

将两份CSV放在项目根目录，与 `main.py` 同级：

| 文件 | 字段 | 样本数 | 用途 |
|---|---|---:|---|
| `train_data.csv` | `text,target` | 7368 | 训练与验证，标签为0–9 |
| `test_data_unlabeled.csv` | `text` | 2457 | 方案确定后的预测 |

实验使用现有CSV文件，文本包含邮件或新闻组消息头。现有材料未明确标注原始数据集名称及公开下载来源，类别保留数字标签。

有标签数据按80%/20%分层划分，`stratify=target`、`random_state=42`，得到5894条训练样本和1474条验证样本。划分保存原始数据的零基索引，清洗过程保持文本与标签对齐。

词表、IDF等统计量只在训练子集拟合，验证集仅执行transform。验证集用于比较候选方案；锁定后在种子21、84的分层划分上复跑固定方案与对照，记录均值和样本标准差，不再据此调整参数。无标签测试集不参与选型，仅在最终模型重训后执行transform和predict；没有可计算的测试准确率。

## 3. 环境配置与运行方法

现有结果来自Windows 11、Python 3.14.3、PowerShell和 `.venv`，BLAS线程限制为2。实际使用的依赖版本为：

| 依赖 | 版本 | 依赖 | 版本 |
|---|---|---|---|
| numpy | 2.5.4 | pandas | 3.0.6 |
| scipy | 1.18.1 | scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 | joblib | 1.6.0 |
| threadpoolctl | 3.7.0 | pypdf | 6.20.1 |

版本固定在 [requirements.txt](requirements.txt)，原运行环境记录在 [environment.json](public_results/environment.json)。pypdf仅用于可选PDF附件记录，缺少附件或该工具不影响训练流程。数据输入只需要上述两个CSV。

在项目根目录的VSCode PowerShell终端运行；已有 `.venv` 时跳过创建步骤，无需激活环境。

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 完整实验：候选模型、特征对照、三划分评估与30轮MLP
.\.venv\Scripts\python.exe main.py experiment --epochs 30

# 按已锁定配置重训全部有标签数据，并生成最终预测
.\.venv\Scripts\python.exe main.py predict --config public_results/locked_scheme.json --output prediction.csv

# 核对完整实验生成的模型、指标与预测产物
.\.venv\Scripts\python.exe verify_results.py
```

若只需使用已锁定方案，安装环境并放置两个CSV后，可直接运行predict命令。输出 `prediction.csv` 为2457行、单列、无表头、无索引的0–9整数，保持测试CSV的原始顺序。

不同操作系统、Python或依赖版本、CPU/BLAS及浮点计算实现可能导致耗时、指标和个别预测差异。固定种子与参数有助于复现，但不保证跨版本逐条预测完全一致。完整实验可能在新环境选出不同候选；predict命令显式使用已有锁定配置。

## 4. 实验设计

基础TF-IDF设置为 `lowercase=True`、`max_features=20000`、`ngram_range=(1,1)`、`min_df=2`、`sublinear_tf=False`、`norm="l2"`。所有主要实验使用相同划分，矩阵保持稀疏形式。特征构建和分类器fit分别计时，基础候选共享一次特征构建。

分类器比较包括MultinomialNB的 `alpha=[0.1,0.5,1.0]`，以及LinearSVC、LogisticRegression的 `C=[0.1,1.0,10.0]`。LR使用lbfgs、最多2000次迭代；LinearSVC初始上限为20000，收敛警告时只增加求解预算并保留记录。

LinearSVC使用LIBLINEAR、squared_hinge损失和one-vs-rest多分类；`SVC(kernel="linear")` 使用LIBSVM、C-SVC和one-vs-one。二者都是线性SVM，但优化实现与多分类策略不同。

在最优基础线性模型的固定参数上，分别改变单一因素：次线性词频、unigram+bigram、不归一化，以及邮件头部处理。头部对照分为完整文本、保留Subject并去除来源字段、仅正文三组，分别在训练子集拟合词表与IDF。解析只处理开头到第一个空行的邮件头，失败原样保留，正文中的冒号与引用保留。

MLP使用至多5000维TF-IDF、单隐藏层 `(50,)` 和Adam，通过partial_fit逐轮训练，默认30轮、最多50轮。每轮结束后固定该轮模型，重新计算训练与验证平均交叉熵、accuracy和macro-F1；保存验证macro-F1最高轮的模型副本。概率列对应 `classes_=[0,...,9]`，总交叉熵等于平均交叉熵乘样本数。内部 `loss_` 单独记录，不与重新评估的交叉熵混同。

## 5. 主要结果

下表为主划分的验证结果；NB和LR展示各自最优参数，LinearSVC各对照固定C=10。除MLP外，表中方案均使用20000维特征。

| 模型或方案 | 验证accuracy | 验证macro-F1 |
|---|---:|---:|
| MultinomialNB，alpha=0.1 | 0.9206 | 0.9213 |
| LogisticRegression，C=10 | 0.9315 | 0.9319 |
| LinearSVC，基础特征 | 0.9335 | 0.9339 |
| LinearSVC，sublinear_tf=True | **0.9369** | **0.9371** |
| LinearSVC，unigram+bigram | 0.9247 | 0.9253 |
| LinearSVC，不归一化 | 0.9193 | 0.9193 |
| LinearSVC，保留Subject、去来源字段 | 0.9349 | 0.9352 |
| LinearSVC，仅正文 | 0.9077 | 0.9081 |
| MLP，最优第16轮 | 0.9199 | 0.9203 |

锁定方案为完整文本、unigram、sublinear_tf=True、L2、LinearSVC C=10。其主验证集错分93条，训练macro-F1约0.9998；三划分验证macro-F1为 **0.932527±0.006691**。去来源字段方案三划分均值为0.933818、标准差为0.001505，但锁定方案仍依据预先指定的主划分选择。

完整参数、训练表现和耗时见 [主实验结果表](public_results/experiments.csv)；稳健性结果见 [三划分逐组结果](public_results/three_split_results.csv) 与 [均值和标准差](public_results/robustness_summary.csv)。逐类表现见 [每类指标](public_results/per_class_metrics.csv) 和 [混淆矩阵](public_results/confusion_matrix.png)。

## 6. 结果分析与局限

次线性词频在主划分提高了macro-F1，但种子21下几乎没有改善。固定20000维预算时，加入bigram分数下降且特征构建更慢，可能与短语占用词表预算有关，不能推断短语特征普遍无效。不归一化也降低了分数；固定C时特征尺度会改变有效正则化，差异不能全部归因于归一化。

去来源字段没有降低三划分平均性能；删除整个头部则明显下降。由于后者同时删除Subject等内容，这一对照不能单独量化标题作用。正文签名与引用仍可能带来源线索，分数变化本身也不能证明数据泄露。

MLP第16轮验证交叉熵为0.2965、macro-F1为0.9203；到第30轮交叉熵进一步降至0.2674，macro-F1却降至0.9163。概率损失与离散分类指标没有同步变化，因此按macro-F1保存第16轮，而不是直接使用最后一轮。真实记录见 [逐轮指标](public_results/mlp_epochs.csv) 和 [训练曲线](public_results/mlp_curves.png)。

每类指标中，标签2的F1为0.8503，标签7为0.8797，低于多数类别。错误分析显示部分文本包含交叉主题，签名中的非核心词也会参与线性打分。类别权重和词贡献用于解释模型的关联线索，不作为因果解释或官方类别名称。

**评分敏感性分析**只依据主训练文本与验证文本的非空规范化正文重合，标记5条验证样本，并使用已保存的最佳验证预测另外评分：

| 评价范围 | 样本数 | accuracy | macro-F1 | 错分数 |
|---|---:|---:|---:|---:|
| 主要验证结果 | 1474 | 0.936906 | 0.937062 | 93 |
| 排除5条正文重合样本后的补充评分 | 1469 | 0.937372 | 0.937529 | 92 |

该分析不重训、不替换主指标，也不参与选型。全文完全重复为0，但正文重合和跨集引用说明随机文本划分中存在关联样本；排除这5条不能消除全部引用、作者或语义近重复偏差。训练标签打乱负对照的accuracy为0.1065、macro-F1为0.1061，接近随机水平，也不能单独证明完全没有泄露。三划分有样本重叠，其标准差不是独立测试的置信区间。

详细补充评分见 [评分敏感性表](public_results/scoring_sensitivity.csv)，方法与审计摘要见 [分析摘要](public_results/audit_sensitivity_summary.md)。

## 7. 项目文件说明

| 文件或目录 | 内容 |
|---|---|
| [main.py](main.py) | 数据检查、固定划分、模型比较、特征对照、MLP曲线及最终预测入口 |
| [requirements.txt](requirements.txt) | 实际使用的依赖版本 |
| [verify_results.py](verify_results.py) | 基于完整实验产物核对模型、验证指标与预测格式 |
| [leakage_audit.py](leakage_audit.py) | 固定配置重放、拟合来源核查、重复诊断与标签打乱负对照 |
| [prepare_submission.py](prepare_submission.py) | 利用保存预测计算评分敏感性，并导出结果摘要 |
| `results/` | 运行生成的配置、日志、划分索引、逐条预测、模型与错误分析 |
| [public_results/](public_results/) | 实验表、三划分结果、MLP指标与曲线、混淆矩阵、锁定配置及分析摘要 |

评分敏感性分析需要主划分和最佳验证预测；模型核验与深入审计需要相应的完整运行产物。深入审计会重新拟合固定方案及负对照，评分敏感性分析仅重新计算已有预测的指标。
