# 《当代人工智能》实验一

本项目实际使用 Windows、PowerShell、Python 3.14.3 和 `.venv` 运行。依赖的**实际验证版本**在 `requirements.txt`。不需要激活虚拟环境，直接使用它的 Python 路径即可。

**运行必须提供的课程文件只有 `train_data.csv` 和 `test_data_unlabeled.csv`。** 从老师提供的材料取得这两个文件，原样放在项目根目录，与main.py同级。前者字段顺序为text、target（本次7368条），后者只有text（本次2457条）。不要改写或公开上传原始CSV。所有命令在项目根目录的VSCode PowerShell终端运行。

`实验一要求.pdf` 仅为可选作业记录：存在时读取并记录哈希，不存在时记录“未提供要求PDF”并继续，读取失败也只记录警告。`draft_main.py`、`TUNING.md` 是可选课程材料，不属于运行必需文件。pypdf仅用于可选PDF记录，可不安装；其他依赖用于数值、稀疏矩阵、模型、绘图和保存。

`public_results/` 保存原Windows运行的真实结果。其他操作系统、Python/依赖版本、CPU/BLAS或求解实现可能产生耗时、数值和个别预测差异，固定种子不能保证跨版本逐条预测完全一致。新环境结果应单独标注，不能覆盖原公开证据来冒充同一次运行。

```powershell
# 已有 .venv 时跳过创建步骤
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 完整实验：数据检查、9 组基础模型、5 组对照、2 个额外划分、30 轮 MLP
.\.venv\Scripts\python.exe main.py experiment --epochs 30
# 使用本项目已公开的锁定方案；全部有标签数据重拟合，测试只transform/predict
.\.venv\Scripts\python.exe main.py predict --config public_results/locked_scheme.json --output prediction.csv
# 检查保存模型、训练子集词表/IDF、指标、最佳 MLP 轮和预测文件
.\.venv\Scripts\python.exe verify_results.py
# 事后评分敏感性、公开证据和github_upload整理；不训练模型
.\.venv\Scripts\python.exe prepare_submission.py
# 泄露核查：按原配置重放26组、训练标签打乱负对照、重复/引用与预测顺序检查
.\.venv\Scripts\python.exe leakage_audit.py
```

只需按锁定方案预测时，安装环境、放置两个CSV后可直接运行上述predict命令，无需重新选型。完整experiment命令复现候选比较，其他环境可能选出不同候选；predict显式使用公开配置可以固定方案，但仍不能保证逐条预测跨环境一致。这些复现命令会产生新结果；本次提交前整理没有执行训练或调参。

verify_results.py需要完整实验生成的私有模型、划分和预测。深入审计会重放模型，属于可选检查，不是训练必需步骤。已有审计目录应保留备份；--reuse-replay仅适用于保有该次审计模型的本地环境。公开上传不携带私有results或模型，只有公开汇总时不能直接运行深度审计。prepare_submission.py需要划分和最佳逐条验证预测；没有历史深入审计时会注明证据缺失，不伪造负对照结论。

最终文件为根目录 `prediction.csv`：无表头、无索引，每行一个 0–9 标签，与测试数据行序一致。`predictions.csv` 是课程示例，程序不读取它作为标签或覆盖它；`results/final_predictions.csv` 保留早先运行的最终预测。无标签测试集不能计算准确率，程序也不以其内容选择模型。预测与实验分开执行，复现无需手工修改代码。再次执行实验会覆盖同名结果并向日志追加记录；需要保留一次运行时先备份 `results/`。`draft_main.py` 仅为老师示例，直接拟合全标签数据，不属于本项目验证实验入口，不应拿其全量特征再划分验证。

划分固定为 80%/20%，分层标签，主划分种子 42，保存原始 CSV **数据记录的零基索引**。`csv_record_number` 是计入表头的记录序号，因文本包含换行，它不是文件的物理行号。所有特征都在训练子集拟合，验证集仅 transform。先确定验证 macro-F1 为选型指标；候选分数相同取预声明顺序中先出现的方案。先对基础模型调参，在最优基础线性模型的固定参数上分别改变 sublinear_tf、ngram_range、norm 或头部处理；不把多个改动组合后再搜索。最终在这些有限候选中按主验证 macro-F1 锁定方案。MLP 用于学习曲线演示，不参与最终线性方案选型。锁定后在 21、84 上复跑全部主要对照，不再调参。

基础 TF-IDF 是 lowercase=True、max_features=20000、unigram、min_df=2、sublinear_tf=False、norm=l2；其他参数及每个分类器全部实际参数在 `results/configs/`。矩阵始终保持稀疏；MLP 的隐藏层和输出概率是小型稠密数组，不把整个 TF-IDF 矩阵稠密化。BLAS 线程固定 2，避免小型实验的线程过度开销。计时区分特征 fit/transform 与分类器 fit；基础模型共享同一特征矩阵，`feature_seconds` 是共享阶段耗时，不能对 9 组重复累加。`feature_plus_fit_seconds` 表示独立构建该方案的阶段耗时。

LinearSVC 是线性 SVM，使用 LIBLINEAR、默认 squared_hinge 损失和 one-vs-rest 多分类；老师的 SVC(kernel="linear") 使用 LIBSVM、hinge 形式的 C-SVC 和 one-vs-one 多分类。两者优化实现、损失、多分类策略不同，不能视为完全相同的模型；LinearSVC 更适合本次高维稀疏文本。逻辑回归使用 lbfgs，max_iter=2000；LinearSVC 初始 max_iter=20000，保存 n_iter_ 和警告。若触发收敛警告，自动只将迭代上限增至 5 倍重拟合一次，不改变 C 或特征；表中保存首次警告、重试次数、最终收敛状态，fit_seconds 包含两次求解。TUNING.md 中关于 SVM 与逻辑回归 C 含义相反的措辞不准确：二者 C 增大一般都意味着较弱的正则化约束。

邮件头仅从第一行解析到第一个空行，接受合法字段和折行，要求 From 和 Subject 存在。失败就保留原文本，成功率和失败样例明确输出。B 组保留 Subject，按 `main.py` 的 SOURCE 集合去掉来源/身份/线程标识字段，保留 Lines、Keywords、Summary 等其他字段；C 组删去成功解析的整个头部。不会全局删除正文中的冒号行。正文中的签名和引用仍可能带来源信息，B 组也保留客户端字段，所以这是部分来源线索的敏感性分析，不是彻底匿名化。实际换行与字面量反斜杠转义分别检查，原文不会经过 unicode_escape。

MLP 5000 维 TF-IDF、单隐藏层 (50,)、Adam，用 partial_fit 每次训练一个 epoch。最多 50 轮，命令默认 30 轮；每轮结束后用固定的该轮模型重新预测完整训练/验证集，计算样本平均交叉熵、accuracy 和 macro-F1。概率列按 `model.classes_=[0,1,...,9]`，log_loss 的 labels 使用同一数组；总交叉熵=平均交叉熵×样本数。内部 loss_ 是批次更新过程中的损失并含正则项，单独保存，不能和重新评估的交叉熵混同。最优轮按验证 macro-F1 选择并深复制保存，训练仍执行全部预定轮数。不同模型不同定义的损失不作数值比较。

主要产物：

- `results/experiments.csv`：每个模型/参数/划分的训练与验证指标、耗时、特征数、密度、警告。
- `data_audit.json`、`*_escape_samples.json`、`header_samples.json`：数据检查和真实清洗样例。
- `split_42.csv`、`split_21.csv`、`split_84.csv`、`split_duplicate_overlap.json`：固定索引和跨子集重复检查。
- `locked_scheme.json`、`protocol.json`、`environment.json`、`configs/`：锁定方案、预声明指标、数据哈希与依赖环境。
- `confusion_matrix.csv/png`、`per_class_metrics.csv`、`misclassified.csv`、`representative_errors.csv`：最佳方案错误诊断。
- `class_weight_words.csv`：类别正负权重词，是模型关联线索，不能作为因果解释。
- `error_term_contributions.csv`：5 条代表性错分中推高预测类相对真实类分数的词贡献；需结合原文理解。
- `robustness_summary.csv`：3 个划分的均值、样本标准差（ddof=1），划分之间有重叠，不是独立外部测试或置信区间。
- `mlp_epochs.csv`、`mlp_curves.png`、`mlp_best_epoch.json`、`mlp_best.joblib`：逐轮真实记录和最佳模型副本。
- `run.log`、`failure.json`（若失败）、`completion.json`：运行记录；警告也保存在对应实验表里。
- `run_notes.md`、`analysis_notes.md`、`verification.json`：修正过程、真实内容分析笔记和产物复核结果。`results_initial_parser/` 保留首次运行的旧解析规则结果，正式结果只在 `results/`。
- `best_validation_predictions.csv`、`validation_predictions/`、`models/`：逐条验证预测、逐组模型及所用划分，后续运行会自动保存。内存共享特征绑定清洗文本、索引与配置，错误缓存会被拒绝。
- `leakage_audit.md`：本次泄露核查结论和代码位置；`leakage_audit_run/` 是独立重放、拟合追踪、负对照、重复/引用和预测顺序证据，旧实验表与锁定方案保留。首次执行审计创建此目录；已有成功重放时用 `leakage_audit.py --reuse-replay` 复核，不重复拟合负对照。根目录 `prediction.csv` 被 gitignore 排除。

课程数据、含文本的结果和模型默认被 `.gitignore` 排除，避免公开课程材料。本阶段不生成最终 PDF、压缩包或仓库链接。joblib 文件只加载本项目自己产生的可信文件。

提交前另做了**事后评分敏感性检查**：只根据主训练/验证文本的非空规范化正文重合标记5条验证样本，利用已保存最佳验证预测另外评分，不读取测试文本来决定标记，不重训、不改数据、主指标或选型。该检查不能排除全部引用或语义近重复偏差。对比见public_results/scoring_sensitivity.csv。

public_results/只放数值表、曲线、混淆矩阵、每类指标、锁定配置、原环境版本及简洁审计/敏感性摘要；没有原始文本、邮箱、姓名、数据、模型、本机绝对路径或逐样本预测。原Windows锁定方案的主验证accuracy=0.9369063772、macro-F1=0.9370619220，三划分macro-F1均值±样本标准差为0.932527±0.006691；这些原主指标没有被敏感性分数替换。

github_upload/是适合手动上传的代码、依赖说明、README、.gitignore及public_results副本；不含原始课程数据、老师PDF、示例/最终预测、模型或检查ZIP。允许提交public_results，私有results继续被gitignore排除。程序不会自动发布仓库或生成最终实验PDF。
