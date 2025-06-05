## 🧠 CGNTM – Causal Graph Neural Topic Model

CGNTM 是一个集成大语言模型（LLM）、结构因果建模（SCM）、图神经网络（GNN）与生成对抗训练（WGAN）的**因果主题发现模型**，用于从无标签文本中构建主题分层结构与主题因果图，并支持反事实文本生成。

------

## 📁 项目结构

```
├── causal_gnn.py             # 可选的 GNN 模块（未主用）
├── cluster.py                # KMeans 层级聚类构建
├── consistency_loss.py       # BERT-based 对抗生成语义对齐
├── data.py / excel.py        # 预处理或旧版脚本（如未用可忽略）
├── discriminator.py          # 判别器（WGAN）
├── evaluate.py               # 节点评估+5项指标（NPMI / TD / CP / RCR / CSA）
├── graph_builder.py          # 构建因果图 & 保证 DAG
├── hierarchical_gnn.py       # 双层因果 GNN 模型
├── llm_extraction.py         # 使用大语言模型抽取关键词与因果关系
├── neural_scm.py             # 多层感知器结构因果模型
├── train.py                  # 联合训练主脚本（SCM ⇆ WGAN ⇆ HGNN）
├── wgan_generator.py         # 条件生成器
├── requirements.txt
└── README.md
```

------

## ⚙️ 环境配置

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

------

## 🗂️ 输入数据要求

放置至 `data/` 目录下：

- `pub.csv`：原始文本 (列: `pmid`, `title`, `abstract`)

- `pub_causal_triplets.json`：由 `llm_extraction.py` 生成，包括：

  ```json
  [{"keywords": ["lung cancer", "mutation", ...],
    "causal_relations": [{"cause": "smoking", "effect": "lung cancer"}, ...]}]
  ```

------

## 🚀 运行流程

按顺序执行以下命令：

```bash
# Step 1: 从 LLM 生成三元组与关键词
python llm_extraction.py

# Step 2: 构建因果图 DAG + 预测边
python graph_builder.py --root_dir .

# Step 3 (可选): 生成节点聚簇结构（供层级 GNN）
python cluster.py --data_dir data/ -k 6

# Step 4: 将 title+abstract 写入语料文件（每行一文）
python export_corpus.py             # 你可以自定义此脚本

# Step 5: 联合训练：SCM + G + D + HGNN
python train.py --gpu

# Step 6: 评估：节点干预 + 五项指标（若文件齐全）
python evaluate.py --metrics --target "cigarette smoking" --report metrics.json
```

------

## 📊 评估指标

`evaluate.py` 可输出以下论文中使用的指标：

| 指标     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **NPMI** | 主题关键词之间的语义一致性                                   |
| **TD**   | Topic Diversity：不同主题词的覆盖度                          |
| **CP**   | Causal Precision：预测边是否与知识库方向一致                 |
| **RCR**  | Reverse Causality Rate：反向错误预测比例                     |
| **CSA**  | Counterfactual Semantic Alignment：生成反事实文本前后语义一致度 |

这些指标依赖以下文件：

| 文件                    | 来源脚本             |
| ----------------------- | -------------------- |
| `eval/topic_words.json` | `llm_extraction.py`  |
| `eval/corpus.txt`       | `export_corpus.py`   |
| `eval/pred_edges.txt`   | `graph_builder.py`   |
| `eval/true_edges.txt`   | 人工准备或外部知识库 |
| `eval/embeddings.npz`   | `train.py` 自动生成  |

------

## ✅ 训练输出文件结构

```text
data/
 ├ causal_graph.npy         # 邻接矩阵（DAG）
 ├ node_names.json
 └ cluster_ids.npy          # 聚簇标签（可选）

models/
 ├ scm_joint.pth
 ├ g_joint.pth
 ├ d_joint.pth
 ├ hgnn_joint.pth
 └ latent.pt                # 文档-节点表示缓存

eval/
 ├ pred_edges.txt
 ├ true_edges.txt
 ├ topic_words.json
 ├ corpus.txt
 └ embeddings.npz
```

------

## 🧩 依赖

主依赖包括：

- `torch`, `transformers`, `scikit-learn`, `matplotlib`
- 推荐使用：`networkx>=3.0`, `tqdm`

完整见 `requirements.txt`。