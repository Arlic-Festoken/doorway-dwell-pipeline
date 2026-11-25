# Doorway Dwell-Time Reconstruction Pipeline

面向商铺门口监控视频的**客户进出配对与停留时长重建**工程流水线。

- 方法原理展示：[`docs/methodology_slides.pptx`](docs/methodology_slides.pptx)
- 输入数据契约与详细协议：[`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md)

> 说明：本仓库仅包含代码与文档，不包含原始监控视频、人物图像裁剪、标注数据或模型权重。

---

## 1. 环境准备

- `workflow/pipeline.py`、`workflow/package.py`：仅需 Python 3 标准库。
- `stages/` 上游模型与特征阶段：

  ```bash
  pip install -r requirements.txt
  ```

- `workflow/export_xlsx.mjs`（可选，用于导出带样式和超链接的 XLSX）：

  ```bash
  npm install   # 需要 Node.js 与 @oai/artifact-tool
  ```

---

## 2. 快速验证（烟雾测试）

无需原始视频或 GPU，先运行脱敏合成数据端到端测试验证环境：

```bash
python3 -m unittest tests/test_pipeline_smoke.py -v
```

---

## 3. 常用命令

### 3.1 汇总去重与生成标注集

准备好各阶段输出目录后（目录结构见 [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md)），通过 `--project-root` 或环境变量 `DWELL_PROJECT_ROOT` 指定项目根目录：

```bash
# 从各阶段产出收集、去重并生成统一标注集
python3 workflow/pipeline.py refresh --project-root /path/to/project_root

# 基于已有 workflow/inputs/ 快照离线重建同一数据集
python3 workflow/pipeline.py build
```

执行后会在输出目录 `dataset/` 下生成：`统一数据集.jsonl`、`统一标注表.csv`、`模型标注任务.jsonl` 与离线审核页 `审核入口.html`。

### 3.2 导出 XLSX 标注表（可选）

```bash
node workflow/export_xlsx.mjs
```

### 3.3 回填人工或视觉模型标注

支持 `.xlsx` 或 `.csv`（按 `配对编号` 如 `Ixxxx_Oyyyy` 增量回填，空白单元格不覆盖历史标注）：

```bash
python3 workflow/pipeline.py import-labels /path/to/annotated.xlsx
python3 workflow/pipeline.py build
```

### 3.4 打包自包含分发包

将流水线代码、统一标注表、离线审核页面及实体媒体文件打包为单个 ZIP：

```bash
python3 workflow/package.py --output dist/dwell_bundle.zip
```

### 3.5 在独立目录重跑模型阶段（Part 2–3）

若已准备好端点特征与视频片段缓存，可在新输出目录中重跑且不覆盖原结果：

```bash
python3 workflow/run_methods.py \
  --project-root /path/to/project_root \
  --run-root /path/to/new_run_output \
  --stages static temporal recall examples \
  --gpu 0
```

---

## 4. 目录说明

```text
doorway-dwell-pipeline/
├── workflow/   # 汇总去重(pipeline.py)、调度(run_methods.py)、导出、打包与审核页
├── stages/     # 各阶段实现（01_endpoints ~ 05_feedback_mining）
├── docs/       # 方法 PPT (methodology_slides.pptx) 与数据契约文档
└── tests/      # 合成数据烟雾测试 (test_pipeline_smoke.py)
```
