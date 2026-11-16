# 工程操作与数据契约指南 (`WORKFLOW_GUIDE`)

本文档补充说明 [`workflow/pipeline.py`](../workflow/pipeline.py) 与各阶段脚本之间的输入/输出契约、离线工作流用法以及下游终表筛选准则。

---

## 1. 数据流转全景

```text
Prepared Doorway Clips, Pose Events & Feature Caches (COPE / Sapiens)
    ├── stages/01_endpoints/           → 有效端点清单与初始几何连续性筛选
    ├── stages/02_static_matching/     → 方案 A–E, K (delivery_records.json + event_dispositions.json)
    ├── stages/03_temporal_matching/   → 方案 F–J (COPE + 膨胀TCN + DTW 融合)
    ├── stages/04_recall_aggregation/  → 多通道双向高召回 R (全部候选.jsonl)
    └── stages/05_feedback_mining/     → 反馈挖掘与补入口 N (new_candidates.json + short_competitors.json)
                                           │
                                           ▼
                       workflow/pipeline.py refresh
                       (按 Ixxxx_Oyyyy 去重、拼接来源血缘与竞争计数)
                                           │
                                           ▼
        dataset/统一数据集.jsonl + 统一标注表.csv + 模型标注任务.jsonl + 审核入口.html
                                           │
                     node workflow/export_xlsx.mjs → 统一标注表.xlsx
                                           │
                     人工 / 视觉模型填写 0/1/2 → pipeline.py import-labels
                                           │
                                           ▼
              Part 5 下游筛选：短停留占位(<30s) → 一对一端点消歧 → 终表导出
```

---

## 2. 核心配置 (`workflow/config.json`)

- `default_project_root`: 默认项目根目录（可通过 `--project-root` 或环境变量 `DWELL_PROJECT_ROOT` 覆盖）。
- `minimum_duration_seconds`: 长停留与短停留竞争分界阈值（默认 `30` 秒）。
- `methods`: 声明收集的方法目录映射（`id`, `name`, `phase`, `directory`）。
- `deduplication_key`: 唯一主键 `["entry_event", "exit_event"]`，格式化为 `I{entry:04d}_O{exit:04d}`。
- `preserve_all_label_history`: 设为 `true`，增量导入标注时保留全部跨轮历史意见而不悄悄多数覆盖。
- `select_one_to_one_before_annotation`: 设为 `false`，确保审核前保留所有一对多竞争配法供比较。

---

## 3. 核心输入数据契约（供接入自定义数据集）

若要将本流程迁移到自己的商铺监控数据集，`--project-root` 下需按 `config.json` 约定组织各阶段输出：

1. **端点索引 (`results/endpoints/{event_id:04d}.json`)**：
   ```json
   {
     "event_id": 12,
     "time": 32410.5,
     "interval": [32410.3, 32410.7],
     "panel": "panels/0012.jpg",
     "videos": [{"path": "videos/0012.mp4"}],
     "views": [{"crop": "crops/0012_01.jpg", "context": "contexts/0012_01.jpg"}]
   }
   ```
2. **各方案输出 (`{phase}/variants/{directory}/delivery_records.json` & `event_dispositions.json`)**：
   每条记录至少包含 `entry_event`, `exit_event`, `entry_seconds`, `exit_seconds`, `entry_interval`, `exit_interval`。
3. **高召回与反馈挖掘输出**：
   - `doorway_recall_20261109/results/全部候选.jsonl`（含 `tier`, `pair_id` 等字段）
   - `doorway_examples_20261109/results/{judgments.json, new_candidates.json, short_competitors.json, positive_union.json, new_entry_proposals.json}`

---

## 4. 标注回填与历史保留机制

执行：
```bash
python3 workflow/pipeline.py import-labels returned_annotations.xlsx
python3 workflow/pipeline.py build
```

- **支持的列**：`配对编号`（必需，格式 `Ixxxx_Oyyyy`）、`本轮标注`（`0` / `1` / `2` / 空）、`进店动作成立`（`是` / `否` / `不确定` / 空）、`备注`。
- **幂等与非破坏性**：
  - 允许只交回部分已标注行；空白单元格不会覆盖已有历史标注。
  - 每次导入以 `(file_sha256, pair_id)` 去重追加到 `workflow/inputs/annotation_updates.jsonl`。
  - 历史各轮原始标注保存在 `workflow/inputs/label_history.json`；跨轮分歧直接标记为 `分歧 X/Y` 供人工裁决。

---

## 5. 下游终表筛选规范（必读）

从统一标注集生成最终可计数的营业停留时长表时，必须依次执行以下三步：

1. **短停留占位（`<30s` 规则）**：先将 `record_use == "短停留竞争"` 且判 `1` 的记录保留在二分图中占据对应入口与出口，防止把真实短访的入口错误关联给更晚出现的其他出口。
2. **一对一端点冲突消歧**：检查所有判 `1` 候选是否存在同一入口对应多个出口或同一出口对应多个入口的竞争簇。按置信层级（多轮/多模型一致 > 单模型判 `1` > 视觉证据人工终裁）为每个冲突端点选定唯一合法进出；同一顾客的多次到访应分属不同事件编号，禁止同一事件编号重复计数。
3. **剔除短停留并导出**：移出所有 `duration_seconds < 30` 的占位项，仅保留标准营业统计列：`进入时间`、`出店时间`、`停留时间`（以及可选外观备注）。
