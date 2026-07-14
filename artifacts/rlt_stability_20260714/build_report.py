#!/usr/bin/env python3
"""Build the portable RLT stability diagnosis artifact from a SwanLab snapshot."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


HEADER_LEN = 7
BLOCK_LEN = 32768
HEADER_MAGIC = 0xE1D6


def scan_swanlab_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete JSON records from a live-safe SwanLab backup snapshot."""
    crc = [0] * 5
    for record_type in range(1, 5):
        crc[record_type] = zlib.crc32(chr(record_type).encode()) & 0xFFFFFFFF

    with path.open("rb") as stream:
        header = stream.read(HEADER_LEN)
        ident, magic, version = struct.unpack("<4sHB", header)
        if ident != b":SWL" or magic != HEADER_MAGIC or version != 1:
            raise ValueError("Unsupported SwanLab backup format")
        index = HEADER_LEN

        while True:
            space_left = BLOCK_LEN - index % BLOCK_LEN
            if space_left < HEADER_LEN:
                padding = stream.read(space_left)
                if len(padding) != space_left:
                    return
                index += space_left

            raw_header = stream.read(HEADER_LEN)
            if len(raw_header) != HEADER_LEN:
                return
            checksum, data_length, record_type = struct.unpack("<IHB", raw_header)
            data = stream.read(data_length)
            index += HEADER_LEN + data_length
            if len(data) != data_length:
                return
            if zlib.crc32(data, crc[record_type]) & 0xFFFFFFFF != checksum:
                return

            if record_type == 1:
                yield json.loads(data.decode())
                continue
            if record_type != 2:
                return

            chunks = [data]
            while True:
                raw_header = stream.read(HEADER_LEN)
                if len(raw_header) != HEADER_LEN:
                    return
                checksum, data_length, record_type = struct.unpack("<IHB", raw_header)
                data = stream.read(data_length)
                index += HEADER_LEN + data_length
                if len(data) != data_length:
                    return
                if zlib.crc32(data, crc[record_type]) & 0xFFFFFFFF != checksum:
                    return
                chunks.append(data)
                if record_type == 4:
                    break
                if record_type != 3:
                    return
            yield json.loads(b"".join(chunks).decode())


def load_scalar_metrics(path: Path) -> dict[str, dict[int, float]]:
    metrics: dict[str, dict[int, float]] = defaultdict(dict)
    for record in scan_swanlab_records(path):
        if record.get("model_type") != "Scalar":
            continue
        data = record["data"]
        metrics[data["key"]][int(data["step"])] = float(data["metric"]["data"])
    return metrics


def query_rows(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def sql_source(
    source_id: str,
    label: str,
    sql: str,
    description: str,
    executed_at: str,
    tables: list[str],
    *,
    filters: list[str] | None = None,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "sqlite3",
            "id": source_id,
            "sql": sql,
            "description": description,
            "language": "sql",
            "executed_at": executed_at,
            "tables_used": tables,
            "filters": filters or [],
            "metric_definitions": metric_definitions or [],
        },
    }


def build_artifact(snapshot_path: Path) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metrics = load_scalar_metrics(snapshot_path)
    loss_by_step = metrics["train/loss"]
    if not loss_by_step:
        raise ValueError("SwanLab snapshot contains no train/loss metrics")

    steps = sorted(loss_by_step)
    latest_step = steps[-1]
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE training_metrics (
            step INTEGER PRIMARY KEY,
            loss REAL,
            grad_norm REAL,
            log10_loss REAL,
            log10_grad_norm REAL,
            image_mse REAL,
            non_image_mse REAL,
            cosine_similarity REAL,
            rl_token_l2 REAL,
            target_max REAL,
            loss_nonfinite INTEGER,
            grad_nonfinite INTEGER
        )
        """
    )

    def finite_or_none(value: float | None) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return value

    for step in steps:
        raw_loss = loss_by_step[step]
        raw_grad_norm = metrics["train/grad_global_norm"].get(step)
        loss = finite_or_none(raw_loss)
        grad_norm = finite_or_none(raw_grad_norm)
        connection.execute(
            "INSERT INTO training_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                step,
                loss,
                grad_norm,
                math.log10(max(loss, 1.0e-30)) if loss is not None else None,
                math.log10(max(grad_norm, 1.0e-30)) if grad_norm is not None else None,
                finite_or_none(metrics["loss/reconstruction_image/mse"].get(step)),
                finite_or_none(metrics["loss/reconstruction_non_image/mse"].get(step)),
                finite_or_none(metrics["loss/reconstruction/cosine_similarity"].get(step)),
                finite_or_none(metrics["rl_token/l2_mean"].get(step)),
                finite_or_none(metrics["embedding/target/max"].get(step)),
                int(not math.isfinite(raw_loss)),
                int(raw_grad_norm is not None and not math.isfinite(raw_grad_norm)),
            ),
        )

    connection.execute(
        """
        CREATE TABLE token_stats (
            token_type TEXT,
            tokens_per_sample INTEGER,
            std REAL,
            min_value REAL,
            max_value REAL,
            mean_l2_per_token REAL,
            interpretation TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO token_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("image", 192, 4.05, -64.5, 164.0, 182.0, "与论文/复现的 image-only 输入一致"),
            ("non-image", 24, 81.0, -185.0, 15168.0, 908.0, "包含固定语言/特殊 token 巨型激活"),
            ("all", 216, 27.28, -185.0, 15168.0, 263.0, "当前缓存实际训练目标"),
        ],
    )

    connection.execute(
        """
        CREATE TABLE architecture_comparison (
            component TEXT,
            paper TEXT,
            yyshadow TEXT,
            current_run TEXT,
            implication TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO architecture_comparison VALUES (?, ?, ?, ?, ?)",
        [
            (
                "Encoder",
                "将 learned RL embedding 追加到 VLA token 序列；取最后位置输出",
                "learned query 通过 cross-attention 读取 prefix",
                "跟随 Yyshadow query cross-attention",
                "当前不是论文 Eq. (1) 的逐字实现",
            ),
            (
                "Decoder",
                "自回归 teacher forcing：[z_rl, stopgrad(z_1:i-1)] → z_i",
                "learned position query 仅 cross-attend RL token",
                "跟随 Yyshadow；不接真实 prefix",
                "这是两条不同的训练目标，不能同时称为 paper-exact",
            ),
            (
                "实验 token",
                "固定指令任务丢弃 language embedding",
                "训练脚本 image_only=True",
                "token_scope=all：192 image + 24 non-image",
                "当前包含论文和复现均排除的高幅值 token",
            ),
            (
                "序列长度",
                "M 个实际 VLA token",
                "实际 image prefix 长度",
                "256 query，只有 216 有效；40 padding query 参与 self-attention",
                "padding 虽不计 loss，仍可影响有效 query",
            ),
        ],
    )

    connection.execute(
        """
        CREATE TABLE recipe_comparison (
            setting TEXT,
            current_run TEXT,
            yyshadow_reference TEXT,
            paper TEXT,
            risk TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO recipe_comparison VALUES (?, ?, ?, ?, ?)",
        [
            ("输入范围", "all tokens", "image_only=True", "实验中丢弃 language", "高：目标尺度病态"),
            ("可训练 RLT 精度", "BF16 autocast", "FP32", "未披露", "高：大激活进入 attention"),
            ("peak LR", "peak=1e-4", "peak=2.5e-5", "未披露", "高：当前 4×"),
            ("warmup", "100 steps", "1,000 steps", "未披露", "高：当前短 10×"),
            ("AdamW beta2", "beta2=0.999", "beta2=0.95", "未披露", "高：二阶矩响应不同"),
            ("weight decay", "WD=1e-4", "WD=1e-10", "未披露", "中：非同一优化器"),
            ("训练步数", "200,000 steps", "5,000 steps", "2,000–10,000 steps", "高：当前超出 20–100×"),
            ("EMA", "none", "decay=0.99", "未披露", "中：评估/保存不对齐"),
            ("缓存/前向", "固定 FP16 cache", "在线 VLA 前向", "单任务 demo", "中：增强与量化不对齐"),
        ],
    )

    training_sql = """
        SELECT step, loss, grad_norm, log10_loss, log10_grad_norm,
               image_mse, non_image_mse, cosine_similarity, rl_token_l2, target_max,
               loss_nonfinite, grad_nonfinite
        FROM training_metrics
        ORDER BY step
    """.strip()
    summary_sql = """
        SELECT
            MAX(step) AS snapshot_step,
            (SELECT loss FROM training_metrics ORDER BY step DESC LIMIT 1) AS latest_loss,
            MAX(loss) AS max_loss,
            MAX(grad_norm) AS max_grad_norm,
            SUM(loss_nonfinite) AS nonfinite_loss_records,
            SUM(grad_nonfinite) AS nonfinite_grad_records
        FROM training_metrics
    """.strip()
    token_sql = """
        SELECT token_type, tokens_per_sample, std, min_value, max_value,
               mean_l2_per_token, interpretation
        FROM token_stats
        ORDER BY CASE token_type WHEN 'image' THEN 1 WHEN 'non-image' THEN 2 ELSE 3 END
    """.strip()
    architecture_sql = """
        SELECT component, paper, yyshadow, current_run, implication
        FROM architecture_comparison
        ORDER BY rowid
    """.strip()
    recipe_sql = """
        SELECT setting, current_run, yyshadow_reference, paper, risk
        FROM recipe_comparison
        ORDER BY rowid
    """.strip()
    timeline_steps = sorted({8000, 8360, 8390, 8500, 11140, latest_step})
    timeline_sql = f"""
        SELECT step, loss, grad_norm, image_mse, non_image_mse, rl_token_l2
        FROM training_metrics
        WHERE step IN ({', '.join(str(step) for step in timeline_steps)})
        ORDER BY step
    """.strip()

    training_rows = query_rows(connection, training_sql)
    summary_rows = query_rows(connection, summary_sql)
    token_rows = query_rows(connection, token_sql)
    architecture_rows = query_rows(connection, architecture_sql)
    recipe_rows = query_rows(connection, recipe_sql)
    timeline_rows = query_rows(connection, timeline_sql)
    nonfinite_grad_records = int(summary_rows[0]["nonfinite_grad_records"] or 0)
    connection.close()

    run_source = sql_source(
        "swanlab_run_sql",
        "SwanLab scalar snapshot (node2)",
        training_sql,
        "Parses complete Scalar records from the read-only SwanLab backup snapshot and selects the logged training series.",
        generated_at,
        ["parsed_swanlab_scalar_records.training_metrics"],
        filters=[f"resume run steps 5010 through {latest_step}", "10-step logging cadence"],
        metric_definitions=[
            "loss = masked mean squared reconstruction error for the current batch",
            "grad_norm = global gradient norm returned before clip_grad_norm_(..., 1.0)",
            "log10 fields are base-10 transforms for readable plotting",
            "non-finite scalar values are flagged and represented as null in chart rows",
        ],
    )
    summary_source = sql_source(
        "run_summary_sql",
        "SwanLab run summary query",
        summary_sql,
        "Computes the latest snapshot step and extrema from the parsed SwanLab metric table.",
        generated_at,
        ["parsed_swanlab_scalar_records.training_metrics"],
    )
    timeline_source = sql_source(
        "timeline_sql",
        "Selected divergence checkpoints",
        timeline_sql,
        "Selects representative stable, precursor, divergence, and latest logged steps.",
        generated_at,
        ["parsed_swanlab_scalar_records.training_metrics"],
    )
    token_source = sql_source(
        "token_audit_sql",
        "Embedding cache token-scale audit",
        token_sql,
        "Selects audited cache statistics from mmap inspection of shard_000000.pt; 64-sample ranges were checked and the fixed extreme coordinates were confirmed across sampled shards.",
        generated_at,
        ["node2_cache_audit.token_stats"],
        filters=["valid tokens only", "image/non-image split from packed_image_mask"],
        metric_definitions=["std and L2 statistics are computed after float16 cache values are cast to float32"],
    )
    architecture_source = sql_source(
        "architecture_audit_sql",
        "Paper / Yyshadow / current architecture audit",
        architecture_sql,
        "Selects the line-by-line architecture comparison assembled from paper Eq. (1)-(2), Yyshadow source, and the current PyTorch port.",
        generated_at,
        ["reviewed_architecture_comparison"],
    )
    recipe_source = sql_source(
        "recipe_audit_sql",
        "Current versus Yyshadow training recipe audit",
        recipe_sql,
        "Selects the reviewed training-recipe comparison from current training_config.json and the retained/open GitHub source defaults.",
        generated_at,
        ["reviewed_recipe_comparison"],
    )
    paper_source = {
        "id": "rlt_paper",
        "label": "RL Token paper, arXiv v2",
        "href": "https://arxiv.org/pdf/2604.23073v2",
    }
    yyshadow_model_source = {
        "id": "yyshadow_model",
        "label": "Yyshadow/openpi-RLT RL token model",
        "href": "https://github.com/Yyshadow/openpi-RLT/blob/main/src/openpi/models/rl_token.py",
    }
    yyshadow_train_source = {
        "id": "yyshadow_train",
        "label": "Yyshadow/openpi-RLT Stage-1 training code",
        "href": "https://github.com/Yyshadow/openpi-RLT/blob/main/scripts/train_rlt.py",
    }
    yyshadow_optimizer_source = {
        "id": "yyshadow_optimizer",
        "label": "Yyshadow/openpi-RLT optimizer defaults",
        "href": "https://github.com/Yyshadow/openpi-RLT/blob/main/src/openpi/training/optimizer.py",
    }
    swanlab_href_source = {
        "id": "swanlab_run",
        "label": "Current SwanLab run",
        "href": "https://swanlab.cn/@duffytec/groot-rlt/runs/93e45fx24psa8pxmtmawk",
    }
    sources = [
        run_source,
        summary_source,
        timeline_source,
        token_source,
        architecture_source,
        recipe_source,
        paper_source,
        yyshadow_model_source,
        yyshadow_train_source,
        yyshadow_optimizer_source,
        swanlab_href_source,
    ]

    title = "RLT Stage-1 训练稳定性诊断"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "对照 arXiv 原文、Yyshadow/openpi-RLT 与 node2 当前运行的只读技术诊断。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "snapshot_step_card",
                    "dataset": "run_summary",
                    "sourceId": "run_summary_sql",
                    "description": "本报告冻结的 SwanLab 快照位置",
                    "metrics": [{"label": "Snapshot step", "field": "snapshot_step", "format": "number"}],
                },
                {
                    "id": "latest_loss_card",
                    "dataset": "run_summary",
                    "sourceId": "run_summary_sql",
                    "description": "快照末尾 batch 的 masked MSE",
                    "metrics": [{"label": "Latest loss", "field": "latest_loss", "format": "number"}],
                },
                {
                    "id": "max_loss_card",
                    "dataset": "run_summary",
                    "sourceId": "run_summary_sql",
                    "description": "恢复运行内的单 batch 最大 loss",
                    "metrics": [{"label": "Max loss", "field": "max_loss", "format": "compact"}],
                },
                {
                    "id": "max_grad_card",
                    "dataset": "run_summary",
                    "sourceId": "run_summary_sql",
                    "description": "裁剪前全局梯度范数峰值",
                    "metrics": [{"label": "Max pre-clip grad", "field": "max_grad_norm", "format": "compact"}],
                },
                {
                    "id": "nonfinite_grad_card",
                    "dataset": "run_summary",
                    "sourceId": "run_summary_sql",
                    "description": "快照中已记录为 Inf/NaN 的梯度范数点数",
                    "metrics": [{"label": "Non-finite grad logs", "field": "nonfinite_grad_records", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "loss_curve",
                    "title": "恢复运行的重建 loss",
                    "subtitle": "loss 多次从约 2.4 跃迁至 10^5 量级；纵轴为 log10(masked MSE)。",
                    "type": "line",
                    "dataset": "training_curve",
                    "sourceId": "swanlab_run_sql",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Optimizer step"},
                        "y": {"field": "log10_loss", "type": "quantitative", "label": "log10(loss)"},
                        "tooltip": [
                            {"field": "step", "type": "quantitative", "label": "Step", "format": "number"},
                            {"field": "loss", "type": "quantitative", "label": "Loss", "format": "number"},
                            {"field": "non_image_mse", "type": "quantitative", "label": "Non-image MSE", "format": "number"},
                        ],
                    },
                    "xAxisTitle": "Optimizer step",
                    "yAxisTitle": "log10(masked MSE)",
                    "layout": "full",
                    "maxRows": len(training_rows),
                },
                {
                    "id": "grad_curve",
                    "title": "裁剪前全局梯度范数",
                    "subtitle": "grad clip=1 几乎全程触发，且灾难性 loss 峰值与 10^8–10^9 级梯度峰值同步。",
                    "type": "line",
                    "dataset": "training_curve",
                    "sourceId": "swanlab_run_sql",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Optimizer step"},
                        "y": {"field": "log10_grad_norm", "type": "quantitative", "label": "log10(pre-clip grad norm)"},
                        "tooltip": [
                            {"field": "step", "type": "quantitative", "label": "Step", "format": "number"},
                            {"field": "grad_norm", "type": "quantitative", "label": "Pre-clip grad", "format": "compact"},
                            {"field": "rl_token_l2", "type": "quantitative", "label": "RL token L2", "format": "number"},
                        ],
                    },
                    "xAxisTitle": "Optimizer step",
                    "yAxisTitle": "log10(pre-clip global norm)",
                    "layout": "full",
                    "maxRows": len(training_rows),
                },
            ],
            "tables": [
                {
                    "id": "timeline_table",
                    "title": "稳定、先导与发散节点",
                    "subtitle": "non-image MSE 先抬升，随后 image 与 latent 一起失稳。",
                    "dataset": "divergence_timeline",
                    "sourceId": "timeline_sql",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "step", "label": "Step", "format": "number", "align": "right"},
                        {"field": "loss", "label": "Loss", "format": "number", "align": "right"},
                        {"field": "grad_norm", "label": "Pre-clip grad", "format": "compact", "align": "right"},
                        {"field": "image_mse", "label": "Image MSE", "format": "number", "align": "right"},
                        {"field": "non_image_mse", "label": "Non-image MSE", "format": "number", "align": "right"},
                        {"field": "rl_token_l2", "label": "RL token L2", "format": "number", "align": "right"},
                    ],
                },
                {
                    "id": "token_stats_table",
                    "title": "缓存 token 尺度",
                    "subtitle": "non-image token 的尺度远高于 image token；固定极值 15,168 贯穿所有 batch。",
                    "dataset": "token_stats",
                    "sourceId": "token_audit_sql",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "token_type", "label": "Token type"},
                        {"field": "tokens_per_sample", "label": "Tokens/sample", "format": "number", "align": "right"},
                        {"field": "std", "label": "Std", "format": "number", "align": "right"},
                        {"field": "min_value", "label": "Min", "format": "number", "align": "right"},
                        {"field": "max_value", "label": "Max", "format": "number", "align": "right"},
                        {"field": "mean_l2_per_token", "label": "Mean token L2", "format": "number", "align": "right"},
                        {"field": "interpretation", "label": "Interpretation"},
                    ],
                },
                {
                    "id": "architecture_table",
                    "title": "论文、Yyshadow 与当前架构数据流",
                    "subtitle": "当前网络跟随 Yyshadow 的 learned-query bottleneck，而不是论文 Eq. (1)–(2)。",
                    "dataset": "architecture_comparison",
                    "sourceId": "architecture_audit_sql",
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "component", "label": "Component", "type": "text"},
                        {"field": "paper", "label": "Paper", "type": "text"},
                        {"field": "yyshadow", "label": "Yyshadow", "type": "text"},
                        {"field": "current_run", "label": "Current", "type": "text"},
                        {"field": "implication", "label": "Implication", "type": "text"},
                    ],
                },
                {
                    "id": "recipe_table",
                    "title": "Stage-1 训练配方对照",
                    "subtitle": "当前的 LR、warmup、AdamW、精度、token 范围和训练长度均未复现 Yyshadow。",
                    "dataset": "recipe_comparison",
                    "sourceId": "recipe_audit_sql",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "setting", "label": "Setting", "type": "text"},
                        {"field": "current_run", "label": "Current", "type": "text"},
                        {"field": "yyshadow_reference", "label": "Yyshadow reference", "type": "text"},
                        {"field": "paper", "label": "Paper", "type": "text"},
                        {"field": "risk", "label": "Risk", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 技术结论\n\n"
                        "当前不稳定是可复现的 activation/latent explosion，不是普通 batch 噪声。主因按证据强度排序为："
                        "（1）训练了论文与 Yyshadow 都排除的高幅值 non-image token；（2）LR、warmup、AdamW 与 FP32 配方没有复现；"
                        "（3）200k horizon 使 LR 在论文的 2k–10k 窗口内几乎不衰减；（4）256 个 decoder query 中 40 个 padding query 仍参与 self-attention。"
                        "另外，[原论文](https://arxiv.org/pdf/2604.23073v2) 与 "
                        "[Yyshadow 实现](https://github.com/Yyshadow/openpi-RLT/blob/main/src/openpi/models/rl_token.py)"
                        "本身采用不同 decoder 目标，因此必须先明确复现对象。"
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "snapshot_step_card",
                        "latest_loss_card",
                        "max_loss_card",
                        "max_grad_card",
                        "nonfinite_grad_card"
                    ],
                    "layout": "full",
                },
                {
                    "id": "run_evidence",
                    "type": "markdown",
                    "sourceId": "swanlab_run_sql",
                    "layout": "full",
                    "body": (
                        "## 运行证据\n\n"
                        "恢复运行在 5010–6500 步保持约 2.4 的 loss，之后多次进入有限值但灾难性的振荡。"
                        "梯度日志是 clip 前范数；即使稳定区间也远高于 clip=1，说明裁剪一直在工作，却没有阻止 Adam 状态和 encoder 参数持续漂移。"
                        f"最新快照中已有 {nonfinite_grad_records} 个梯度范数记录成为非有限值。"
                    ),
                },
                {"id": "loss_curve_block", "type": "chart", "chartId": "loss_curve", "layout": "full"},
                {"id": "grad_curve_block", "type": "chart", "chartId": "grad_curve", "layout": "full"},
                {"id": "timeline_block", "type": "table", "tableId": "timeline_table", "layout": "full"},
                {
                    "id": "data_root_cause",
                    "type": "markdown",
                    "sourceId": "token_audit_sql",
                    "layout": "full",
                    "body": (
                        "## 根因 1：训练目标被 non-image 极值支配\n\n"
                        "当前每个样本包含 192 个 image token 和 24 个 non-image token。第 0 个 non-image token 的固定分量 "
                        "15,168、9,536、940 单独贡献了零预测目标能量的约 97.74%。Cross-attention 的 memory K/V 未先做归一化，"
                        "这些极值再进入 BF16 matmul，形成病态优化问题。首次大爆炸前，non-image MSE 已先于 image MSE 抬升。"
                    ),
                },
                {"id": "token_stats_block", "type": "table", "tableId": "token_stats_table", "layout": "full"},
                {
                    "id": "paper_method",
                    "type": "markdown",
                    "sourceId": "rlt_paper",
                    "layout": "full",
                    "body": (
                        "## 原论文真正写的 Stage 1\n\n"
                        "论文第 4 页 Eq. (1) 将 learned RL embedding 追加到 VLA final-layer token 序列，并取特殊位置输出；"
                        "Eq. (2) 的 decoder 是自回归 teacher forcing：输入 `[z_rl, stopgrad(z_1:i-1)]` 预测 `z_i`。"
                        "附录报告每任务训练 2,000–10,000 gradient steps，并在固定指令实验中丢弃 language embeddings。"
                        "论文没有公开 optimizer、LR、batch size、精度、层数、heads 或 loss reduction 的实现细节。"
                    ),
                },
                {
                    "id": "yyshadow_method",
                    "type": "markdown",
                    "sourceId": "yyshadow_model",
                    "layout": "full",
                    "body": (
                        "## Yyshadow 复现是另一条实现路径\n\n"
                        "Yyshadow 使用 learned query cross-attention encoder，并用 learned position queries 仅从 RL token 重建 prefix；"
                        "这与论文的 appended-token encoder 和 teacher-forced autoregressive decoder 不同。当前 PyTorch 主干基本复现了这条路径，"
                        "但训练数据、精度、优化器、padding 处理和初始化并未完全复现。"
                    ),
                },
                {"id": "architecture_block", "type": "table", "tableId": "architecture_table", "layout": "full"},
                {
                    "id": "recipe_root_cause",
                    "type": "markdown",
                    "sourceId": "recipe_audit_sql",
                    "layout": "full",
                    "body": (
                        "## 根因 2：网络形状对齐了，训练配方没有\n\n"
                        "当前 peak LR 是 Yyshadow 默认值的 4 倍，warmup 只有其十分之一，beta2 与 weight decay 也不同；"
                        "而 200k cosine horizon 令 10k 附近 LR 仍约 9.9e-5。稳定的 005000.pt 恰好位于 Yyshadow 配置的终点，"
                        "继续以近峰值 LR 训练后 encoder 参数范数和 Adam 二阶矩显著漂移。"
                    ),
                },
                {"id": "recipe_block", "type": "table", "tableId": "recipe_table", "layout": "full"},
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 建议的下一次实验\n\n"
                        "1. **先选择复现目标。** Paper-exact 应改为 appended RL token + teacher-forced autoregressive decoder；"
                        "Yyshadow-exact 才保留当前 learned-query encoder / position-query decoder。\n\n"
                        "2. **无论选哪条，都从 step 0 重训。** 重建 image-only、实际长度 192、无 padding query 的缓存；"
                        "RLT 全程 FP32。不要沿用 005000.pt 或 010000.pt，因为前者已学习错误目标，后者还包含发散后的 optimizer 状态。\n\n"
                        "3. **若复现 Yyshadow：** peak LR 2.5e-5、warmup 1000、AdamW betas=(0.9,0.95)、eps=1e-8、"
                        "weight decay=1e-10、clip=1、EMA=0.99、5,000 steps。若只改命令行后 resume，旧 optimizer state 会覆盖新 betas/WD。\n\n"
                        "4. **先做最小 A/B：** 同一随机种子各跑 1,000 steps：当前 all-token BF16 与 image-only FP32 reference recipe；"
                        "比较 loss、pre-clip grad、RL-token L2、encoder parameter L2。后者稳定后再扩到 5,000 steps。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 方法、范围与限制\n\n"
                        "这是只读诊断：解析当前 SwanLab backup 的完整 Scalar 记录，mmap 抽查缓存 shard，审计 005000/010000 checkpoint 统计，"
                        "并逐行对照 arXiv v2、Yyshadow 公开源码及当前 PyTorch port。没有停止、重启或修改训练。"
                        "缓存统计以 shard_000000 的 64–256 个样本为定量样本，并跨抽样 shard 验证固定极值；它足以说明尺度问题，但不是全缓存逐元素扫描。"
                        "论文未披露 Stage-1 optimizer 等实现细节，因此只能做到 paper-architecture exact，不能从论文单独恢复 paper-recipe exact。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 进一步验证问题\n\n"
                        "- image-only FP32 + reference optimizer 是否能在 5,000 步内保持 grad norm 与 RL-token norm 有界？\n"
                        "- Paper teacher forcing 与 Yyshadow pure bottleneck 的 downstream RL token 质量，哪一个在相同 reconstruction 指标下更好？\n"
                        "- 在线抽取 prefix 与固定 FP16 cache 的差异，来自数据增强还是量化？\n"
                        "- 若必须使用缓存，float32 cache 与逐 token 标准化是否有必要，且是否破坏与 VLA 表征空间的一致性？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "training_curve": training_rows,
                "run_summary": summary_rows,
                "divergence_timeline": timeline_rows,
                "token_stats": token_rows,
                "architecture_comparison": architecture_rows,
                "recipe_comparison": recipe_rows,
            },
        },
        "sources": sources,
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = build_artifact(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
