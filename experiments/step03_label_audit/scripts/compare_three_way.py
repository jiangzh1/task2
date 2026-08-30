#!/usr/bin/env python3
"""比较官方标签、Gemma 与 Qwen，并生成中文报告；不修改任何标签。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


LABELS = ["Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"]


def read_ok(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                result[row["sample_id"]] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    gemma = read_ok(args.gemma)
    qwen = read_ok(args.qwen)
    common = sorted(set(gemma) & set(qwen))
    relations = collections.Counter()
    by_official = {label: collections.Counter() for label in LABELS}
    text_agreement = 0
    high_priority = []

    for sample_id in common:
        g = gemma[sample_id]
        q = qwen[sample_id]
        official = g["source_origin_anno"]
        gv = g["sticker_inference"]["parsed"]["primary_label"]
        qv = q["sticker_inference"]["parsed"]["primary_label"]
        gp = g["text_inference"]["parsed"]["polarity"]
        qp = q["text_inference"]["parsed"]["polarity"]

        if official == gv == qv:
            relation = "三方一致"
        elif gv == qv and gv != official:
            relation = "双模型一致反对官方"
            high_priority.append(
                {"sample_id": sample_id, "official": official, "model_consensus": gv}
            )
        elif official == gv and official != qv:
            relation = "官方与Gemma一致"
        elif official == qv and official != gv:
            relation = "官方与Qwen一致"
        else:
            relation = "三方各异"
        relations[relation] += 1
        by_official[official][relation] += 1
        if gp == qp:
            text_agreement += 1

    summary = {
        "gemma_ok": len(gemma),
        "qwen_ok": len(qwen),
        "common_samples": len(common),
        "image_label_relations": dict(relations),
        "relations_by_official": {label: dict(by_official[label]) for label in LABELS},
        "text_polarity_model_agreement": {
            "count": text_agreement,
            "total": len(common),
            "rate": text_agreement / len(common) if common else None,
        },
        "high_priority_candidate_count": len(high_priority),
        "high_priority_candidates": high_priority,
        "policy": "仅生成候选，不修改 origin_anno 或 corrected_label。",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 步骤03双模型三方复核报告",
        "",
        "## 执行范围",
        "",
        f"- Gemma 成功记录：{len(gemma)}条。",
        f"- Qwen 成功记录：{len(qwen)}条。",
        f"- 可进行三方比较的共同样本：{len(common)}条。",
        "- 本报告只生成疑似错标候选，不修改任何官方标签。",
        "",
        "## 图片标签三方关系",
        "",
        "| 关系 | 数量 |",
        "|---|---:|",
    ]
    for name in ["三方一致", "双模型一致反对官方", "官方与Gemma一致", "官方与Qwen一致", "三方各异"]:
        lines.append(f"| {name} | {relations[name]} |")
    rate = text_agreement / len(common) if common else 0
    lines.extend(
        [
            "",
            "## 文本极性双模型关系",
            "",
            f"Gemma 与 Qwen 的文本极性一致 {text_agreement}/{len(common)} 条（{rate:.2%}）。模型不一致样本暂不生成正式文本极性标签。",
            "",
            "## 当前决策",
            "",
            f"- 两个视觉模型共同给出同一非官方类别的高优先级候选共{len(high_priority)}条。",
            "- “高优先级候选”不等于已确认错标，不能自动覆盖 `origin_anno`。",
            "- 下一步需结合实际图片、官方描述字段及类别规则，确定可复现的最终裁决策略。",
            "- 在裁决规则冻结前，不生成正式冲突标签，不修改论文正文或表格。",
            "",
        ]
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
