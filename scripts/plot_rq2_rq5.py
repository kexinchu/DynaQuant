#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TEST1 = ROOT / "results/evaluation/Test1"
FIG = ROOT / "SIGMETRICS_2027/figures"
TABLES = ROOT / "SIGMETRICS_2027/tables"

VARIANT_LABEL = {
    "full": r"\systemname",
    "ind_value": "Independent-value",
    "slack_only": "Slack-only",
    "fixed_partition": "Fixed-partition",
    "single_timescale": "Single-timescale",
    "no_tenure": "No-tenure",
    "no_hysteresis": "No-hysteresis",
    "blocking": "Blocking",
}
EXCHANGE_LABEL = {
    ("lookahead", "fidelity"): r"Look $\leftarrow$ fid",
    ("lookahead", "residency"): r"Look $\leftarrow$ res",
    ("residency", "fidelity"): r"Res $\leftarrow$ fid",
    ("fidelity", "residency"): r"Fid $\leftarrow$ res",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str) -> float:
    return float(row[key])


def maybe(row: dict, key: str) -> float | None:
    text = row.get(key, "")
    return float(text) if text else None


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

JOINT_ORDER = ("seq_rf", "seq_fr", "dynabyte")
JOINT_LABEL = {"dynabyte": "DynaByte", "seq_fr": "Seq-FR", "seq_rf": "Seq-RF"}
JOINT_COLOR = {"dynabyte": "#c0392b", "seq_fr": "#148f77", "seq_rf": "#6c3483"}


def plot_necessity(
    joint: list[dict],
    adaptation: list[dict],
    joint_path: Path,
    adapt_path: Path,
    legend_path: Path,
) -> None:
    by_policy = {row["policy"]: row for row in joint}
    rows = [by_policy[name] for name in JOINT_ORDER]
    style = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8}
    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=(6.4*0.5, 4.8*0.5), constrained_layout=True)
        y = np.arange(len(rows))
        values = [num(row, "p95_tpot_ms") for row in rows]
        xerr = [
            [num(row, "p95_tpot_ms") - num(row, "ci_low") for row in rows],
            [num(row, "ci_high") - num(row, "p95_tpot_ms") for row in rows],
        ]
        axis.barh(
            y, values, xerr=xerr, height=0.62,
            color=[JOINT_COLOR[row["policy"]] for row in rows],
            error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": "#2c3e50"},
        )
        for index, row in enumerate(rows):
            axis.text(
                num(row, "ci_high") + 4, index,
                rf"$\Delta$PPL {num(row, 'delta_ppl'):.2f}",
                va="center", fontsize=8, color="#2c3e50",
            )
        axis.set_yticks(y, [JOINT_LABEL[row["policy"]] for row in rows])
        axis.set_xlabel("P95 TPOT (ms)")
        axis.set_xlim(42.0, 56.0)
        save(fig, joint_path)

    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=(6.4*0.5, 4.8*0.5), constrained_layout=True)
        time = [num(row, "t_s") for row in adaptation]
        for key, label, color in (
            ("fid_pct", "Fidelity", "#1f4e79"),
            ("res_pct", "Residency", "#1e8449"),
            ("look_pct", "Lookahead", "#b9770e"),
        ):
            axis.plot(time, [num(row, key) for row in adaptation], marker="o", color=color, label=label, linewidth=1.2)
        axis.set_ylim(0, 100)
        twin = axis.twinx()
        twin.plot(time, [num(row, "dynabyte_tpot_ms") for row in adaptation], color="#c0392b", linestyle="--", marker="s", label="DynaByte TPOT")
        twin.plot(time, [num(row, "static_tpot_ms") for row in adaptation], color="#5d6d7e", linestyle=":", marker="s", label="Static-Joint TPOT")
        twin.set_ylim(40.0, 62.0)
        axis.axvline(0, color="#922b21", linewidth=0.8, linestyle="--")
        axis.text(-28, 90, "decode-heavy", fontsize=8, color="#922b21", ha="left")
        axis.set_xlabel("Time from slice change (s)")
        axis.set_ylabel("Bytes (% of B)")
        twin.set_ylabel("p95 TPOT (ms)")
        save(fig, adapt_path)
        entries = (
            ("Fidelity", {"color": "#1f4e79", "marker": "o", "linestyle": "-"}),
            ("Residency", {"color": "#1e8449", "marker": "o", "linestyle": "-"}),
            ("Lookahead", {"color": "#b9770e", "marker": "o", "linestyle": "-"}),
            ("DynaByte TPOT", {"color": "#c0392b", "marker": "s", "linestyle": "--"}),
            ("Static-Joint TPOT", {"color": "#5d6d7e", "marker": "s", "linestyle": ":"}),
        )
        legend_handles = [
            Line2D([0], [0], linewidth=1.2, markersize=4.5, **style) for _, style in entries
        ]
        legend = plt.figure(figsize=(7.05, 0.28))
        legend.legend(
            legend_handles, [label for label, _ in entries], loc="center", ncol=5,
            frameon=False, fontsize=8, handlelength=2.4, columnspacing=1.1,
        )
        save(legend, legend_path)


ABLATION_ORDER = ("fixed_partition", "slack_only", "ind_value", "full")
ABLATION_COLOR = {
    "full": "#c0392b",
    "ind_value": "#1f4e79",
    "slack_only": "#b9770e",
    "fixed_partition": "#148f77",
}


def plot_mechanisms(ablation: list[dict], paths: dict[str, Path]) -> None:
    limits = {"Qwen3-30B": (42.0, 58.0), "Qwen3-Next-80B": (100, 240)}
    style = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8}
    for model, path in paths.items():
        by_variant = {row["variant"]: row for row in ablation if row["model"] == model}
        rows = [by_variant[name] for name in ABLATION_ORDER]
        with plt.rc_context(style):
            fig, axis = plt.subplots(figsize=(6.4 * 0.5, 4.8 * 0.5), constrained_layout=True)
            y = np.arange(len(rows))
            values = [num(row, "p95_tpot_ms") for row in rows]
            xerr = [
                [num(row, "p95_tpot_ms") - num(row, "ci_low") for row in rows],
                [num(row, "ci_high") - num(row, "p95_tpot_ms") for row in rows],
            ]
            axis.barh(
                y, values, xerr=xerr, height=0.62,
                color=[ABLATION_COLOR[row["variant"]] for row in rows],
                error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": "#2c3e50"},
            )
            static_ms = num(rows[0], "static_ms")
            axis.axvline(static_ms, color="#2c3e50", linestyle="--", linewidth=0.9)
            axis.set_xlim(*limits[model])
            axis.set_yticks(y, [VARIANT_LABEL[row["variant"]].replace(r"\systemname", "DynaByte") for row in rows])
            axis.set_xlabel("P95 TPOT (ms)")
            axis.legend(
                [Line2D([0], [0], color="#2c3e50", linestyle="--", linewidth=0.9)],
                ["Static-Joint"],
                frameon=False, fontsize=8, loc="upper right",
            )
            save(fig, path)


def plot_validation(wait, risk, recall, scaling, paths: dict[str, Path]) -> None:
    style = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}
    panel = (6.4 * 0.5, 4.8 * 0.5)
    colors = {"prefill": "#1f4e79", "mixed": "#b9770e", "decode": "#1e8449"}
    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=panel, constrained_layout=True)
        for regime, color in colors.items():
            rows = [row for row in wait if row["regime"] == regime]
            axis.scatter(
                [num(row, "predicted_ms") for row in rows],
                [num(row, "measured_ms") for row in rows],
                s=18, color=color, label=regime, zorder=3,
            )
        limit = 30
        axis.plot([0, limit], [0, limit], color="#7f8c8d", linewidth=0.8, linestyle="--")
        bias = sum(num(row, "predicted_ms") - num(row, "measured_ms") for row in wait) / len(wait)
        axis.text(0.97, 0.06, f"bias {bias:+.1f} ms", transform=axis.transAxes, ha="right", fontsize=9)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_xlabel("Predicted (ms/tok)")
        axis.set_ylabel("Measured (ms/tok)")
        axis.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.98), ncol=1, borderaxespad=0)
        save(fig, paths["wait"])

        fig, axis = plt.subplots(figsize=panel, constrained_layout=True)
        kind_color = {"accept": "#1e8449", "reject": "#1f4e79", "conservative": "#b9770e", "unsafe": "#c0392b"}
        kind_label = {"accept": "safe accept", "reject": "reject", "conservative": "conservative", "unsafe": "unsafe"}
        for kind in ("accept", "reject", "conservative", "unsafe"):
            rows = [row for row in risk if row["kind"] == kind]
            axis.scatter(
                [num(row, "predicted_frac") for row in rows],
                [num(row, "measured_frac") for row in rows],
                s=16 if kind == "unsafe" else 16,
                color=kind_color[kind], alpha=0.9 if kind == "unsafe" else 0.45,
                label=f"{kind_label[kind]} ({len(rows)})", zorder=4 if kind == "unsafe" else 2,
            )
        axis.axhline(1, color="#922b21", linestyle="--", linewidth=0.7)
        axis.axvline(1, color="#922b21", linestyle="--", linewidth=0.7)
        axis.set_xlim(0, 1.85)
        axis.set_ylim(0, 2.05)
        axis.set_xlabel(r"$\widehat R$ / limit")
        axis.set_ylabel(r"Measured $\Delta$PPL / $\epsilon$")
        axis.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, 1.32), ncol=2, borderaxespad=0)
        save(fig, paths["risk"])

        fig, axis = plt.subplots(figsize=panel, constrained_layout=True)
        order = ["prefill", "mixed", "decode"]
        values = [num(next(row for row in recall if row["regime"] == regime), "recall") for regime in order]
        axis.bar(order, values, color=[colors[regime] for regime in order], width=0.62)
        for index, value in enumerate(values):
            axis.text(index, value + 0.03, f"{value:.2f}", ha="center", fontsize=9)
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Recall")
        save(fig, paths["recall"])

        fig, axis = plt.subplots(figsize=panel, constrained_layout=True)
        widths = sorted({int(row["W"]) for row in scaling})
        for width, color in zip(widths, ("#1f4e79", "#148f77", "#b9770e", "#6c3483")):
            rows = [row for row in scaling if int(row["W"]) == width]
            axis.plot(
                [num(row, "K") for row in rows], [num(row, "replay_ms") for row in rows],
                marker="o", color=color, label=f"W={width}",
            )
        star = next(row for row in scaling if int(row["K"]) == 8 and int(row["W"]) == 32)
        axis.scatter([8], [num(star, "replay_ms")], marker="*", s=70, color="#c0392b", zorder=5, label="K=8, W=32")
        axis.annotate("0.15 ms", (8, num(star, "replay_ms")), textcoords="offset points", xytext=(6, -1), fontsize=8, color="#c0392b")
        axis.axhline(8.3, color="#922b21", linestyle="--", linewidth=0.7)
        axis.text(62, 4.2, "one layer", color="#922b21", fontsize=9, ha="right")
        axis.axhspan(33, 130, color="#922b21", alpha=0.10, zorder=0)
        axis.text(62, 70, "33–130 ms", color="#922b21", fontsize=9, ha="right", va="center")
        axis.set_ylim(0.05, 180)
        axis.set_yscale("log")
        axis.set_xlabel("Candidate requests K")
        axis.set_ylabel("Replay time (ms)")
        axis.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, 1.25), ncol=3, borderaxespad=0)
        save(fig, paths["replay"])


def plot_robustness(budget: list[dict], path: Path) -> None:
    rows = sorted(
        (row for row in budget if row["phi"] == "1.3"),
        key=lambda row: float(row["copy_gbps"]),
    )
    dynabyte = [num(row, "dynabyte_ms") for row in rows]
    simpler = [num(row, "static_ms") for row in rows]
    labels = ["25*" if row["copy_gbps"] == "25" else row["copy_gbps"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}):
        fig, axis = plt.subplots(figsize=(6.4, 4.8*0.5), constrained_layout=True)
        axis.bar(x - width / 2, dynabyte, width, color="#c0392b", label="DynaByte")
        axis.bar(x + width / 2, simpler, width, color="#5d6d7e", label="Static-Joint")
        # for index, (left, right) in enumerate(zip(dynabyte, simpler)):
        #     axis.text(index, max(left, right) + 10, f"{left:.0f} vs {right:.0f}", ha="center", va="bottom", fontsize=8)
        axis.set_ylim(40.0, 70.0)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Copy rate (GB/s)")
        axis.set_ylabel("P95 TPOT (ms)")
        axis.legend(frameon=False, fontsize=9, loc="upper right")
        save(fig, path)


def tex_cell(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def write_rq2(joint: list[dict], column: list[dict], adaptation: list[dict]) -> str:
    tex_policy = {"dynabyte": r"\systemname", "seq_fr": "Seq-FR", "seq_rf": "Seq-RF"}
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{\textcolor{red}{Draft layout, not a measurement.}",
        r"  Coordinates of Figure~\ref{fig:eval-necessity}(a).",
        r"  Qwen3-30B on the balanced-length slice at $\Phi=1.3$.",
        r"  Brackets are placeholder 95\% intervals.",
        r"  All three rows meet $\Delta$PPL $\leq 2.1$.}",
        r"  \label{tab:eval-rq2-joint}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}lrr@{}}",
        r"    \toprule",
        r"    Policy & $\Delta$PPL & p95 TPOT (ms) \\",
        r"    \midrule",
    ]
    for name in ("dynabyte", "seq_fr", "seq_rf"):
        row = next(item for item in joint if item["policy"] == name)
        lines.append(
            f"    {tex_policy[name]} & {num(row, 'delta_ppl'):.2f} & "
            f"{num(row, 'p95_tpot_ms'):.0f} "
            f"[{num(row, 'ci_low'):.0f}, {num(row, 'ci_high'):.0f}] \\\\"
        )
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{\textcolor{red}{Draft layout, not a measurement.}",
        r"  Feasible fidelity column at $\Phi=1.3$.",
        r"  Fidelity is fixed at 70\% of $B$ and residency receives the remainder.",
        r"  The star is the lowest p95 in this column.",
        r"  These rows are static shares, not Seq-FR or Seq-RF.}",
        r"  \label{tab:eval-rq2-column}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}rrrr@{}}",
        r"    \toprule",
        r"    Look \% & Res.\ \% & $\Delta$PPL & p95 TPOT (ms) \\",
        r"    \midrule",
    ])
    for row in sorted(column, key=lambda item: int(item["look_pct"])):
        tpot = f"{num(row, 'p95_tpot_ms'):.0f}"
        if tpot == "458":
            tpot += r"$^{\star}$"
        lines.append(
            f"    {num(row, 'look_pct'):.0f} & {num(row, 'res_pct'):.0f} & "
            f"{num(row, 'delta_ppl'):.2f} & {tpot} \\\\"
        )
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{\textcolor{red}{Draft layout, not a measurement.}",
        r"  Coordinates of Figure~\ref{fig:eval-necessity}(b).",
        r"  The stream is the balanced-length slice until $t=0$, then the",
        r"  decode-heavy slice.",
        r"  Static-Joint keeps the pre-change allocation.",
        r"  Migration is cumulative host-to-device bytes and is not plotted.",
        r"  Oracle is the settled test-side upper bound and is not plotted.}",
        r"  \label{tab:eval-rq2-shift}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2.6pt}",
        r"  \begin{tabular}{@{}rrrrrrrr@{}}",
        r"    \toprule",
        r"    $t$ (s) & Fid.\ \% & Res.\ \% & Look \% & \systemname & Static-Joint & Oracle & Migr.\ (GB) \\",
        r"    \midrule",
    ])
    for row in adaptation:
        oracle = row["oracle_tpot_ms"]
        oracle_tex = f"{float(oracle):.0f}" if oracle else "--"
        lines.append(
            f"    {num(row, 't_s'):.0f} & {num(row, 'fid_pct'):.0f} & {num(row, 'res_pct'):.0f} & "
            f"{num(row, 'look_pct'):.0f} & {num(row, 'dynabyte_tpot_ms'):.0f} & "
            f"{num(row, 'static_tpot_ms'):.0f} & {oracle_tex} & "
            f"{num(row, 'migration_gb'):.1f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_rq4(wait, risk, recall, scaling) -> str:
    kinds: dict[str, int] = {}
    for row in risk:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    by_recall = {row["regime"]: row["recall"] for row in recall}
    measured = [num(row, "measured_ms") for row in wait]
    predicted = [num(row, "predicted_ms") for row in wait]
    mae = sum(abs(p - m) for p, m in zip(predicted, measured)) / len(wait)
    bias = sum(p - m for p, m in zip(predicted, measured)) / len(wait)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{\textcolor{red}{Draft layout, not a measurement.}",
        r"  Numbers read from Figure~\ref{fig:eval-validation}.",
        r"  Wait error is the mean over the plotted points.",
        r"  Risk counts are the points in panel (b).",
        r"  Recall is panel (c), at envelope depth 3.",
        r"  The starred replay is the implemented point in panel (d).}",
        r"  \label{tab:eval-rq4}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{@{}ll@{}}",
        r"    \toprule",
        r"    Quantity & Draft value \\",
        r"    \midrule",
        f"    Wait MAE / bias (ms/tok) & {mae:.1f} / {bias:+.1f} \\\\",
        f"    Decisions & {len(risk)} \\\\",
        f"    Unsafe accepts & {kinds['unsafe']} \\\\",
        f"    Conservative rejects & {kinds['conservative']} \\\\",
        f"    Recall prefill / mixed / decode & {by_recall['prefill']} / {by_recall['mixed']} / {by_recall['decode']} \\\\",
        r"    \midrule",
    ]
    for row in scaling:
        mark = r"^{\star}" if row["K"] == "8" and row["W"] == "32" else ""
        lines.append(
            f"    Replay $K{{=}}{row['K']}$, $W{{=}}{row['W']}{mark}$ & {num(row, 'replay_ms'):.2f} ms \\\\"
        )
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=TEST1)
    parser.add_argument("--figures", type=Path, default=FIG)
    parser.add_argument("--tables", type=Path, default=TABLES)
    args = parser.parse_args()
    column = read_csv(args.draft / "rq2_heatmap.csv")
    joint = read_csv(args.draft / "rq2_joint.csv")
    adaptation = read_csv(args.draft / "rq2_adaptation.csv")
    ablation = read_csv(args.draft / "rq3_ablation.csv")
    wait = read_csv(args.draft / "rq4_wait.csv")
    risk = read_csv(args.draft / "rq4_risk.csv")
    recall = read_csv(args.draft / "rq4_recall.csv")
    scaling = read_csv(args.draft / "rq4_scaling.csv")
    budget = read_csv(args.draft / "rq5_budget.csv")
    batch = read_csv(args.draft / "rq5_batch.csv")
    plot_necessity(
        joint,
        adaptation,
        args.figures / "eval_necessity_joint.pdf",
        args.figures / "eval_necessity_adapt.pdf",
        args.figures / "eval_necessity_legend.pdf",
    )
    plot_mechanisms(
        ablation,
        {
            "Qwen3-30B": args.figures / "eval_mechanisms_qwen30b.pdf",
            "Qwen3-Next-80B": args.figures / "eval_mechanisms_qwen80b.pdf",
        },
    )
    plot_validation(wait, risk, recall, scaling, {
        "wait": args.figures / "eval_validation_wait.pdf",
        "risk": args.figures / "eval_validation_risk.pdf",
        "recall": args.figures / "eval_validation_recall.pdf",
        "replay": args.figures / "eval_validation_replay.pdf",
    })
    plot_robustness(budget, args.figures / "eval_robustness.pdf")
    args.tables.mkdir(parents=True, exist_ok=True)
    (args.tables / "rq2_necessity.tex").write_text(write_rq2(joint, column, adaptation), encoding="utf-8")
    (args.tables / "rq4_validation.tex").write_text(write_rq4(wait, risk, recall, scaling), encoding="utf-8")
    print("wrote RQ2--RQ5 draft figures and tables")


if __name__ == "__main__":
    main()
