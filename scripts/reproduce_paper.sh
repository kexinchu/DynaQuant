#!/usr/bin/env bash
# Safe entry point for reproducible experiment commands.
#
# This wrapper intentionally does not iterate over method names.  Earlier
# versions produced several differently named JSON files by invoking the same
# unconfigured model each time.  Every invocation now carries one explicit,
# verifiable method and checkpoint through to the artifact.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${command_name}" in
    quality)
        python -m dynaexq.experiments.eval_quality "$@"
        ;;
    perf)
        python -m dynaexq.experiments.eval_perf "$@"
        ;;
    moe-infinity)
        python scripts/benchmark_moe_infinity.py "$@"
        ;;
    shift)
        python -m dynaexq.experiments.run_shift "$@"
        ;;
    dynamic)
        python -m dynaexq.experiments.eval_dynamic "$@"
        ;;
    compare-quality)
        python scripts/compare_quality_artifacts.py "$@"
        ;;
    build-ppl-curve)
        python scripts/build_perplexity_curve.py "$@"
        ;;
    activation-density)
        python scripts/collect_activation_density.py "$@"
        ;;
    routing-trace)
        python scripts/collect_routing_active_set_trace.py "$@"
        ;;
    offload-waiting)
        python scripts/benchmark_blocking_offload.py "$@"
        ;;
    test)
        python -m pytest "$@"
        ;;
    help|-h|--help)
        echo "Usage:"
        echo "  bash scripts/reproduce_paper.sh quality <eval_quality arguments>"
        echo "  bash scripts/reproduce_paper.sh perf    <eval_perf arguments>"
        echo "  bash scripts/reproduce_paper.sh moe-infinity <baseline arguments>"
        echo "  bash scripts/reproduce_paper.sh shift   <run_shift arguments>"
        echo "  bash scripts/reproduce_paper.sh dynamic <eval_dynamic arguments>"
        echo "  bash scripts/reproduce_paper.sh compare-quality <comparison arguments>"
        echo "  bash scripts/reproduce_paper.sh build-ppl-curve <curve arguments>"
        echo "  bash scripts/reproduce_paper.sh activation-density <collector arguments>"
        echo "  bash scripts/reproduce_paper.sh routing-trace <collector arguments>"
        echo "  bash scripts/reproduce_paper.sh offload-waiting <benchmark arguments>"
        echo "  bash scripts/reproduce_paper.sh test [pytest arguments]"
        echo
        echo "Quality/perf labels cannot activate DynaExQ or an external offload runtime."
        echo "Use 'dynamic' for validated DynaExQ quality, performance, and ablation runs."
        echo "Use 'moe-infinity' for the pinned official Qwen3-30B baseline."
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        exit 2
        ;;
esac
