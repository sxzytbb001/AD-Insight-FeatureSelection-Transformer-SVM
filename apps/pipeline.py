import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, List

from apps import config


@dataclass(frozen=True)
class PipelineOptions:
    skip_feature_selection: bool = False
    skip_transformer: bool = False
    skip_svm: bool = False
    skip_external_validation: bool = False
    skip_statistics: bool = False
    continue_on_error: bool = False


@dataclass(frozen=True)
class PipelineStep:
    name: str
    run: Callable[[], object]
    enabled: bool = True


def _safe_print(message=""):
    """Print safely on Windows terminals with limited encodings."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_message = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_message)


def _run_feature_selection():
    from apps.preprocessing.feature_selection import feature_selection

    return feature_selection()


def _run_transformer_training():
    from apps.training.train_transformer import train_transformer

    return train_transformer()


def _run_svm_training():
    from apps.training.train_svm import train_svm

    return train_svm()


def _run_external_validation():
    from apps.evaluation.external_validation import evaluate_external_models

    return evaluate_external_models()


def _run_statistical_analysis():
    from apps.analysis.statistical_analysis import run_statistical_analysis

    return run_statistical_analysis()


def build_pipeline_steps(options: PipelineOptions) -> List[PipelineStep]:
    return [
        PipelineStep("Feature selection", _run_feature_selection, not options.skip_feature_selection),
        PipelineStep("Transformer training", _run_transformer_training, not options.skip_transformer),
        PipelineStep("SVM training", _run_svm_training, not options.skip_svm),
        PipelineStep("External validation", _run_external_validation, not options.skip_external_validation),
        PipelineStep("Statistical analysis", _run_statistical_analysis, not options.skip_statistics),
    ]


def _run_step(step: PipelineStep):
    if not step.enabled:
        _safe_print(f"[SKIP] {step.name}")
        return {"name": step.name, "status": "skipped", "seconds": 0.0}

    _safe_print(f"\n>>> Start: {step.name}")
    start_time = time.time()
    try:
        step.run()
        elapsed = time.time() - start_time
        _safe_print(f"[OK] {step.name} ({elapsed:.1f}s)")
        return {"name": step.name, "status": "success", "seconds": elapsed}
    except Exception as exc:
        elapsed = time.time() - start_time
        _safe_print(f"[ERR] {step.name} ({elapsed:.1f}s)")
        _safe_print(f"Reason: {exc}")
        traceback.print_exc()
        return {"name": step.name, "status": "failed", "seconds": elapsed, "error": str(exc)}


def run_pipeline(options: PipelineOptions):
    config.ensure_dirs()

    _safe_print("=" * 72)
    _safe_print("Gene Expression Classification Pipeline")
    _safe_print("=" * 72)
    _safe_print(f"Project root: {config.BASE_DIR}")
    _safe_print(f"Training data: {config.TRAIN_DATA_DIR}")
    _safe_print(f"Results dir: {config.RESULTS_DIR}")

    results = []
    for step in build_pipeline_steps(options):
        outcome = _run_step(step)
        results.append(outcome)

        if outcome["status"] == "failed" and not options.continue_on_error:
            _safe_print("\nPipeline stopped because a step failed.")
            break

    succeeded = sum(item["status"] == "success" for item in results)
    failed = [item for item in results if item["status"] == "failed"]
    skipped = sum(item["status"] == "skipped" for item in results)

    _safe_print("\n" + "=" * 72)
    _safe_print("Pipeline summary")
    _safe_print("=" * 72)
    for item in results:
        status = item["status"]
        icon = "[OK]" if status == "success" else "[ERR]" if status == "failed" else "[SKIP]"
        _safe_print(f"{icon} {item['name']} - {status}")

    _safe_print("-" * 72)
    _safe_print(f"Success: {succeeded} | Failed: {len(failed)} | Skipped: {skipped}")
    _safe_print(f"Results dir: {config.RESULTS_DIR}")
    _safe_print("=" * 72)

    return len(failed) == 0


def run_full_pipeline(
    skip_feature_selection=False,
    skip_transformer=False,
    skip_svm=False,
    skip_external_validation=False,
    skip_statistics=False,
    continue_on_error=False,
):
    """Backward-compatible pipeline entry point used by older callers."""
    return run_pipeline(
        PipelineOptions(
            skip_feature_selection=skip_feature_selection,
            skip_transformer=skip_transformer,
            skip_svm=skip_svm,
            skip_external_validation=skip_external_validation,
            skip_statistics=skip_statistics,
            continue_on_error=continue_on_error,
        )
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the gene expression classification pipeline.")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Skip feature selection.")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip Transformer training.")
    parser.add_argument("--skip-svm", action="store_true", help="Skip SVM training.")
    parser.add_argument("--skip-external-validation", action="store_true", help="Skip external validation.")
    parser.add_argument("--skip-statistics", action="store_true", help="Skip statistical analysis.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the remaining steps after a failure.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    ok = run_full_pipeline(
        skip_feature_selection=args.skip_feature_selection,
        skip_transformer=args.skip_transformer,
        skip_svm=args.skip_svm,
        skip_external_validation=args.skip_external_validation,
        skip_statistics=args.skip_statistics,
        continue_on_error=args.continue_on_error,
    )

    if not ok:
        raise SystemExit(1)
