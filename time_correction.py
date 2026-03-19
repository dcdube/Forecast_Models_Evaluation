from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _find_load_file(folder: Path) -> Optional[Path]:
    if not folder.exists() or not folder.is_dir():
        return None

    for candidate in folder.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".csv":
            if candidate.name.lower() == "load_forecast_vs_actual.csv":
                return candidate
    return None


def _find_forecast_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        [
            candidate
            for candidate in folder.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() == ".csv"
            and candidate.name.lower().endswith("_forecast_vs_actual.csv")
        ]
    )


def _find_run_forecast_files(sampling_dir: Path) -> List[Path]:
    if not sampling_dir.exists() or not sampling_dir.is_dir():
        return []

    direct = _find_forecast_files(sampling_dir)
    if direct:
        return direct

    run_files: List[Path] = []
    for child in sorted(path for path in sampling_dir.iterdir() if path.is_dir()):
        run_files.extend(_find_forecast_files(child))
    return run_files


def _get_reference_timelines(
    dataset_dir: Path,
    reference_model: str,
    sampling_name: str,
) -> Dict[str, List[str]]:
    ref_sampling_dir = dataset_dir / reference_model / sampling_name
    ref_files = _find_run_forecast_files(ref_sampling_dir)
    if not ref_files:
        raise FileNotFoundError(
            f"No *_forecast_vs_actual.csv found for reference model '{reference_model}' in {sampling_name}"
        )

    timelines: Dict[str, List[str]] = {}
    for ref_file in ref_files:
        ref_df = pd.read_csv(ref_file)
        lower_map = {col.lower().strip(): col for col in ref_df.columns}
        datetime_col = lower_map.get("datetime")
        if datetime_col is None:
            raise ValueError(f"Reference file has no datetime column: {ref_file}")

        timeline = ref_df[datetime_col].astype(str).tolist()
        if not timeline:
            raise ValueError(f"Reference file has empty datetime timeline: {ref_file}")

        timelines[ref_file.name.lower()] = timeline

    return timelines


def _iter_sampling_run_dirs(sampling_dir: Path) -> List[Path]:
    if not sampling_dir.exists() or not sampling_dir.is_dir():
        return []

    if _find_forecast_files(sampling_dir):
        return [sampling_dir]

    run_dirs: List[Path] = []
    for child in sorted(path for path in sampling_dir.iterdir() if path.is_dir()):
        if _find_forecast_files(child):
            run_dirs.append(child)

    return run_dirs


def _is_numbered_datetime(values: pd.Series) -> bool:
    text = values.astype(str).str.strip()
    if text.empty:
        return False

    numeric_ratio = pd.to_numeric(text, errors="coerce").notna().mean()
    return bool(numeric_ratio >= 0.9)


def correct_model_timelines(
    dataset_dir: Path,
    target_models: Iterable[str],
    reference_model: str,
    sampling_names: Iterable[str],
) -> None:
    changed_files = 0
    skipped_files = 0

    for sampling_name in sampling_names:
        try:
            reference_timelines = _get_reference_timelines(dataset_dir, reference_model, sampling_name)
        except Exception as exc:
            print(f"[WARN] Skipping {sampling_name}: {exc}")
            continue

        print(f"[INFO] Processing {sampling_name} with {len(reference_timelines)} reference timelines")

        for model_name in target_models:
            sampling_dir = dataset_dir / model_name / sampling_name
            run_dirs = _iter_sampling_run_dirs(sampling_dir)
            if not run_dirs:
                print(f"[WARN] No forecast CSV files found for model: {model_name} in {sampling_name}")
                continue

            for run_dir in run_dirs:
                forecast_files = _find_forecast_files(run_dir)
                for forecast_file in forecast_files:
                    try:
                        df = pd.read_csv(forecast_file)
                    except Exception as exc:
                        skipped_files += 1
                        print(f"[WARN] Could not read {forecast_file}: {exc}")
                        continue

                    lower_map = {col.lower().strip(): col for col in df.columns}
                    datetime_col = lower_map.get("datetime")
                    if datetime_col is None:
                        skipped_files += 1
                        print(f"[WARN] Missing datetime column in {forecast_file}")
                        continue

                    timeline = reference_timelines.get(forecast_file.name.lower())
                    if timeline is None:
                        skipped_files += 1
                        print(
                            f"[WARN] No reference timeline for {forecast_file.name} "
                            f"in {sampling_name}"
                        )
                        continue

                    if len(df) != len(timeline):
                        skipped_files += 1
                        print(
                            f"[WARN] Length mismatch in {forecast_file}: rows={len(df)} "
                            f"vs reference={len(timeline)}"
                        )
                        continue

                    original = df[datetime_col].astype(str)
                    if not _is_numbered_datetime(df[datetime_col]):
                        print(f"[OK] Datetime already non-numeric: {forecast_file}")
                        continue

                    replacement = pd.Series(timeline, index=df.index, dtype="object")
                    if original.equals(replacement):
                        print(f"[OK] Already corrected: {forecast_file}")
                        continue

                    df[datetime_col] = replacement
                    df.to_csv(forecast_file, index=False)
                    changed_files += 1
                    print(f"[FIXED] Updated datetime timeline: {forecast_file}")

    print(f"Done. Corrected files: {changed_files}, skipped files: {skipped_files}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct datetime columns in model load CSV files to match a reference timeline."
    )
    parser.add_argument(
        "--dataset",
        default="london_zonnedael",
        help="Dataset suffix used in results/results_<dataset>.",
    )
    parser.add_argument(
        "--reference-model",
        default="AutoARIMA",
        help="Model used as source for the datetime timeline.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["KNNRegression", "LightGBM", "NaiveMovingAverage"],
        help="Target model folders to patch.",
    )
    parser.add_argument(
        "--samplings",
        nargs="+",
        default=["Sampling_25", "Sampling_33", "Sampling_50", "Sampling_100"],
        help="Sampling folder names to patch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "results" / f"results_{args.dataset}"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset results folder not found: {dataset_dir}")

    correct_model_timelines(
        dataset_dir=dataset_dir,
        target_models=args.models,
        reference_model=args.reference_model,
        sampling_names=args.samplings,
    )


if __name__ == "__main__":
    main()
