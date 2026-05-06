from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _run_json(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return json.loads(result.stdout)


def _completed_condition(execution: dict) -> dict | None:
    conditions = execution.get("status", {}).get("conditions", [])
    for condition in conditions:
        if condition.get("type") == "Completed":
            return condition
    return None


def _check_gcs_object(path: str) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "ls", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Cloud Run Job execution completion for CLI batch runs"
    )
    parser.add_argument("execution_name", help="Cloud Run Job execution name")
    parser.add_argument(
        "--job-name",
        default="cli-alpha-council-job",
        help="Cloud Run Job name",
    )
    parser.add_argument(
        "--region",
        default="asia-east1",
        help="GCP region of the Cloud Run Job",
    )
    parser.add_argument(
        "--project",
        default="dassa-lab",
        help="GCP project ID",
    )
    parser.add_argument(
        "--gcs-object",
        help="Optional GCS object path to verify after execution completes",
    )
    args = parser.parse_args()

    execution = _run_json(
        [
            "gcloud",
            "run",
            "jobs",
            "executions",
            "describe",
            args.execution_name,
            "--region",
            args.region,
            "--project",
            args.project,
            "--format=json",
        ]
    )

    completed = _completed_condition(execution)
    status = completed.get("status") if completed else "Unknown"
    message = completed.get("message", "") if completed else "Completed condition missing"

    summary = {
        "execution_name": args.execution_name,
        "job_name": args.job_name,
        "project": args.project,
        "region": args.region,
        "completed_status": status,
        "message": message,
        "log_uri": execution.get("status", {}).get("logUri", ""),
    }

    if args.gcs_object:
        summary["gcs_object"] = args.gcs_object
        summary["gcs_exists"] = _check_gcs_object(args.gcs_object)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if status == "True":
        if args.gcs_object and not summary["gcs_exists"]:
            return 2
        return 0

    if status == "Unknown":
        return 3

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
