#!/usr/bin/env python3
"""Dispatch a SoccerNet GSR evaluation to a Lightning AI Studio.

This runs inside the GitHub Actions runner. It does NOT do any heavy compute:
it starts a Lightning Studio, syncs this repository onto it at the exact commit
that triggered the workflow, then submits the evaluation as an *asynchronous*
GPU Job (so the Action returns immediately and is not blocked by the long run).

Required environment variables (set as GitHub Actions secrets / variables):
    LIGHTNING_USER_ID    Lightning account user id      (Settings > Keys)  [secret]
    LIGHTNING_API_KEY    Lightning API key              (Settings > Keys)  [secret]
    LIGHTNING_USER       Lightning username                                [secret/var]
    LIGHTNING_TEAMSPACE  Teamspace that owns the Studio                    [secret/var]
    LIGHTNING_STUDIO     Studio name to run in (created if missing)        [var]

Provided automatically by GitHub Actions:
    GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_SHA, GITHUB_RUN_ID

NOTE: method/enum names below follow the documented lightning_sdk API. Confirm
they match your installed lightning_sdk version when you wire up Lightning.
"""

from __future__ import annotations

import argparse
import os
import sys

from lightning_sdk import Job, Machine, Studio


def env(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if required and not value:
        sys.exit(f"ERROR: missing required environment variable: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="both",
                        choices=["valid", "test", "both"],
                        help="Which SoccerNet GSR split(s) to evaluate.")
    parser.add_argument("--nvid", default="-1",
                        help="Number of videos per split (-1 = all).")
    parser.add_argument("--machine", default="A10",
                        help="Lightning Machine name, e.g. T4, L4, A10, A100.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = "valid test" if args.eval_set == "both" else args.eval_set

    try:
        machine = getattr(Machine, args.machine)
    except AttributeError:
        sys.exit(f"ERROR: unknown Lightning machine '{args.machine}'.")

    server = env("GITHUB_SERVER_URL", required=False, default="https://github.com")
    repo = env("GITHUB_REPOSITORY")            # e.g. "owner/sn-gamestate"
    sha = env("GITHUB_SHA", required=False, default="main")
    run_id = env("GITHUB_RUN_ID", required=False, default="manual")
    clone_url = f"{server}/{repo}.git"
    repo_dir = repo.split("/")[-1]

    print(f"Starting Studio '{os.environ.get('LIGHTNING_STUDIO')}' ...")
    studio = Studio(
        name=env("LIGHTNING_STUDIO"),
        teamspace=env("LIGHTNING_TEAMSPACE"),
        user=env("LIGHTNING_USER"),
        create_ok=True,
    )
    studio.start()

    # Single self-contained command: sync repo at this commit, then evaluate.
    # Guarded clone so re-runs reuse the existing checkout (and cached dataset/models).
    job_command = (
        "set -e; "
        f"if [ -d {repo_dir}/.git ]; then "
        f"  cd {repo_dir} && git fetch --all --quiet && git checkout --quiet {sha}; "
        f"else "
        f"  git clone --quiet {clone_url} {repo_dir} && cd {repo_dir} && git checkout --quiet {sha}; "
        f"fi; "
        f"SPLITS='{splits}' NVID='{args.nvid}' bash scripts/lightning_eval.sh"
    )

    print(f"Submitting asynchronous GPU job on machine '{args.machine}' ...")
    job = Job.run(
        name=f"gsr-eval-{run_id}",
        command=job_command,
        studio=studio,
        machine=machine,
    )

    print(f"Submitted Lightning job: {getattr(job, 'name', 'gsr-eval')}")
    try:
        print(f"Track progress here: {job.link}")
    except Exception:  # noqa: BLE001 - link is best effort
        pass

    # The job runs on its own allocated machine, so the orchestration Studio
    # can be stopped immediately to avoid extra cost.
    studio.stop()
    print("Studio stopped. The evaluation job continues running on Lightning.")


if __name__ == "__main__":
    main()
