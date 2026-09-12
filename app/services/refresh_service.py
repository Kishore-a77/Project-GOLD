"""
app/services/refresh_service.py

Orchestrates automatic data refresh when dashboard data is STALE or MISSING.
Supports:
1. Priority 1: GitHub Actions workflow dispatch if GITHUB_TOKEN is available.
2. Priority 2: Safe local background runner with process-level and lockfile deduplication.
3. Checking active pipeline runs to prevent concurrent overlapping executions.
4. Bounded polling for fresh predictions.
5. Structured logging without exposing tokens or secrets.
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple

import requests

from app.db.supabase_client import get_supabase_client, execute_with_retry
from app.services.database_service import fetch_pipeline_runs, check_system_readiness, STATE_READY

logger = logging.getLogger("refresh_service")

ROOT_DIR = Path(__file__).resolve().parents[2]
LOCK_FILE = ROOT_DIR / ".pipeline_run.lock"
_IN_PROCESS_LOCK = threading.Lock()
_LAST_TRIGGER_TIME = 0.0
TRIGGER_COOLDOWN_SECONDS = 300.0  # At least 5 minutes between refresh triggers


def is_pipeline_currently_running() -> Tuple[bool, Optional[str]]:
    """
    Check if a pipeline run is currently in progress.
    Checks:
    1. Lock file existence and freshness.
    2. Supabase pipeline_runs table for recent running status.
    """
    # 1. Check lock file
    if LOCK_FILE.exists():
        try:
            mtime = LOCK_FILE.stat().st_mtime
            age = time.time() - mtime
            if age < 1800:  # 30 minutes max lock duration
                return True, f"Local pipeline process active (age: {int(age)}s)"
            else:
                # Stale lock
                try:
                    LOCK_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    # 2. Check Supabase pipeline_runs
    try:
        latest_run, _ = fetch_pipeline_runs()
        if latest_run and latest_run.get("status") in ("running", "in_progress"):
            started = latest_run.get("started_at")
            if started:
                now_utc = datetime.now(timezone.utc)
                if hasattr(started, "tzinfo") and started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                age = (now_utc - started).total_seconds()
                if age < 1800:
                    return True, f"Supabase pipeline run in progress (started {int(age)}s ago)"
    except Exception as e:
        logger.warning("Could not check pipeline status in Supabase: %s", e)

    # Supabase may not show a newly dispatched run immediately. Ask GitHub
    # directly as a second concurrency guard before dispatching another run.
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        active, reason = _github_workflow_currently_running(token)
        if active:
            return True, reason

    return False, None


def _github_workflow_currently_running(token: str) -> Tuple[bool, Optional[str]]:
    """Return whether daily_pipeline.yml is queued or running on main."""
    repo = os.getenv("GITHUB_REPOSITORY", "Kishore-a77/Project-GOLD")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/daily_pipeline.yml/runs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        response = requests.get(
            url,
            headers=headers,
            params={"branch": "main", "per_page": 10},
            timeout=10,
        )
        response.raise_for_status()
        for run in response.json().get("workflow_runs", []):
            if run.get("status") in ("queued", "in_progress", "requested", "waiting"):
                return True, f"GitHub Actions run already {run.get('status')}"
    except Exception as exc:
        # An API outage must not be treated as proof that no run exists;
        # conservatively pause dispatch to avoid overlapping executions.
        logger.warning("Could not verify active GitHub Actions runs: %s", exc)
        return True, "GitHub Actions status could not be verified; refresh paused"
    return False, None


def trigger_github_workflow_dispatch() -> Tuple[bool, str]:
    """
    Attempt to trigger the daily pipeline via GitHub Actions REST API.
    Requires GITHUB_TOKEN (or GH_TOKEN) and GITHUB_REPOSITORY (e.g. 'Kishore-a77/Project-GOLD').
    """
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "Kishore-a77/Project-GOLD")

    if not token:
        return False, "GITHUB_TOKEN not configured"

    url = f"https://api.github.com/repos/{repo}/actions/workflows/daily_pipeline.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = {"ref": "main"}

    try:
        logger.info("Triggering GitHub Actions workflow daily_pipeline.yml for %s", repo)
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        if resp.status_code in (201, 204):
            logger.info("GitHub Actions workflow successfully dispatched")
            return True, "GitHub Actions workflow dispatched successfully"
        else:
            msg = f"GitHub API error: HTTP {resp.status_code}"
            logger.warning("%s: %s", msg, resp.text[:200])
            return False, msg
    except Exception as e:
        logger.error("Failed to trigger GitHub Actions workflow: %s", e)
        return False, str(e)


def trigger_local_background_pipeline() -> Tuple[bool, str]:
    """
    Trigger the existing run_daily_pipeline.py script as a detached background subprocess.
    Ensures single-instance execution via process lock and lockfile.
    """
    global _LAST_TRIGGER_TIME

    with _IN_PROCESS_LOCK:
        now = time.time()
        if now - _LAST_TRIGGER_TIME < TRIGGER_COOLDOWN_SECONDS:
            return False, f"Refresh cooldown active. Please wait {int(TRIGGER_COOLDOWN_SECONDS - (now - _LAST_TRIGGER_TIME))}s"

        pipeline_script = ROOT_DIR / "run_daily_pipeline.py"
        if not pipeline_script.exists():
            return False, "run_daily_pipeline.py not found"

        # Create lock file
        try:
            LOCK_FILE.write_text(f"started_at={datetime.now(timezone.utc).isoformat()}\npid={os.getpid()}")
        except Exception:
            pass

        _LAST_TRIGGER_TIME = now

        try:
            logger.info("Launching background daily pipeline: %s", pipeline_script)
            # Launch in background, non-blocking
            if sys.platform == "win32":
                # DETACHED_PROCESS on Windows
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                subprocess.Popen(
                    [sys.executable, str(pipeline_script)],
                    cwd=str(ROOT_DIR),
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    close_fds=True,
                )
            else:
                subprocess.Popen(
                    [sys.executable, str(pipeline_script)],
                    cwd=str(ROOT_DIR),
                    start_new_session=True,
                    close_fds=True,
                )
            logger.info("Background pipeline process successfully initiated")
            return True, "Background pipeline initiated"
        except Exception as e:
            logger.error("Failed to spawn background pipeline: %s", e)
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return False, str(e)


def initiate_automatic_refresh() -> Tuple[bool, str]:
    """
    Automatically trigger refresh mechanism without manual user clicks.
    Priority:
    1. If already running -> do not duplicate.
    2. Try GitHub Actions dispatch if token present.
    3. Fallback to local background pipeline runner.
    """
    # 1. Concurrency Guard
    running, reason = is_pipeline_currently_running()
    if running:
        logger.info("Automatic refresh skipped: %s", reason)
        return True, f"Pipeline already in progress: {reason}"

    # 2. Try GitHub Actions Dispatch
    gh_success, gh_msg = trigger_github_workflow_dispatch()
    if gh_success:
        return True, gh_msg

    # Local execution is an explicit development-only opt-in. A deployed
    # dashboard must never silently run the heavy forecast pipeline itself.
    allow_local = os.getenv("GOLD_ALLOW_LOCAL_PIPELINE_REFRESH", "").lower() in {
        "1", "true", "yes"
    }
    if allow_local and not os.getenv("STREAMLIT_CLOUD"):
        local_success, local_msg = trigger_local_background_pipeline()
        return local_success, local_msg

    return False, (
        "GitHub Actions refresh unavailable: GITHUB_TOKEN is not configured. "
        "Set GITHUB_TOKEN for automatic production refresh, or explicitly set "
        "GOLD_ALLOW_LOCAL_PIPELINE_REFRESH=1 for local development."
    )


def poll_for_fresh_data(
    max_wait_seconds: int = 30,
    check_interval: float = 3.0,
    initial_gold_date: Optional[Any] = None,
    initial_total_preds: int = 0,
    initial_pred_generated_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Poll Supabase for newly generated predictions with bounded timeout.

    Returns the latest readiness report once fresh data appears or timeout expires.
    """
    logger.info("Polling for fresh data (timeout=%ds, interval=%.1fs)...", max_wait_seconds, check_interval)
    start_time = time.time()

    while time.time() - start_time < max_wait_seconds:
        readiness = check_system_readiness()
        state = readiness.get("state")

        # If system is ready and has more/newer predictions than before
        if state == STATE_READY:
            curr_date = readiness.get("latest_gold_date")
            curr_preds = readiness.get("total_predictions", 0)
            curr_generated = readiness.get("pred_generated_date")
            generation_changed = (
                initial_pred_generated_date is None
                or curr_generated is not None
                and curr_generated > initial_pred_generated_date
            )
            if (
                initial_gold_date is None
                or curr_date > initial_gold_date
                or curr_preds > initial_total_preds
                or generation_changed
            ):
                logger.info("Fresh data detected during polling! (gold_date=%s, preds=%d)", curr_date, curr_preds)
                return readiness

        time.sleep(check_interval)

    logger.info("Polling finished (timeout reached).")
    return check_system_readiness()
