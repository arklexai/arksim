# SPDX-License-Identifier: Apache-2.0
"""Arkdock attribute-discovery endpoints.

Exposes the two Python-side routes the Go backend calls for the adaptive
scenario discovery wizard (spec §17.4 and §17.6):

    POST /arkdock/attribute-discovery/run
    POST /arkdock/attribute-discovery/{run_id}/cancel

The dispatch endpoint starts a background thread that:
  1. Downloads conversations.json from S3 (or reads a local path in dev mode).
  2. Runs GoalDiscoveryPipeline with knobs from discovery_config.
  3. Formats the result as the artifacts shape (spec §17.9).
  4. Writes status + artifacts back to the attribute_discovery_run row via a
     direct MySQL connection (requires ARKDOCK_DB_* env vars) or logs if the
     DB is not configured (dev / test mode).

S3 download uses boto3 when ARKDOCK_S3_BUCKET is set; otherwise the file_key
is resolved as a local filesystem path (dev mode only).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from arksim.scenario.goal_discovery.arkdock import (
    ArkdockDiscoveryConfig,
    to_arkdock_artifacts,
)
from arksim.scenario.goal_discovery.models import ConversationInput

logger = logging.getLogger(__name__)

router = APIRouter(tags=["arkdock-discovery"])

# ── In-process job registry ──────────────────────────────────────────────────
# Maps task_id -> {"run_id": str, "cancelled": bool, "thread": Thread}
# Entries are kept until the process restarts; this is sufficient because Go
# only calls cancel while the job is running (spec §17.6).

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── Request / response shapes ────────────────────────────────────────────────


class DispatchRequest(BaseModel):
    run_id: str
    arkdock_organization_id: str
    file_key: str
    discovery_config: dict[str, Any] = {}


class CancelRequest(BaseModel):
    task_id: str


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/attribute-discovery/run")
def dispatch_discovery(body: DispatchRequest) -> dict:
    """Enqueue a discovery job and return a task_id for cancellation.

    Called by the Go backend after it persists the attribute_discovery_run row
    (spec §17.4 step 4). Returns {"status": true, "task_id": "..."}.
    """
    task_id = str(uuid.uuid4())

    logger.info(
        "POST /arkdock/attribute-discovery/run received: run_id=%s org=%s file_key=%s config=%s",
        body.run_id,
        body.arkdock_organization_id,
        body.file_key,
        body.discovery_config,
    )

    cfg = ArkdockDiscoveryConfig.model_validate(body.discovery_config)
    logger.info(
        "discovery config parsed: method=goal_discovery clustering=%s k_range=(2,%d) min_support=%d model=%s",
        cfg.clustering_method,
        cfg.approved_top_k,
        cfg.min_support,
        cfg.llm_model,
    )

    entry: dict = {"run_id": body.run_id, "cancelled": False, "thread": None}
    with _jobs_lock:
        _jobs[task_id] = entry

    thread = threading.Thread(
        target=_run_discovery,
        args=(task_id, body.run_id, body.arkdock_organization_id, body.file_key, cfg),
        daemon=True,
    )
    entry["thread"] = thread
    thread.start()

    logger.info(
        "arkdock discovery dispatched: run_id=%s task_id=%s",
        body.run_id,
        task_id,
    )
    return {"status": True, "task_id": task_id}


@router.post("/attribute-discovery/{run_id}/cancel")
def cancel_discovery(run_id: str, body: CancelRequest) -> dict:
    """Mark a job cancelled so the background thread exits early.

    Called by Go after it sets the DB row to cancelled (spec §17.6 step 3).
    Best-effort: if the Celery/thread task has already finished, this is a
    no-op (the thread is gone).
    """
    with _jobs_lock:
        entry = _jobs.get(body.task_id)

    if entry is None:
        # The task may have already completed or was never registered.
        logger.warning(
            "arkdock cancel: unknown task_id=%s for run_id=%s", body.task_id, run_id
        )
        raise HTTPException(status_code=404, detail="task not found")

    if entry["run_id"] != run_id:
        raise HTTPException(status_code=400, detail="task_id / run_id mismatch")

    with _jobs_lock:
        entry["cancelled"] = True
    logger.info(
        "arkdock discovery cancel requested: run_id=%s task_id=%s", run_id, body.task_id
    )
    return {"status": True}


# ── Background worker ────────────────────────────────────────────────────────


def _run_discovery(
    task_id: str,
    run_id: str,
    org_id: str,
    file_key: str,
    cfg: ArkdockDiscoveryConfig,
) -> None:
    """Background thread: download -> discover -> write artifacts."""
    try:
        # Check cancellation under lock before any DB write to avoid overwriting
        # a 'cancelled' status that Go may have set before this thread started.
        with _jobs_lock:
            entry = _jobs.get(task_id, {})
            if entry.get("cancelled"):
                logger.info(
                    "arkdock discovery cancelled before start: run_id=%s", run_id
                )
                return

        _db_update_status(run_id, "running")

        # Step 1: load conversations
        conversations = _load_conversations(file_key)
        logger.info(
            "arkdock discovery loaded %d conversations: run_id=%s",
            len(conversations),
            run_id,
        )

        with _jobs_lock:
            if entry.get("cancelled"):
                logger.info("arkdock discovery cancelled after load: run_id=%s", run_id)
                return

        # Step 2: run goal discovery
        pipeline = cfg.to_pipeline()
        logger.info(
            "goal discovery pipeline starting: run_id=%s conversations=%d embedding=%s/%s clustering=%s",
            run_id,
            len(conversations),
            pipeline.embedding_provider,
            pipeline.embedding_model or "default",
            pipeline.clustering_method,
        )
        result = pipeline.discover(conversations)
        logger.info(
            "goal discovery pipeline complete: run_id=%s goals=%d method=%s",
            run_id,
            len(result.goals),
            result.method,
        )

        # Step 3: format and persist artifacts. Check cancellation first to avoid
        # overwriting a 'cancelled' status that Go set while the pipeline ran.
        with _jobs_lock:
            if entry.get("cancelled"):
                logger.info(
                    "arkdock discovery cancelled after pipeline: run_id=%s", run_id
                )
                return

        artifacts = to_arkdock_artifacts(result)
        _db_write_success(run_id, artifacts)

    except Exception as exc:
        logger.exception("arkdock discovery failed: run_id=%s error=%s", run_id, exc)
        _db_update_status(run_id, "error", error_message=str(exc)[:500])
    finally:
        with _jobs_lock:
            _jobs.pop(task_id, None)


# ── File loading ─────────────────────────────────────────────────────────────


def _load_conversations(file_key: str) -> list[ConversationInput]:
    """Download and parse the conversations file.

    Resolution order:
      1. If ARKDOCK_S3_BUCKET is set, download from S3 using boto3.
      2. Otherwise treat file_key as a local filesystem path (dev mode).

    The file must be a JSON array whose items follow the conversations-sample
    schema (id, messages, ...).
    """
    raw: bytes

    s3_bucket = os.getenv("ARKDOCK_S3_BUCKET")
    if s3_bucket:
        raw = _s3_download(s3_bucket, file_key)
    else:
        path = Path(file_key)
        if not path.exists():
            raise FileNotFoundError(
                f"file_key not found locally and ARKDOCK_S3_BUCKET is not set: {file_key}"
            )
        raw = path.read_bytes()

    records: list[dict] = json.loads(raw)
    return [_parse_record(r) for r in records]


def _parse_record(record: dict) -> ConversationInput:
    """Detect format by key presence and dispatch to the right factory."""
    if "messages" in record:
        return ConversationInput.from_conversations_record(record)
    return ConversationInput.from_flat_record(record)


def _s3_download(bucket: str, key: str) -> bytes:
    try:
        import boto3  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 downloads. "
            "Install it with: pip install arksim[arkdock]"
        ) from e

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


# ── DB write-back ─────────────────────────────────────────────────────────────
# Writes status transitions and artifacts directly to the Go backend's MySQL DB.
# Requires env vars: ARKDOCK_DB_HOST, ARKDOCK_DB_PORT, ARKDOCK_DB_USER,
#                    ARKDOCK_DB_PASSWORD, ARKDOCK_DB_NAME.
# Without those vars the calls are no-ops and a warning is logged. This is
# useful for running the discovery pipeline locally without a backend DB.


def _db_connection() -> object | None:
    """Return a pymysql connection, or None if DB env vars are absent."""
    host = os.getenv("ARKDOCK_DB_HOST")
    if not host:
        return None
    try:
        import pymysql  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "pymysql is required for DB write-back. "
            "Install it with: pip install arksim[arkdock]"
        ) from e

    return pymysql.connect(
        host=host,
        port=int(os.getenv("ARKDOCK_DB_PORT", "3306")),
        user=os.getenv("ARKDOCK_DB_USER", ""),
        password=os.getenv("ARKDOCK_DB_PASSWORD", ""),
        database=os.getenv("ARKDOCK_DB_NAME", ""),
        autocommit=True,
    )


_STATUS_IDS = {
    "pending": 1,
    "running": 2,
    "success": 3,
    "error": 4,
    "cancelled": 5,
}


def _db_update_status(
    run_id: str, status: str, error_message: str | None = None
) -> None:
    conn = _db_connection()
    if conn is None:
        logger.warning(
            "ARKDOCK_DB_HOST not set; skipping DB status update: run_id=%s status=%s",
            run_id,
            status,
        )
        return

    status_id = _STATUS_IDS.get(status, 4)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE attribute_discovery_run
                   SET async_task_status_id = %s,
                       error_message = %s,
                       modified_at = NOW()
                 WHERE id = %s
                """,
                (status_id, error_message, run_id),
            )
    finally:
        conn.close()


def _db_write_success(run_id: str, artifacts: dict[str, Any]) -> None:
    conn = _db_connection()
    if conn is None:
        logger.warning(
            "ARKDOCK_DB_HOST not set; discovery artifacts not persisted: run_id=%s",
            run_id,
        )
        logger.info("artifacts: %s", json.dumps(artifacts, indent=2))
        return

    artifacts_json = json.dumps(artifacts)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE attribute_discovery_run
                   SET async_task_status_id = %s,
                       artifacts = %s,
                       error_message = NULL,
                       modified_at = NOW()
                 WHERE id = %s
                """,
                (_STATUS_IDS["success"], artifacts_json, run_id),
            )
    finally:
        conn.close()
