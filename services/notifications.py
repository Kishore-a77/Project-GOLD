"""Extensible pipeline notification hooks.

The default implementation logs only. No external messages are sent until a
notification provider is intentionally configured, keeping CI and local runs
safe while giving future email/Slack/Telegram integrations one stable seam.
"""

import logging

logger = logging.getLogger("project_gold.notifications")


def notify_pipeline_failure(stage, error, **context):
    """Record a failure notification event without contacting external users."""
    logger.error("Pipeline failure notification: stage=%s error=%s context=%s", stage, error, context)
    return False


def notify_pipeline_success(**context):
    """Record a success notification event without contacting external users."""
    logger.info("Pipeline success notification: context=%s", context)
    return False
