"""Automatic result submission, kept outside the editable notebook cells.

The member never retypes a score. ``Challenge`` captures the result and the code that
produced it, attaches the Google account email reported by Colab, and posts one row to
the organizer's spreadsheet.

Storage sits behind :func:`post_submission` so the payload builder never learns which
backend is in use. Swapping Google Sheets for a database later replaces that one
function and leaves :func:`build_payload` and its tests untouched.
"""

from datetime import datetime, timezone
from pathlib import Path
import inspect
import json
import os
import random
import time
import urllib.request
from urllib.parse import urlencode

DEFAULT_ENDPOINT = "https://script.google.com/macros/s/AKfycbzVCs48MKOEC-OzXN4zU5JjP5EtTsU1JS9Q-BltFHUZaJdpcTRHWQeDVqJMCJkir0Yo/exec"
DEFAULT_WORKSHOP_ID = "2026-fall-mandelbrot-beta"
DEFAULT_TOKEN = "e3fa904c82001b3cdaa89a107bc962a4951c6c833cb20a5a8926e18f43936b12"
FALLBACK_DIR = "outputs/submissions"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

POST_ATTEMPTS = 3
POST_TIMEOUT_SECONDS = 8.0
USERINFO_TIMEOUT_SECONDS = 10.0

_cached_member = None


# The endpoint travels inside the member's own Colab runtime, so it is discoverable no
# matter where it is stored. These overrides exist for routing (test vs. live, one
# workshop vs. the next), not for secrecy; the Apps Script validates every request.
def endpoint():
    return os.environ.get("MANDELBROT_SUBMIT_URL", DEFAULT_ENDPOINT)


def workshop_id():
    return os.environ.get("MANDELBROT_WORKSHOP_ID", DEFAULT_WORKSHOP_ID)


def shared_token():
    return os.environ.get("MANDELBROT_SUBMIT_TOKEN", DEFAULT_TOKEN)


def dashboard_url():
    """Return the live, public-safe progress dashboard for this workshop."""
    return f"{endpoint()}?{urlencode({'workshop': workshop_id()})}"


def fallback_dir():
    return Path(os.environ.get("MANDELBROT_SUBMIT_DIR", FALLBACK_DIR))


def _ask(prompt, attempts=3):
    """Prompt until the member types something, tolerating non-interactive runs."""
    for _ in range(attempts):
        try:
            answer = input(prompt).strip()
        except (EOFError, OSError):
            return ""
        if answer:
            return answer
    return ""


def detect_email():
    """Return ``(email, source)`` for the Google account signed in to Colab.

    ``authenticate_user`` requests only the ``email`` scope, so a display name is never
    available here; the member supplies that separately. The access token is used to
    call Google's own userinfo endpoint and never leaves Google.
    """
    try:
        from google.colab import auth
    except ImportError:
        return None, "manual"

    try:
        auth.authenticate_user()
        import google.auth
        import google.auth.transport.requests

        credentials, _project = google.auth.default(scopes=["email"])
        credentials.refresh(google.auth.transport.requests.Request())
        request = urllib.request.Request(
            USERINFO_URL, headers={"Authorization": f"Bearer {credentials.token}"}
        )
        with urllib.request.urlopen(request, timeout=USERINFO_TIMEOUT_SECONDS) as response:
            info = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(f"Could not read your Google account automatically ({error}).")
        return None, "manual"

    email = (info or {}).get("email")
    return (email, "colab_oauth") if email else (None, "manual")


def resolve_member(name=None, email=None):
    """Establish who is submitting, asking only for what Colab cannot supply."""
    global _cached_member
    if _cached_member is not None and name is None and email is None:
        return _cached_member

    if email:
        detected, source = email, "manual"
    else:
        detected, source = detect_email()
    if not detected:
        detected = _ask("Google email address for submissions: ")
        source = "manual"
    if not name:
        name = _ask("Display name for the leaderboard: ")

    _cached_member = {
        "member_name": name.strip(),
        "member_email": detected.strip().lower(),
        "email_source": source,
    }
    return _cached_member


def reset_member():
    """Forget the cached identity so the next call prompts again."""
    global _cached_member
    _cached_member = None


def describe_model(model):
    """Return ``(class source, layer summary)`` for the member's model."""
    try:
        source = inspect.getsource(type(model))
    except (OSError, TypeError):
        source = ""
    try:
        summary = str(model)
    except Exception:
        summary = ""
    return source, summary


def _package_version():
    try:
        from importlib.metadata import version

        return version("mandelbrot-network-challenge")
    except Exception:
        return "unknown"


def _torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"


def build_payload(
    result,
    model,
    optimizer,
    loss_function,
    batch_size,
    scheduler,
    member,
    run_stamp="",
):
    """Build the submission row. Pure: named fields only, no network."""
    model_source, model_repr = describe_model(model)
    return {
        "workshop_id": workshop_id(),
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "member_name": member.get("member_name", ""),
        "member_email": member.get("member_email", ""),
        "email_source": member.get("email_source", "manual"),
        "validation_mae": result.validation_mae,
        "parameter_count": result.parameter_count,
        "training_seconds": result.training_seconds,
        "steps": result.steps,
        "samples_per_second": result.samples_per_second,
        "gpu_name": result.gpu_name,
        "batch_size": batch_size,
        "model_source": model_source,
        "model_repr": model_repr,
        "optimizer_repr": repr(optimizer),
        "loss_repr": repr(loss_function),
        "scheduler_repr": repr(scheduler),
        "package_version": _package_version(),
        "torch_version": _torch_version(),
        "run_stamp": run_stamp,
    }


def _save_local(payload):
    directory = fallback_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = directory / f"submission_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def post_submission(payload):
    """Save the row locally, then POST it. Returns whether the POST succeeded."""
    path = _save_local(payload)
    url = endpoint()
    if "REPLACE_WITH_DEPLOYMENT_ID" in url:
        print(f"No submission endpoint configured yet; kept this run at {path}")
        return False

    body = json.dumps({"token": shared_token(), "payload": payload}).encode("utf-8")
    for attempt in range(POST_ATTEMPTS):
        try:
            # Apps Script answers /exec with a redirect; urllib follows it as a GET,
            # which still returns the ContentService body.
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(request, timeout=POST_TIMEOUT_SECONDS) as response:
                answer = json.loads(response.read().decode("utf-8") or "{}")
            if answer.get("ok"):
                return True
            last_error = answer.get("error", "the server rejected the submission")
        except Exception as error:
            last_error = error
        if attempt < POST_ATTEMPTS - 1:
            time.sleep(0.5 * 2**attempt + random.uniform(0, 0.5))

    print(f"Could not reach the submission server ({last_error}); kept this run at {path}")
    return False


def submit_result(
    result,
    model,
    optimizer,
    loss_function,
    batch_size,
    scheduler,
    member,
    run_stamp="",
):
    """Capture and send one training run. Never raises: training must not fail here."""
    try:
        payload = build_payload(
            result, model, optimizer, loss_function, batch_size, scheduler, member, run_stamp
        )
        return post_submission(payload)
    except Exception as error:
        print(f"Submission skipped after an unexpected error: {error}")
        return False
