import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from mandelbrot_challenge import submission


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4), nn.GELU(), nn.Linear(4, 1), nn.Sigmoid())

    def forward(self, points):
        return self.layers(points)


class FakeResponse:
    def __init__(self, body):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def fake_result():
    return SimpleNamespace(
        validation_mae=0.0123,
        parameter_count=1234,
        training_seconds=30.1,
        steps=900,
        samples_per_second=245_000.0,
        gpu_name="Tesla T4",
    )


def member():
    return {
        "member_name": "Ada Lovelace",
        "member_email": "ada@example.com",
        "email_source": "colab_oauth",
    }


def build(**environment):
    with patch.dict(os.environ, environment):
        return submission.build_payload(
            fake_result(),
            TinyModel(),
            "AdamW(lr=0.002)",
            "MSELoss()",
            8192,
            None,
            member(),
            run_stamp="20260831-120000",
        )


class PayloadTests(unittest.TestCase):
    def test_payload_carries_named_result_identity_and_code_fields(self):
        payload = build(MANDELBROT_WORKSHOP_ID="fall-2026")
        self.assertEqual(payload["workshop_id"], "fall-2026")
        self.assertEqual(payload["member_email"], "ada@example.com")
        self.assertEqual(payload["email_source"], "colab_oauth")
        self.assertEqual(payload["validation_mae"], 0.0123)
        self.assertEqual(payload["parameter_count"], 1234)
        self.assertEqual(payload["steps"], 900)
        self.assertEqual(payload["gpu_name"], "Tesla T4")
        self.assertEqual(payload["batch_size"], 8192)
        self.assertEqual(payload["scheduler_repr"], "None")
        self.assertEqual(payload["run_stamp"], "20260831-120000")
        self.assertIn("class TinyModel", payload["model_source"])
        self.assertIn("Linear", payload["model_repr"])

    def test_payload_is_json_serializable(self):
        json.dumps(build())

    def test_payload_excludes_private_challenge_data(self):
        # AGENTS.md puts the underscore-prefixed tensors off-limits; nothing derived
        # from them may reach the spreadsheet.
        payload = build()
        self.assertFalse([key for key in payload if key.startswith("_")])
        for forbidden in ("_train_points", "_train_targets", "_validation_points", "_validation_targets"):
            self.assertNotIn(forbidden, json.dumps(payload))

    def test_workshop_id_falls_back_to_the_package_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(submission.workshop_id(), submission.DEFAULT_WORKSHOP_ID)

    def test_describe_model_survives_a_model_without_source(self):
        source, summary = submission.describe_model(object())
        self.assertEqual(source, "")
        self.assertIsInstance(summary, str)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        submission.reset_member()
        self.addCleanup(submission.reset_member)

    def test_detect_email_falls_back_when_colab_is_unavailable(self):
        email, source = submission.detect_email()
        self.assertIsNone(email)
        self.assertEqual(source, "manual")

    def test_explicit_details_skip_every_prompt(self):
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            resolved = submission.resolve_member(name="Ada", email="Ada@Example.com")
        self.assertEqual(resolved["member_email"], "ada@example.com")
        self.assertEqual(resolved["member_name"], "Ada")

    def test_identity_is_cached_so_rerunning_setup_does_not_reprompt(self):
        submission.resolve_member(name="Ada", email="ada@example.com")
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            self.assertEqual(submission.resolve_member()["member_name"], "Ada")

    def test_name_is_requested_when_only_the_email_is_known(self):
        with patch("builtins.input", return_value="Grace") as prompt:
            resolved = submission.resolve_member(email="grace@example.com")
        self.assertEqual(resolved["member_name"], "Grace")
        self.assertEqual(prompt.call_count, 1)

    def test_non_interactive_prompts_do_not_raise(self):
        with patch("builtins.input", side_effect=EOFError):
            resolved = submission.resolve_member(email="grace@example.com")
        self.assertEqual(resolved["member_name"], "")


class TransportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        patcher = patch.dict(
            os.environ,
            {
                "MANDELBROT_SUBMIT_URL": "https://example.test/exec",
                "MANDELBROT_SUBMIT_DIR": str(self.directory),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def saved_payloads(self):
        return sorted(self.directory.glob("submission_*.json"))

    def test_successful_post_reports_success(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse('{"ok": true}')) as opened:
            self.assertTrue(submission.post_submission({"member_name": "Ada"}))
        self.assertEqual(opened.call_count, 1)

    def test_post_retries_a_transient_failure_then_succeeds(self):
        responses = [OSError("connection reset"), FakeResponse('{"ok": true}')]

        def flaky(*_args, **_kwargs):
            outcome = responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("urllib.request.urlopen", side_effect=flaky) as opened:
            with patch("time.sleep"):
                self.assertTrue(submission.post_submission({"member_name": "Ada"}))
        self.assertEqual(opened.call_count, 2)

    def test_total_failure_still_keeps_the_run_on_disk(self):
        with patch("urllib.request.urlopen", side_effect=OSError("no route to host")):
            with patch("time.sleep"):
                self.assertFalse(submission.post_submission({"member_name": "Ada"}))
        saved = self.saved_payloads()
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text())["member_name"], "Ada")

    def test_server_rejection_is_not_reported_as_success(self):
        rejection = FakeResponse('{"ok": false, "error": "bad token"}')
        with patch("urllib.request.urlopen", return_value=rejection):
            with patch("time.sleep"):
                self.assertFalse(submission.post_submission({"member_name": "Ada"}))

    def test_unconfigured_endpoint_saves_locally_without_posting(self):
        with patch.dict(os.environ, {"MANDELBROT_SUBMIT_URL": submission.DEFAULT_ENDPOINT}):
            with patch("urllib.request.urlopen", side_effect=AssertionError("should not post")):
                self.assertFalse(submission.post_submission({"member_name": "Ada"}))
        self.assertEqual(len(self.saved_payloads()), 1)

    def test_post_sends_the_shared_token_and_payload(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse('{"ok": true}')) as opened:
            with patch.dict(os.environ, {"MANDELBROT_SUBMIT_TOKEN": "secret"}):
                submission.post_submission({"member_name": "Ada"})
        sent = json.loads(opened.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent["token"], "secret")
        self.assertEqual(sent["payload"]["member_name"], "Ada")

    def test_submission_failure_can_never_break_training(self):
        with patch.object(submission, "build_payload", side_effect=RuntimeError("boom")):
            outcome = submission.submit_result(
                fake_result(), TinyModel(), "AdamW", "MSELoss", 8192, None, member()
            )
        self.assertFalse(outcome)

    def test_submit_result_posts_a_complete_payload(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse('{"ok": true}')) as opened:
            outcome = submission.submit_result(
                fake_result(), TinyModel(), "AdamW", "MSELoss", 8192, None, member()
            )
        self.assertTrue(outcome)
        sent = json.loads(opened.call_args.args[0].data.decode("utf-8"))["payload"]
        self.assertEqual(sent["member_email"], "ada@example.com")
        self.assertEqual(sent["validation_mae"], 0.0123)


if __name__ == "__main__":
    unittest.main()
