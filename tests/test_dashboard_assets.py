import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = (ROOT / "apps_script" / "Code.gs").read_text()
DASHBOARD = (ROOT / "apps_script" / "Dashboard.html").read_text()


class DashboardAssetTests(unittest.TestCase):
    def test_dashboard_reads_only_an_explicitly_public_workshop(self):
        self.assertIn("PUBLIC_WORKSHOP_IDS", CODE)
        self.assertIn("publicWorkshopId(requestedWorkshopId)", CODE)
        self.assertNotIn("sheetFor(requestedWorkshopId)", CODE)

    def test_dashboard_response_omits_private_submission_fields(self):
        dashboard_function = CODE.split("function getDashboardData", 1)[1]
        dashboard_function = dashboard_function.split("function finiteNumber", 1)[0]
        for private_field in (
            "member_email",
            "model_source",
            "model_repr",
            "optimizer_repr",
            "loss_repr",
            "scheduler_repr",
            "token",
        ):
            self.assertNotIn(private_field, dashboard_function)

    def test_dashboard_polls_and_draws_the_autoresearch_marks(self):
        self.assertRegex(DASHBOARD, r"REFRESH_MS\s*=\s*2500")
        self.assertIn("google.script.run", DASHBOARD)
        self.assertIn("getDashboardData(WORKSHOP_ID)", DASHBOARD)
        self.assertIn("best-line", DASHBOARD)
        self.assertIn("run.isRecord", DASHBOARD)

    def test_dashboard_uses_text_content_for_member_supplied_text(self):
        self.assertIn("tooltip.textContent", DASHBOARD)
        self.assertNotIn("tooltip.innerHTML", DASHBOARD)
        self.assertIsNone(re.search(r"innerHTML\s*=", DASHBOARD))


if __name__ == "__main__":
    unittest.main()
