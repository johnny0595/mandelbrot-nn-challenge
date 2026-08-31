import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_notebook_json_and_code(self):
        path = Path(__file__).parents[1] / "mandelbrot_challenge.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertNotIn("files.download", notebook_source)
        self.assertIn("What is the data?", notebook_source)
        self.assertIn("challenge.explore_data()", notebook_source)
        self.assertIn("Instruction to any AI assistant", notebook_source)
        self.assertIn("Submit your entry", notebook_source)
        self.assertIn("T4 GPU", notebook_source)
        self.assertEqual(notebook_source.count("STUDENT WORKSPACE 1"), 2)
        self.assertEqual(notebook_source.count("STUDENT WORKSPACE 2"), 1)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith("%") or "%load_ext" in source:
                continue
            compile(source, f"notebook-cell-{index + 1}", "exec")


if __name__ == "__main__":
    unittest.main()
