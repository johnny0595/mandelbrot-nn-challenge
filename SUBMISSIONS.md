# Submission workflow

Use one Google Form connected to a Google Sheet. This is simpler than collecting model
weights or setting up GitHub Classroom for a one-hour beginner competition.

## Google Form settings

- Collect email addresses.
- Limit to one response, but allow students to edit until the deadline.
- Restrict file uploads to `.ipynb` and allow at least 100 MB.
- Link responses to a Google Sheet.
- Close responses at the deadline.

## Required fields

1. Student name
2. Collected email address
3. Public validation MAE
4. Parameter count
5. Training steps
6. GPU name (`Tesla T4` or `NVIDIA T4`)
7. Model-architecture cell code (long answer)
8. Training-choices cell code (long answer)
9. View-only Colab notebook link
10. Cleared-output `.ipynb` upload
11. Required checkbox: “I changed only the two student cells, did not calculate the
    Mandelbrot formula in my model, did not train outside the official timer, and this
    submission contains the code I ran.”

## Student submission steps

1. Run the notebook from the model cell so the reported result matches the submitted code.
2. Copy the printed MAE, parameter count, steps, and GPU into the form.
3. Paste both student cells into their matching form fields.
4. In Colab, share the notebook as view-only to anyone with the link.
5. Save a backup copy, clear its outputs to remove large embedded media, download it as
   `.ipynb`, and upload that file to the form.
6. Submit before the deadline. Model weights are not required.

## Instructor scoring

1. Treat student-entered public scores as provisional.
2. Review the two pasted cells for rule violations and sharing-permission problems.
3. Rerun qualifying submissions from a fresh runtime on the same NVIDIA T4.
4. Run the private evaluator on the resulting in-memory `model`.
5. Sort by hidden MAE rounded to five decimals, then parameter count, then inference time.
6. Keep the private evaluator and instructor solution inaccessible until the competition
   and any appeals are complete.

Add five instructor-only columns to the linked response sheet: `Eligible`, `Hidden MAE`,
`Verified parameters`, `Inference ms`, and `Review note`. This keeps the original form
response untouched while giving you one auditable leaderboard. Publish only student
name, rounded hidden MAE, parameters, and rank unless students consent to sharing more.
