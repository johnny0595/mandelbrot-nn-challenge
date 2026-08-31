# Mandelbrot Network Challenge

A concise Google Colab competition: build the neural network that best approximates
the Mandelbrot set during a fixed 30-second NVIDIA T4 optimization budget.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnny0595/mandelbrot-network-challenge-students/blob/main/mandelbrot_challenge.ipynb)

## What students can change

- Model architecture
- Optimizer and learning rate
- Loss function
- Batch size
- Learning-rate scheduler

Datasets, seeds, target generation, the timer, public scoring, views, rendering,
TensorBoard capture, and artifact generation are fixed behind the imported
`mandelbrot_challenge` API. Final rankings should use the separate private evaluator;
the public validation score is for iteration only.

## Competition rules

- Edit only the two `STUDENT WORKSPACE` cells.
- AI assistance is allowed, but the same rules apply to the AI.
- Tutorials, public code, and ideas from projects such as `fractalsearch` are allowed,
  provided the submitted implementation fits in the two student cells and follows every
  other rule.
- Do not modify or bypass the timer, data, seed, target, scoring, or challenge package.
- Do not access internal training/validation tensors or train outside `challenge.train`.
- Do not calculate the Mandelbrot formula inside the model or use pretrained weights.
- Use an NVIDIA T4. The challenge refuses to train on a different GPU.

Final order is: hidden MAE rounded to five decimals, then fewer parameters, then
faster inference. There is no parameter cap.

## Preserved project features

- Smooth and periodic Mandelbrot targets
- Full-view and deep-zoom comparisons
- Interactive target/prediction/error explorer with cursor zoom, pan, preset jumps,
  and fresh deep-zoom re-rendering in Colab
- True-fractal and trained-model zoom videos
- Predefined views captured during training and encoded to MP4
- TensorBoard loss curve and prediction snapshots throughout training
- Model saving/loading

## About benchmark numbers

Scores from this classroom notebook are not numerically comparable with the
fractalsearch experiment log. This challenge reports **MAE**, trains for 30 seconds,
and uses a fixed smooth-escape target dataset. Fractalsearch ranked **MSE** after a
300-second run on a periodic log-distance target with fresh adaptive sampling. Use the
separate private evaluator—and the same Colab GPU type—to compare student entries.

## Submissions

Use a Google Form connected to a Google Sheet. Students submit their email, reported
public score, parameter count, both editable code cells, a view-only Colab link, and a
cleared-output `.ipynb` file. Do not submit model weights. See [SUBMISSIONS.md](SUBMISSIONS.md)
for the exact form fields and instructor workflow.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```
