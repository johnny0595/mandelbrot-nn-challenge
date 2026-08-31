# Mandelbrot Network Challenge

A concise Google Colab competition: build the neural network that best approximates
the Mandelbrot set during a fixed 30-second NVIDIA T4 optimization budget.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnny0595/mandelbrot-nn-challenge/blob/main/mandelbrot_challenge.ipynb)

## What members can change

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

- Edit only the two `MEMBER WORKSPACE` cells.
- AI assistance is allowed, but the same rules apply to the AI.
- General tutorials, documentation, and AI help are allowed. Do not look up or copy an
  existing solution to this specific challenge.
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

## Submissions

Use a Google Form connected to a Google Sheet. Members submit their email, reported
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
