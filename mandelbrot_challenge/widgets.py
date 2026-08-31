"""Interactive notebook explorer kept out of the student-facing cells."""

import base64
from io import BytesIO
import json
import math
from uuid import uuid4

import matplotlib.pyplot as plt

from .rendering import PRESET_VIEWS, View, render_comparison


def _png_data_url(image):
    buffer = BytesIO()
    plt.imsave(buffer, image, cmap="inferno", origin="lower", vmin=0, vmax=1, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _viewer_html(truth, prediction, error, callback_name="", base_depth=100):
    viewer_id = f"mandelbrot-viewer-{uuid4().hex}"
    images = {
        "target": _png_data_url(truth),
        "prediction": _png_data_url(prediction),
        "error": _png_data_url(error),
    }
    views = {
        name: {
            "xmin": view.xmin,
            "xmax": view.xmax,
            "ymin": view.ymin,
            "ymax": view.ymax,
        }
        for name, view in PRESET_VIEWS.items()
    }
    full = views["Full set"]
    aspect_ratio = truth.shape[1] / truth.shape[0]
    template = r"""
<div id="__VIEWER_ID__" class="mb-viewer">
  <style>
    #__VIEWER_ID__ { font: 14px/1.4 system-ui, sans-serif; max-width: 1000px; color: #e8e8ed; }
    #__VIEWER_ID__ .mb-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 10px; }
    #__VIEWER_ID__ button, #__VIEWER_ID__ select {
      border: 1px solid #555; border-radius: 7px; background: #25252b; color: inherit;
      padding: 7px 11px; cursor: pointer;
    }
    #__VIEWER_ID__ button.active { background: #f06a39; border-color: #f06a39; color: #111; font-weight: 700; }
    #__VIEWER_ID__ button:disabled { cursor: wait; opacity: .55; }
    #__VIEWER_ID__ .mb-stage {
      position: relative; width: 100%; aspect-ratio: __ASPECT__; overflow: hidden;
      border: 1px solid #555; border-radius: 10px; background: #08080a;
      cursor: grab; touch-action: none; user-select: none;
    }
    #__VIEWER_ID__ .mb-stage.dragging { cursor: grabbing; }
    #__VIEWER_ID__ img {
      position: absolute; inset: 0; width: 100%; height: 100%; object-fit: fill;
      transform-origin: 0 0; image-rendering: pixelated; pointer-events: none;
    }
    #__VIEWER_ID__ .mb-hud {
      position: absolute; left: 10px; bottom: 10px; padding: 6px 9px; border-radius: 6px;
      background: rgba(0, 0, 0, .72); font: 12px ui-monospace, monospace; pointer-events: none;
    }
    #__VIEWER_ID__ .mb-help { color: #aaa; margin: 8px 2px 0; }
  </style>
  <div class="mb-toolbar">
    <button data-layer="target" class="active">Target</button>
    <button data-layer="prediction">Prediction</button>
    <button data-layer="error">Absolute error</button>
    <select aria-label="Jump to view"></select>
    <button data-action="refine">Re-render this view</button>
    <button data-action="reset">Reset view</button>
  </div>
  <div class="mb-stage">
    <img alt="Interactive Mandelbrot comparison">
    <div class="mb-hud"></div>
  </div>
  <div class="mb-help">Scroll and drag to choose an area, then re-render for fresh pixels; repeat to go deeper · double-click to reset</div>
</div>
<script>
(() => {
  const root = document.getElementById('__VIEWER_ID__');
  const stage = root.querySelector('.mb-stage');
  const image = root.querySelector('img');
  const hud = root.querySelector('.mb-hud');
  const select = root.querySelector('select');
  const images = __IMAGES__;
  const views = __VIEWS__;
  const full = __FULL__;
  const callbackName = __CALLBACK_NAME__;
  const refineButton = root.querySelector('[data-action="refine"]');
  let baseBounds = {...full};
  let renderDepth = __BASE_DEPTH__;
  let currentLayer = 'target';
  let scale = 1, tx = 0, ty = 0, dragging = false, rendering = false, lastX = 0, lastY = 0;

  Object.keys(views).forEach(name => {
    const option = document.createElement('option');
    option.value = option.textContent = name;
    select.appendChild(option);
  });

  function visibleBounds() {
    const rect = stage.getBoundingClientRect();
    const left = (0 - tx) / scale;
    const right = (rect.width - tx) / scale;
    const top = (0 - ty) / scale;
    const bottom = (rect.height - ty) / scale;
    return {
      xmin: baseBounds.xmin + left / rect.width * (baseBounds.xmax - baseBounds.xmin),
      xmax: baseBounds.xmin + right / rect.width * (baseBounds.xmax - baseBounds.xmin),
      ymax: baseBounds.ymax - top / rect.height * (baseBounds.ymax - baseBounds.ymin),
      ymin: baseBounds.ymax - bottom / rect.height * (baseBounds.ymax - baseBounds.ymin),
    };
  }

  function apply() {
    image.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    const visible = visibleBounds();
    const x = (visible.xmin + visible.xmax) / 2;
    const y = (visible.ymin + visible.ymax) / 2;
    const totalZoom = (full.xmax - full.xmin) / (visible.xmax - visible.xmin);
    hud.textContent = `${totalZoom.toFixed(totalZoom < 10 ? 2 : 1)}×  center ${x.toFixed(7)} ${y < 0 ? '−' : '+'} ${Math.abs(y).toFixed(7)}i  ·  ${renderDepth} iterations`;
  }

  function jumpLocally(name) {
    const view = views[name];
    const rect = stage.getBoundingClientRect();
    const left = (view.xmin - baseBounds.xmin) / (baseBounds.xmax - baseBounds.xmin) * rect.width;
    const right = (view.xmax - baseBounds.xmin) / (baseBounds.xmax - baseBounds.xmin) * rect.width;
    const top = (baseBounds.ymax - view.ymax) / (baseBounds.ymax - baseBounds.ymin) * rect.height;
    const bottom = (baseBounds.ymax - view.ymin) / (baseBounds.ymax - baseBounds.ymin) * rect.height;
    scale = Math.min(rect.width / (right - left), rect.height / (bottom - top));
    tx = rect.width / 2 - ((left + right) / 2) * scale;
    ty = rect.height / 2 - ((top + bottom) / 2) * scale;
    select.value = name;
    apply();
  }

  function fitStageAspect(bounds) {
    const rect = stage.getBoundingClientRect();
    const wanted = rect.width / rect.height;
    let width = bounds.xmax - bounds.xmin;
    let height = bounds.ymax - bounds.ymin;
    const cx = (bounds.xmin + bounds.xmax) / 2;
    const cy = (bounds.ymin + bounds.ymax) / 2;
    if (width / height > wanted) height = width / wanted;
    else width = height * wanted;
    return {xmin: cx - width / 2, xmax: cx + width / 2, ymin: cy - height / 2, ymax: cy + height / 2};
  }

  async function renderBounds(bounds, presetName='') {
    if (!callbackName || !window.google?.colab?.kernel || rendering) return;
    bounds = fitStageAspect(bounds);
    rendering = true;
    refineButton.disabled = true;
    hud.textContent = 'Rendering fresh target, prediction, and error pixels…';
    try {
      const response = await google.colab.kernel.invokeFunction(
        callbackName,
        [bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax],
        {}
      );
      let payload = response.data['application/json'];
      if (typeof payload === 'string') payload = JSON.parse(payload);
      Object.assign(images, payload.images);
      baseBounds = {...bounds};
      renderDepth = payload.max_depth;
      scale = 1; tx = 0; ty = 0;
      image.src = images[currentLayer];
      if (presetName) select.value = presetName;
      apply();
    } catch (error) {
      hud.textContent = `Render failed: ${error.message}`;
      console.error(error);
    } finally {
      rendering = false;
      refineButton.disabled = false;
    }
  }

  root.querySelectorAll('[data-layer]').forEach(button => {
    button.addEventListener('click', () => {
      currentLayer = button.dataset.layer;
      image.src = images[currentLayer];
      root.querySelectorAll('[data-layer]').forEach(item => item.classList.toggle('active', item === button));
    });
  });
  select.addEventListener('change', () => {
    if (callbackName) renderBounds(views[select.value], select.value);
    else jumpLocally(select.value);
  });
  refineButton.addEventListener('click', () => renderBounds(visibleBounds()));
  root.querySelector('[data-action="reset"]').addEventListener('click', () => {
    if (callbackName) renderBounds(full, 'Full set');
    else jumpLocally('Full set');
  });
  stage.addEventListener('dblclick', () => {
    if (callbackName) renderBounds(full, 'Full set');
    else jumpLocally('Full set');
  });
  stage.addEventListener('wheel', event => {
    event.preventDefault();
    const rect = stage.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const next = Math.min(80, Math.max(1, scale * Math.exp(-event.deltaY * .0015)));
    const ratio = next / scale;
    tx = x - (x - tx) * ratio;
    ty = y - (y - ty) * ratio;
    scale = next;
    apply();
  }, { passive: false });
  stage.addEventListener('pointerdown', event => {
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    stage.classList.add('dragging'); stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener('pointermove', event => {
    if (!dragging) return;
    tx += event.clientX - lastX; ty += event.clientY - lastY;
    lastX = event.clientX; lastY = event.clientY; apply();
  });
  stage.addEventListener('pointerup', () => { dragging = false; stage.classList.remove('dragging'); });
  if (!callbackName || !window.google?.colab?.kernel) {
    refineButton.disabled = true;
    refineButton.title = 'Fresh re-rendering is available when this notebook runs in Google Colab.';
  }
  image.src = images.target;
  requestAnimationFrame(apply);
})();
</script>
"""
    return (
        template.replace("__VIEWER_ID__", viewer_id)
        .replace("__ASPECT__", str(aspect_ratio))
        .replace("__IMAGES__", json.dumps(images))
        .replace("__VIEWS__", json.dumps(views))
        .replace("__FULL__", json.dumps(full))
        .replace("__CALLBACK_NAME__", json.dumps(callback_name))
        .replace("__BASE_DEPTH__", str(base_depth))
    )


def _zoom_payload(model, bounds, target="smooth", base_depth=100, width=1024):
    bounds = tuple(float(value) for value in bounds)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("View coordinates must be finite numbers.")
    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin or min(xmax - xmin, ymax - ymin) < 1e-14:
        raise ValueError("This view is beyond float64 zoom precision.")

    full = PRESET_VIEWS["Full set"]
    zoom = max(1.0, (full.xmax - full.xmin) / (xmax - xmin))
    depth = min(2_000, base_depth + round(20 * math.log2(zoom)))
    view = View("Interactive zoom", xmin, xmax, ymin, ymax)
    truth, prediction, error = render_comparison(
        model,
        view,
        width=width,
        target=target,
        max_depth=depth,
        target_precision=64,
    )
    return {
        "images": {
            "target": _png_data_url(truth),
            "prediction": _png_data_url(prediction),
            "error": _png_data_url(error),
        },
        "max_depth": depth,
    }


def view_picker(model, target="smooth", max_depth=100):
    """Display a shared pan/zoom canvas for target, prediction, and error."""
    from IPython.display import HTML, display

    callback_name = ""
    try:
        from google.colab import output

        callback_name = f"mandelbrot_challenge.render_{uuid4().hex}"

        def render_zoom(xmin, xmax, ymin, ymax):
            return _zoom_payload(model, (xmin, xmax, ymin, ymax), target, max_depth)

        output.register_callback(callback_name, render_zoom)
    except ImportError:
        pass

    truth, prediction, error = render_comparison(
        model, PRESET_VIEWS["Full set"], width=1024, target=target, max_depth=max_depth
    )
    viewer = HTML(_viewer_html(truth, prediction, error, callback_name, max_depth))
    display(viewer)
