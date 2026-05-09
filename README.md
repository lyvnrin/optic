# optic

A growing collection of computer vision scripts built with OpenCV and MediaPipe. Each one is a self-contained experiment - gesture detection, live filters, tracking, whatever seemed interesting that week.

## scripts

| script | what it does |
|---|---|
| `gesture_filters.py` | Use hand signals to switch live camera filters (B&W, warm, sepia, etc.) |

## setup

Each script is standalone. Install the common dependencies and run directly.

```bash
pip install opencv-python mediapipe numpy
python <script_name>.py
```

Some scripts may have additional dependencies — check the docstring at the top of the file.

## structure

Scripts live at the root. If a script grows into something larger (multiple files, assets, a config), it gets its own folder.

## built with

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- Python 3.10+
