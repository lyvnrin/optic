# optic

A growing collection of computer vision scripts built with OpenCV and MediaPipe. Each one is a self-contained experiment - gesture detection, live filters, tracking, whatever seemed interesting that week.

## scripts

| script | what it does | link |
|---|---|---|
| `nerd-point.py` | Real-time hand gesture detection that overlays a 🤓 emoji whenever an index finger point is detected | [Here](https://youtu.be/y8IPCXzQ8bY)
| `point-size.py` | Point both index fingers at the camera and get the pixel distance between them measured in real time. | [Here](https://youtu.be/VjPwPB4dxVw)
| `point-colour.py` | The distance between two pointing index fingers maps to a colour across the full HSV spectrum, tinting the live camera feed in real time | [Here](https://youtu.be/j46T5VPBS6Q)
| `gesture_filters.py` | Hold up 0–5 fingers to cycle through live camera filters: Original, B&W, Warm, Cool, Sepia, and Vignette | [Here](https://youtu.be/qMUBqgRjAWI)

## setup

Each script is standalone, so install the common dependencies and run directly.

```bash
pip install opencv-python mediapipe numpy
python <script_name>.py
```

Some scripts may have additional dependencies, so check the docstring at the top of the file.

## structure

Scripts live at the root. If a script grows into something larger (multiple files, assets, a config), it gets its own folder.

## built with

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- Python 3.10+
