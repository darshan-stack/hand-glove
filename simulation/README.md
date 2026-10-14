# HandGlove Studio — software-only simulation

Browser-based companion for the Adaptive Personalized Sign-to-Speech Smart Glove. This is a software simulation, not calibrated hardware emulation.

## Run
From repository root run `python3 -m http.server 8000` and open `http://localhost:8000/simulation/`.

Virtual glove mode works without camera access. Webcam mode requires internet for MediaPipe Tasks Vision assets and browser camera permission. See https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js.

## Features
- Virtual controls for five synthetic finger-bend channels and wrist orientation.
- Optional webcam hand-landmark detection.
- Custom gesture enrollment, phrase mapping, and browser-local persistence.
- Nearest-example similarity ranking and configurable rejection threshold.
- Browser speech synthesis in English, Hindi, and Marathi where voices are available.
- JSON export/import and explicit labeling that virtual values are synthetic.

## Limits
Similarity is a heuristic, not a calibrated probability or measured accuracy. Train multiple varied examples per gesture and test on separate examples before reporting metrics. This is not a universal Indian Sign Language translator: continuous signs, two-hand signs, facial expressions, grammar, and context require further data and modeling. Webcam landmarks are not flex sensor/IMU readings. User data stays in local browser storage unless exported. Existing Python ML code is not yet connected to this standalone UI; hardware acquisition remains a separate milestone.
