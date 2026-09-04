# ML Architecture

## Objective
Recognize static and dynamic hand gestures from 5 flex sensors + 6-axis IMU, adapt to individual users, reject uncertain predictions, and map accepted gestures to multilingual speech intents.

## End-to-end pipeline

Raw CSV -> quality checks -> per-user calibration -> filtering -> normalization -> temporal windowing -> feature engineering -> baseline classifier -> temporal classifier -> confidence estimation -> temporal consistency -> unknown/reject gate -> intent mapping -> multilingual TTS.

## Inputs

11 raw channels: flex1..flex5, ax, ay, az, gx, gy, gz. Derived channels include acceleration magnitude and gyro magnitude. Timestamp is retained for temporal processing but is not a model feature unless explicitly engineered.

## Dataset protocol

Each sample contains user_id, gesture, repetition and timestamp. Collect multiple repetitions from multiple users. Keep users separated for the primary generalization test: training users, validation users, and unseen test users. Do not randomly split adjacent rows from one continuous recording across train/test.

Recommended first dataset: 8-12 gestures, 5+ users, 20+ repetitions per gesture/user where feasible. Start smaller for rapid prototyping, then expand.

## Stage A — baseline

Random Forest on window-level statistical features. Report accuracy, macro precision, macro recall, macro F1, confusion matrix and inference time.

## Stage B — temporal recognition

Use fixed-length overlapping sequences (initially 32 samples; tune using validation data). Candidate models: 1D CNN, GRU or LSTM. Use the simplest model that improves dynamic-gesture recognition without unacceptable latency.

## Stage C — personalization

At enrollment, collect neutral/open/closed calibration poses. Fit per-user center/scale normalization. For new gestures, store a validated embedding/prototype only after repeated consistent demonstrations and explicit label confirmation. Never silently learn a single noisy sample.

## Stage D — confidence and rejection

Do not speak every argmax prediction. Combine classifier confidence, temporal consistency and sensor-quality checks. If confidence is below a validation-set threshold, or signals are inconsistent, output REJECT/REPEAT and trigger haptic feedback. Thresholds must be selected on validation data, not test data.

## Stage E — multimodal extension

Camera is a second branch, not a replacement for glove sensing. Extract visual features/hand landmarks and fuse them with the sensor representation. Compare sensor-only versus vision-only versus fused performance using the same user-separated test protocol.

## Stage F — multilingual intent layer

The classifier predicts a stable gesture/intent ID first. A separate language layer maps the intent to configured Hindi, Marathi, English or other supported phrases. This prevents language wording from changing the gesture classifier.

## Real-time inference

For live inference, maintain a rolling sensor buffer. Every stride samples, preprocess the newest window, run the classifier, apply temporal agreement and confidence gates, and emit speech only when an accepted prediction persists. Use cooldown/debouncing so one held gesture does not repeatedly speak.

## Research experiments

1. Cross-user baseline vs personalized calibration.
2. Static gestures vs dynamic gestures.
3. Sensor-only vs sensor+camera fusion.
4. With vs without confidence/rejection gate.
5. Latency and CPU/RAM footprint on Raspberry Pi 4.
6. Personalization sample efficiency: measure performance after 0, 1, 3, 5 and 10 demonstrations per new gesture/user.

## Evaluation metrics

Accuracy, macro precision, macro recall, macro F1, confusion matrix, rejection rate, false-accept rate, false-reject rate, per-user performance, calibration time, inference latency and resource usage.

## Research integrity

All reported accuracy, latency and battery values must come from actual experiments. Illustrative targets in presentations must be explicitly labelled as targets. This repository does not claim patentability; novelty requires a formal prior-art search and patent examination.
