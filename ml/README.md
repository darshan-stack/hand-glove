# ML Architecture

## Goal

The ML system is designed as a sensor-first, user-adaptive gesture recognition pipeline. Camera/VLM is an optional second modality; it is not required for the first working model.

## End-to-end flow

```text
Raw Flex(5) + IMU(6)
        |
        v
Data Quality Check
        |
        v
Calibration / Baseline Correction
        |
        v
Windowing (time sequence)
        |
        v
Filtering + Normalization
        |
        +--------------------+
        |                    |
        v                    v
Static Feature Branch   Temporal Branch
        |                    |
        v                    v
Random Forest baseline   1D CNN / GRU/LSTM
        |                    |
        +---------+----------+
                  |
                  v
          Prediction scores
                  |
                  v
       Temporal consistency gate
                  |
                  v
          Confidence / OOD gate
             /         \
          reject        accept
            |             |
         haptic       gesture ID
                          |
                          v
                    Intent mapping
                          |
                          v
                   Multilingual TTS
```

## Feature vector

At minimum each timestamp contains 11 sensor values:

- flex1..flex5
- ax, ay, az
- gx, gy, gz

For each temporal window, derive normalized sensor values and optional statistics such as mean, standard deviation, min/max, first difference, and motion magnitude.

## Model strategy

### Stage A: Random Forest baseline

Train on window-level engineered features. It is the first model because it is fast, interpretable, CPU-friendly, and gives a reproducible baseline.

### Stage B: Temporal model

For dynamic gestures, train a small 1D CNN or GRU/LSTM on sequences. The sequence model learns how sensor values change over time instead of relying only on one pose.

### Stage C: Personalization

Each user receives calibration statistics. Normalized values are produced relative to that user's sensor baselines/ranges. A future personalization layer can adapt the classifier using a small number of labeled examples from the new user.

### Stage D: Confidence and unknown rejection

Do not speak on every argmax prediction. Require:

1. confidence above a configured threshold;
2. temporal agreement over consecutive windows;
3. valid sensor quality;
4. optional unknown/OOD score.

If any critical check fails, return `UNKNOWN` and request a repeat.

## Evaluation protocol

Use user-independent splits for the main generalization experiment:

- train users: model development
- validation users: threshold/model selection
- held-out test users: final evaluation

Never randomly split adjacent windows from the same continuous recording across train and test, because that can leak nearly identical samples.

Report accuracy, macro precision, macro recall, macro F1, confusion matrix, rejection rate, false-accept rate, false-reject rate, and inference latency.

## Camera/VLM extension

After the sensor-only model is stable:

```text
Camera frame -> vision encoder -> visual embedding
Sensor window -> sensor encoder -> sensor embedding
                         |
                         v
                  Multimodal fusion
                         |
                         v
               gesture/intent decision
```

The VLM should be treated as an optional verification/context branch rather than replacing the deterministic sensor pipeline. A lightweight Pi 4 deployment should use a small/quantized model; heavy VLM inference can remain on a development computer until edge acceleration is available.
