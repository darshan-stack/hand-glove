# Adaptive Personalized Sign-to-Speech Smart Glove

A final-year engineering prototype for real-time sign/gesture-to-speech communication using 5 flex sensors, an IMU, Raspberry Pi 4 Model B, machine learning, personalization, temporal validation, confidence gating, and multilingual text-to-speech.

## Project goal

Build a wearable glove that converts hand gestures into meaningful spoken output while adapting to different users and rejecting uncertain gestures.

## Planned architecture

```text
5 Flex Sensors ──> ADS1115 ADC ──┐
                                 ├─> Raspberry Pi 4 ─> Preprocessing
MPU6050 IMU ─────────────────────┘                     │
                                                       ├─> Sensor ML
Camera (Phase 2) ──────────────────────────────────────┤
                                                       ├─> Temporal validation
                                                       ├─> Confidence gate
                                                       ├─> Personalization
                                                       └─> Multilingual TTS
```

## Development phases

1. Sensor acquisition and CSV logging
2. Automatic calibration and normalization
3. Baseline gesture classifier
4. Temporal/dynamic gesture recognition
5. Confidence-based rejection
6. User-specific gesture learning
7. Camera-assisted multimodal recognition
8. Multilingual speech output

## Initial hardware

- Raspberry Pi 4 Model B
- 5 flex sensors
- ADS1115 16-bit ADC
- MPU6050 IMU
- Glove
- Breadboard/perfboard, resistors and jumper wires
- Speaker/headset for TTS
- Optional camera for Phase 2

## Important research positioning

Existing gesture gloves are used as the baseline. This project focuses on the combination of automatic user calibration, personalized gesture learning, flex+IMU sensor fusion, temporal validation, confidence/unknown rejection, and optional camera-assisted multimodal recognition. Patentability is not assumed; novelty and prior art must be examined formally.

## Repository status

The repository is being built incrementally. The first milestone is reliable live sensor acquisition and dataset generation before training a model.
