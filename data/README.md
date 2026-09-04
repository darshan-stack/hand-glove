# Dataset

Raw sensor recordings should be stored outside Git when large. Keep a small sample in the repository if useful.

## Required fields

`timestamp,user_id,gesture,flex1,flex2,flex3,flex4,flex5,ax,ay,az,gx,gy,gz`

## Collection rules

- Record multiple repetitions per gesture.
- Record multiple users.
- Keep user IDs so cross-user evaluation can be performed.
- Record static and dynamic gestures separately.
- Keep calibration recordings separate from test recordings.
- Never mix samples from the same continuous recording across train and test in a way that causes leakage.
