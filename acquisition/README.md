# Sensor acquisition

This module will contain Raspberry Pi 4 sensor acquisition and dataset logging.

## Target inputs

- Flex1..Flex5 through ADS1115 ADC
- MPU6050 accelerometer and gyroscope
- Timestamp
- Optional user ID / gesture label metadata

## First milestone

Run a stable loop that reads all sensors, prints a timestamped row, and saves rows to CSV. Do not train the model until sensor readings are reliable.

Expected CSV columns:

```text
timestamp,user_id,gesture,flex1,flex2,flex3,flex4,flex5,ax,ay,az,gx,gy,gz
```

Hardware pinout and Raspberry Pi implementation will be added after the exact sensor modules are confirmed.