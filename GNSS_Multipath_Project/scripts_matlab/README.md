# MATLAB: NMEA / RTK Validation Utilities

Supporting tools, not analysis. These validate positioning accuracy directly from NMEA output, independently of the Python pipeline.

## Problem

The Python work computes positioning error from parsed GNSS logs through a WLS solver written for this project. That solver is validated against the shipped `WlsPosition` (0.064 m median agreement; see `../notebooks_python/positioning/`), but both sit inside the same codebase and share the same parsing assumptions.

These scripts provide an **independent check** on a different footing: compare an NMEA track against a reference directly, using the GNSS community's standard evaluation conventions, with no shared code path.

## Data

NMEA sentence files:

- **Test track**: chipset or RTK NMEA output (`*_chipset.nmea`)
- **Reference**: either a SPAN NMEA ground-truth track (`SPAN_*_1Hz.nmea`, `SPAN_*_10Hz.nmea`) or a fixed `[latDeg, lonDeg, altM]` vector for a stationary reference

Both are present in the 2020 SDC drives under `data/01_raw/2020-*-US-MTV-*/`.

## Method

| File | What it does |
|---|---|
| `utils/Nmea2RtkMetrics.m` | Computes RTK evaluation metrics for a test NMEA file against a reference. Takes an optional `thresh1MeterSec`, the minimum continuous seconds below 1 m error required to count as having *reached* 1 m accuracy (default 10 s), which is the convention that stops a momentary dip being scored as convergence |
| `utils/Nmea2ErrorPlot.m` | Plots the horizontal error time series for a test track against its reference |

```matlab
rtkMetrics = Nmea2RtkMetrics(testFileName, refFileName, dirName, thresh1MeterSec)
```

`refFileName` may be a `1x3` vector `[latDeg, lonDeg, altM]`, in which case it is treated as a stationary reference rather than a track.

## Results

These produce per-track diagnostics rather than a headline result, and they are not the source of any number quoted in the project's findings. They were used to sanity-check the 2020 drives' reference tracks before the `pixel/` pipeline's time-synchronisation step; specifically, to confirm the SPAN tracks were sane before trusting the Haversine error label built from them.

The positioning results the dissertation reports come from the Python WLS solver in `../notebooks_python/positioning/position_mitigation.ipynb`, which carries its own validation.

## How to run

Needs MATLAB. From this directory:

```matlab
addpath('utils');
dirName = '../data/01_raw/2020-05-14-US-MTV-1';
metrics = Nmea2RtkMetrics('Pixel4_chipset.nmea', 'SPAN_Pixel4_1Hz.nmea', dirName);
Nmea2ErrorPlot('Pixel4_chipset.nmea', 'SPAN_Pixel4_1Hz.nmea', dirName);
```

Use the **1 Hz** SPAN file unless you specifically need 10 Hz. The chipset NMEA is 1 Hz, so the higher rate only adds interpolation.

## Attribution

Adapted from the reference NMEA/RTK evaluation tools published by Google with the **Smartphone Decimeter Challenge**.
