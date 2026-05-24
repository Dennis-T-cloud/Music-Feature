# Dataset

This project uses MAESTRO v3.0.0 MIDI files.

Official dataset page:

```text
https://magenta.tensorflow.org/datasets/maestro
```

Direct MIDI zip used for this project:

```text
https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
```

Recommended local layout after download and extraction:

```text
data/
  maestro-v3.0.0/
    maestro-v3.0.0.csv
    maestro-v3.0.0.json
    2004/
    2006/
    ...
```

The training script assumes this path by default:

```text
data/maestro-v3.0.0
```

## Splits Used

The MAESTRO metadata file provides the official `train`, `validation`, and `test` splits.

For the two-stage experiment:

- Stage 1 uses the MAESTRO train split for general adaptation and the validation split for evaluation.
- Stage 2 continues from Stage 1 and uses the Chopin Etude subset for style adaptation.
- Final reported evaluation uses the MAESTRO test split, Chopin Etude subset.

## Included Metadata

`chopin_etude_metadata.csv` lists the MAESTRO entries whose composer/title match the Chopin Etude subset used in Stage 2.
It is metadata only; MIDI files are not included in this repository.
