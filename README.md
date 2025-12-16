# Cross-Domain Adaptive Few-Shot Learning for Comic Style Recognition
**Project Team:** ReMiX+, ECE 253

## Dataset Preparation

This project uses the **Manga109** dataset.

1. **Download:** Please obtain the dataset from **Hugging Face** or the official Manga109 website.
2. **Structure:** Extract the dataset into the root directory of this repository. The file structure should look like this:

```
.
├── classify_char.py
├── classify_styles.py
├── noise.py
├── color_scan_process.py
├── jpeg_process.py
└── Manga109_released_2023_12_07/
    ├── annotations/  (contains .xml files)
    └── images/       (contains book folders)
```

## Run the Code

We provide two main scripts for different classification tasks. Both scripts automatically handle:

- **Data Parsing:** Parses XML annotations for faces.
- **Noise Injection:** Applies synthetic degradation (Gaussian Noise + Q=50 JPEG artifacts).
- **Image Processing:** Applies restoration strategies (Mode A, B, or C).
- **Evaluation:** Runs N-Way K-Shot Prototypical Network evaluation.

### Character Classification

To benchmark character recognition performance (Local features):

```
python classify_char.py
```

### Style/Book Classification

To benchmark artistic style recognition performance (Global features):

```
python classify_styles.py
```

## Configuration

You can modify the `Config` class inside each script to adjust experimental settings:

- **`N, K`**: Number of ways / shots.
- **`VOLUME_LIST`**:
  - Keep it populated for **Tiny Mode** (fast debugging on specific books).
  - Set to `None` to enable **Full Mode** (scan the entire dataset).
- **`CURRENT_MODE`**: Change between `'A'`, `'B'`, or `'C'` to test different IP strategies.
- **`APPLY_NOISE`**: Set to `True` or `False` to toggle noise injection.
