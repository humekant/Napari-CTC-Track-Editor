# 🧬 CTC Track Editor for Napari

**CTC Track Editor** is a lightweight Napari widget specifically designed for the [Cell Tracking Challenge (CTC)](http://celltrackingchallenge.net/) data format.

It empowers researchers to **visualize, correct, and edit** cell tracking results (Segmentation + Lineage) through an intuitive graphical interface. Handle track merging, splitting, and lineage tree construction in real-time with 3D track visualization.

---

## ✨ Core Features

* **🚀 High-Speed Async Loading**: Multi-threaded data loading for massive 3D/4D TIFF stacks and raw image sequences.
* **🛠️ Complete Editing Toolkit**:
* **Merge**: Combine over-segmented fragments while maintaining lineage hierarchy.
* **Split**: Break a track at a specific frame and assign a new ID (correcting under-segmentation).
* **Link Lineage**: Manually establish Parent-Child relationships for cell division events.
* **Delete**: Supports global physical deletion or truncation from the current frame onwards.


* **🌲 3D Lineage Visualization**: Real-time updates to Napari `Tracks` layers, displaying division trees and movement paths.
* **📝 Live Log Console**: Built-in terminal to track operation history, warnings, and processing progress.
* **💾 Standard CTC Export**: Automatically generates compliant `man_seg*.tif` sequences and `man_track.txt` lineage files.

---

## 📦 Dependencies

Ensure you have the following libraries installed in your Python environment:

```bash
pip install napari[all] numpy pandas tifffile imageio scipy qtpy

```

*Tested on Python 3.9+*

---

## 📂 Data Format (CTC Standards)

This tool follows the Cell Tracking Challenge standard file structure:

```text
Dataset/
├── 01/                  <-- Raw Images (Optional)
│   ├── t000.tif
│   └── ...
└── 01_RES/              <-- Segmentation Masks (Target Folder)
    ├── mask000.tif      (or man_seg000.tif)
    ├── ...
    └── res_track.txt    (or man_track.txt) Lineage File

```

* **Masks**: 16-bit integer images (0 for background, unique IDs for cells).
* **Track TXT**: Space-separated file with `L B E P` (Label, Begin, End, Parent).

---

## 📖 User Guide

### 1. Launching the Editor

Run your script to open Napari:

```python
python main.py

```

The widget will appear in the right dock area.

### 2. Data Import

1. **Mask Path**: Browse to the folder containing your segmentation `.tif` files.
2. **Raw Path**: (Optional) If left empty, the tool auto-detects `01` or `02` folders in the parent directory.
3. Click **`🚀 Async Load Data`**.

### 3. Editing Operations

| Feature | UI Section | Description |
| --- | --- | --- |
| **Merge** | `[3. Editing Tools]` | Enter **Keep ID** and **Merge ID**. Click `🤝 Merge Tracks` to unify pixels and lineage. |
| **Split** | `[3. Editing Tools]` | Enter **Split ID** and **Start T**. Click `✂️ Split as New` to branch the track into a new ID from that frame. |
| **Link** | `[3. Editing Tools]` | Enter **Parent P** and **Child A/B**. Click `🔗 Link Lineage` to create division branches. |
| **Delete** | `[4. System & Helpers]` | `❌ Delete All`: Erase ID from all frames.<br>

<br>`✂️ Delete After`: Erase ID from current frame to the end. |
| **Undo** | `[4. System & Helpers]` | Click `↩️ Undo Action` to revert your last edit (supports 15-step history). |

### 4. Saving Results

* **💾 Save (Over)**: Saves to a `RES_modified` subfolder in your data directory.
* **📁 Save As...**: Select a custom directory for export.
* *Note: Saving automatically triggers a full re-scan to ensure `man_track.txt` is perfectly synced with the masks.*

---

## 📝 License

MIT License. (See LICENSE file for details).