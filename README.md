# Napari CTC Track Editor

A lightweight, high-performance Napari plugin designed for manual correction and editing of cell tracking results in **CTC (Cell Tracking Challenge)** format.

## 🚀 Features

* **Asynchronous Loading:** High-speed loading of large mask stacks and raw images.
* **Lineage Visualization:** Specialized 3D track layer with "Parent-Child" ID labeling (e.g., `8-21`).
* **Editing Toolkit:**
* **Merge:** Join broken tracks while maintaining lineage records.
* **Split:** Cut a track at a specific frame to create a new cell ID.
* **Link Lineage:** Manually establish division relationships (Parent → Children).
* **Delete:** Support for global deletion or truncation from the current frame.


* **History Management:** Support for up to 15 steps of Undo.
* **Standard CTC Output:** Automatically generates `man_segXXXX.tif` masks and `man_track.txt` lineage files.

## 🛠 Installation

1. Ensure you have Python 3.9+ and [Napari](https://napari.org/) installed.
2. Clone this repository:
```bash
git clone https://github.com/yourusername/napari-ctc-editor.git
cd napari-ctc-editor

```

3. Install dependencies:
```bash
pip install numpy pandas tifffile imageio napari[all] qtpy scipy

```


## 📖 Usage

1. Run the script: `python main.py`.
2. **Data Import:**
* Select the **Mask** folder (containing `.tif` files).
* (Optional) Select the **Raw** image folder.


3. **Navigate:** Use the "Jump to ID" box or click the info table to locate specific cells.
4. **Edit:** Use the "Editing Tools" to correct segmentation or tracking errors.
5. **Save:** Click "Save As" to export results in standard CTC format.