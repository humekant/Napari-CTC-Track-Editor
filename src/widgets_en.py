import re
import numpy as np
import pandas as pd
import tifffile as tiff
import imageio.v3 as iio
import copy
from pathlib import Path
from datetime import datetime

import napari
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from scipy.ndimage import find_objects, center_of_mass
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QMessageBox,
    QLabel,
    QSpinBox,
    QPushButton,
    QFileDialog,
    QFrame,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
)

import src.core_logic as core_logic


class CTCEditorWidget(QWidget):
    """Napari Track Editor optimized for CTC format - English Version"""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # --- Core Data State ---
        self.lineage_data = {}
        self.track_stats = {}
        self.centroids_cache = {}
        self.frame_to_ids = {}
        self.data_path = None
        self.labels_layer = None
        self.history = []
        self.max_history = 15

        self._init_ui()
        self._connect_signals()

        # Initialize Log
        self.log_message("Plugin started. Waiting for data loading...", "info")

    def _init_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # 1. Data Import
        self.main_layout.addWidget(QLabel("<b>[ 1. Data Import ]</b>"))
        path_layout = QGridLayout()
        path_layout.addWidget(QLabel("Mask Path:"), 0, 0)
        self.edit_mask_path = QLineEdit()
        path_layout.addWidget(self.edit_mask_path, 0, 1)
        self.btn_browse_mask = QPushButton("Browse")
        path_layout.addWidget(self.btn_browse_mask, 0, 2)

        path_layout.addWidget(QLabel("Raw Path:"), 1, 0)
        self.edit_raw_path = QLineEdit()
        self.edit_raw_path.setPlaceholderText("Leave empty to auto-find 01/02 folder")
        path_layout.addWidget(self.edit_raw_path, 1, 1)
        self.btn_browse_raw = QPushButton("Browse")
        path_layout.addWidget(self.btn_browse_raw, 1, 2)

        self.btn_load = QPushButton("🚀 High-Speed Async Load")
        self.btn_load.setStyleSheet(
            "font-weight: bold; height: 32px; background-color: #2c3e50; color: white;"
        )

        path_layout.addWidget(self.btn_load, 2, 0, 1, 3)
        self.main_layout.addLayout(path_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        self._add_line()

        # 2. Navigation & Status
        self.main_layout.addWidget(QLabel("<b>[ 2. Status & Navigation ]</b>"))
        nav_layout = QGridLayout()
        nav_layout.addWidget(QLabel("🔍 Jump to ID:"), 0, 0)
        self.spin_jump_id = QSpinBox()
        self.spin_jump_id.setRange(0, 99999)
        nav_layout.addWidget(self.spin_jump_id, 0, 1)

        self.btn_go_first = QPushButton("⏮ First Frame")
        nav_layout.addWidget(self.btn_go_first, 1, 0)
        self.btn_next_id = QPushButton("⏭ Next ID")
        nav_layout.addWidget(self.btn_next_id, 1, 1)
        self.main_layout.addLayout(nav_layout)

        self.info_table = QTableWidget(0, 4)
        self.info_table.setHorizontalHeaderLabels(["ID", "Start", "End", "Parent"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.info_table.setFixedHeight(150)
        self.main_layout.addWidget(self.info_table)

        self._add_line()

        # 3 & 4. Editing Tools
        grid_container = QGridLayout()
        grid_container.addWidget(QLabel("<b>[ 3. Edit Tools ]</b>"), 0, 0)
        grid_container.addWidget(QLabel("Keep ID:"), 1, 0)
        self.spin_m_keep = QSpinBox()
        self.spin_m_keep.setRange(0, 99999)
        grid_container.addWidget(self.spin_m_keep, 1, 1)
        grid_container.addWidget(QLabel("Merge ID:"), 2, 0)
        self.spin_m_del = QSpinBox()
        self.spin_m_del.setRange(0, 99999)
        grid_container.addWidget(self.spin_m_del, 2, 1)
        self.btn_merge = QPushButton("🤝 Merge Tracks")
        grid_container.addWidget(self.btn_merge, 3, 0, 1, 2)

        grid_container.addWidget(QLabel("Split ID:"), 4, 0)
        self.spin_s_id = QSpinBox()
        self.spin_s_id.setRange(0, 99999)
        grid_container.addWidget(self.spin_s_id, 4, 1)
        grid_container.addWidget(QLabel("From T:"), 5, 0)
        self.spin_s_time = QSpinBox()
        self.spin_s_time.setRange(0, 99999)
        grid_container.addWidget(self.spin_s_time, 5, 1)
        self.btn_split = QPushButton("✂️ Split as New")
        grid_container.addWidget(self.btn_split, 6, 0, 1, 2)

        grid_container.addWidget(QLabel("Parent:"), 7, 0)
        self.spin_p = QSpinBox()
        self.spin_p.setRange(0, 99999)
        grid_container.addWidget(self.spin_p, 7, 1)
        grid_container.addWidget(QLabel("Child A:"), 8, 0)
        self.spin_c1 = QSpinBox()
        self.spin_c1.setRange(0, 99999)
        grid_container.addWidget(self.spin_c1, 8, 1)
        grid_container.addWidget(QLabel("Child B:"), 9, 0)
        self.spin_c2 = QSpinBox()
        self.spin_c2.setRange(0, 99999)
        grid_container.addWidget(self.spin_c2, 9, 1)
        self.btn_link = QPushButton("🔗 Link Lineage")
        grid_container.addWidget(self.btn_link, 10, 0, 1, 2)

        grid_container.addWidget(QLabel("<b>[ 4. System & Aux ]</b>"), 0, 2)
        grid_container.addWidget(QLabel("Target ID:"), 1, 2)
        self.spin_target_del = QSpinBox()
        self.spin_target_del.setRange(0, 99999)
        grid_container.addWidget(self.spin_target_del, 1, 3)
        self.btn_del_all = QPushButton("❌ Delete Track")
        grid_container.addWidget(self.btn_del_all, 2, 2, 1, 2)
        self.btn_del_after = QPushButton("✂️ Delete After T")
        grid_container.addWidget(self.btn_del_after, 3, 2, 1, 2)
        self.btn_undo = QPushButton("↩️ Undo Action")
        grid_container.addWidget(self.btn_undo, 5, 2, 1, 2)
        self.btn_refresh = QPushButton("✨ Refresh Tracks")
        grid_container.addWidget(self.btn_refresh, 6, 2, 1, 2)

        grid_container.addWidget(QLabel("<b>[ 5. Save ]</b>"), 8, 2)
        self.btn_save_over = QPushButton("💾 Overwrite RES")
        grid_container.addWidget(self.btn_save_over, 9, 2, 1, 2)
        self.btn_save_as = QPushButton("📁 Save As...")
        grid_container.addWidget(self.btn_save_as, 10, 2, 1, 2)

        self.main_layout.addLayout(grid_container)

        self._add_line()

        # --- Console Logs ---
        self.main_layout.addWidget(QLabel("<b>[ 6. System Logs ]</b>"))
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(120)
        self.log_console.setStyleSheet(
            """
            background-color: #1e1e1e; 
            color: #cccccc; 
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 11px;
            border: 1px solid #444;
            padding: 4px;
        """
        )
        self.main_layout.addWidget(self.log_console)
        self.main_layout.addStretch()

    def _add_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)

    def log_message(self, message, level="info"):
        """Output system logs with HTML color support"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": "#cccccc",
            "success": "#4ec9b0",
            "warning": "#ce9178",
            "error": "#f44747",
        }
        color = colors.get(level, "#cccccc")
        html_msg = f'<span style="color:#666666;">[{timestamp}]</span> <span style="color:{color};">{message}</span>'

        self.log_console.appendHtml(html_msg)
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )
        print(f"[{level.upper()}] {message}")

    def _connect_signals(self):
        self.btn_browse_mask.clicked.connect(
            lambda: self.edit_mask_path.setText(
                QFileDialog.getExistingDirectory(self, "Select Mask Folder")
            )
        )
        self.btn_browse_raw.clicked.connect(
            lambda: self.edit_raw_path.setText(
                QFileDialog.getExistingDirectory(self, "Select Raw Folder")
            )
        )
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.spin_jump_id.valueChanged.connect(self._on_jump_id_changed)
        self.btn_go_first.clicked.connect(self._go_to_first_frame)
        self.btn_next_id.clicked.connect(self._jump_to_next_id)
        self.info_table.cellClicked.connect(self._on_table_click)
        self.btn_merge.clicked.connect(self.merge_tracks_action)
        self.btn_split.clicked.connect(self.split_track_action)
        self.btn_link.clicked.connect(self.link_lineage_batch)
        self.btn_del_all.clicked.connect(self.delete_track_globally)
        self.btn_del_after.clicked.connect(self.delete_track_afterwards)
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_refresh.clicked.connect(self.update_tracks_layer)
        self.btn_save_over.clicked.connect(self.save_overwrite)
        self.btn_save_as.clicked.connect(self.save_as)
        self.viewer.dims.events.current_step.connect(self.update_info_table)

    def _read_image_folder(self, folder_path):
        """Read images: .tif with tifffile, others with imageio"""
        if not folder_path or not Path(folder_path).exists():
            return None

        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        files = sorted(
            [f for f in Path(folder_path).glob("*") if f.suffix.lower() in exts]
        )
        if not files:
            return None

        img_list = []
        for f in files:
            ext = f.suffix.lower()
            try:
                if ext in {".tif", ".tiff"}:
                    img = tiff.imread(str(f))
                else:
                    img = iio.imread(str(f))
                img_list.append(img)
            except Exception as e:
                self.log_message(f"Failed to read {f.name}: {e}", "error")
                continue

        if not img_list:
            return None
        return np.stack(img_list)

    @thread_worker
    def _full_load_worker(self, mask_path, raw_path_input):
        mask_stack = core_logic.read_image_folder(mask_path)
        if mask_stack is None:
            return None

        raw_stack = None
        mask_p = Path(mask_path).parent
        if raw_path_input and Path(raw_path_input).exists():
            raw_stack = core_logic.read_image_folder(raw_path_input)
        else:
            match = re.match(r"^(\d+)", mask_p.name)
            if match:
                auto_path = mask_p.parent / match.group(1)
                if auto_path.exists():
                    raw_stack = core_logic.read_image_folder(auto_path)

        # Statistical calculations
        stats, cents, f2ids = {}, {}, {t: [] for t in range(len(mask_stack))}
        for t in range(len(mask_stack)):
            f2ids[t] = core_logic.scan_frame_for_stats(t, mask_stack[t], stats, cents)
            yield ("progress", int((t + 1) / len(mask_stack) * 100))

        # Read Lineage
        lin = {}
        for fn in ["res_track.txt", "man_track.txt"]:
            txt = Path(mask_path) / fn
            if txt.exists():
                try:
                    df = pd.read_csv(txt, sep=r"\s+", header=None)
                    for _, r in df.iterrows():
                        if int(r[3]) > 0:
                            lin[int(r[0])] = int(r[3])
                except:
                    pass
                break

        return mask_stack, raw_stack, stats, cents, f2ids, lin

    def _on_load_clicked(self):
        p_str = self.edit_mask_path.text()
        if not p_str:
            self.log_message("Mask path not selected, load cancelled.", "warning")
            return

        self.data_path = Path(p_str)
        raw_p = Path(self.edit_raw_path.text()) if self.edit_raw_path.text() else None

        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.log_message(f"Loading data from {self.data_path.name}...", "info")

        worker = self._full_load_worker(self.data_path, raw_p)
        worker.yielded.connect(lambda p: self.progress_bar.setValue(p[1]))
        worker.returned.connect(self._on_load_finished)
        worker.start()

    def _on_load_finished(self, result):
        if result is None:
            self.log_message("No valid image files found! Check your path.", "error")
            show_info("❌ No valid images found!")
            self.btn_load.setEnabled(True)
            return

        mask, raw, stats, cents, f2ids, lin = result

        for name in ["RawImage", "SegLabels", "LineageTracks", "ID_Labels"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        if raw is not None:
            self.viewer.add_image(
                raw, name="RawImage", blending="additive", opacity=0.8
            )
            self.log_message("Raw image layer loaded", "info")
        else:
            self.log_message("Raw images not found, displaying Mask only", "warning")

        self.labels_layer = self.viewer.add_labels(mask, name="SegLabels", opacity=0.5)
        self.labels_layer.show_label_index = True

        self.track_stats, self.centroids_cache, self.frame_to_ids, self.lineage_data = (
            stats,
            cents,
            f2ids,
            lin,
        )

        self.btn_load.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.history = []
        self.update_info_table()
        self.update_tracks_layer()

        self.log_message(
            f"Loading complete: {len(mask)} frames, {len(stats)} cell IDs", "success"
        )
        if lin:
            self.log_message(f"Loaded {len(lin)} division records", "success")
        show_info("✅ Data loading complete")

    def update_info_table(self, event=None):
        if not self.frame_to_ids:
            return
        t = self.viewer.dims.current_step[0]
        ids = self.frame_to_ids.get(t, [])
        self.info_table.setRowCount(len(ids))
        for i, tid in enumerate(sorted(ids)):
            s_e = self.track_stats.get(tid, [0, 0])
            self.info_table.setItem(i, 0, QTableWidgetItem(str(tid)))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(s_e[0])))
            self.info_table.setItem(i, 2, QTableWidgetItem(str(s_e[1])))
            self.info_table.setItem(
                i, 3, QTableWidgetItem(str(self.lineage_data.get(tid, 0)))
            )
            row_height = self.info_table.rowHeight(0) if len(ids) > 0 else 30
            total_height = min(400, (len(ids) * row_height) + 40)
            self.info_table.setFixedHeight(total_height)

    def _on_table_click(self, row, col):
        try:
            val = int(self.info_table.item(row, 0).text())
            self.spin_jump_id.setValue(val)
        except:
            pass

    def _on_jump_id_changed(self, val):
        if val == 0 or not self.labels_layer:
            return
        self.labels_layer.selected_label = val
        t = self.viewer.dims.current_step[0]
        if (t, val) in self.centroids_cache:
            self.viewer.camera.center = self.centroids_cache[(t, val)]

    def _go_to_first_frame(self):
        self.viewer.dims.set_current_step(0, 0)
        self.log_message("Jumped to first frame", "info")

    def _jump_to_next_id(self):
        if not self.track_stats:
            return
        all_ids = sorted(self.track_stats.keys())
        curr = self.spin_jump_id.value()
        next_id = next((i for i in all_ids if i > curr), all_ids[0])
        self.spin_jump_id.setValue(next_id)
        self.viewer.dims.set_current_step(0, self.track_stats[next_id][0])

    def update_tracks_layer(self):
        """Update track visuals and point layer with custom ID labels (Parent-Child)"""
        if not self.centroids_cache:
            for name in ["LineageTracks", "ID_Labels"]:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            return

        # 1. Track Layer Data (ID, T, Y, X)
        pts_list = []
        for (t, tid), (y, x) in self.centroids_cache.items():
            pts_list.append([int(tid), int(t), y, x])

        pts = np.array(pts_list)
        if len(pts) == 0:
            return

        graph = {
            int(c): [int(p)]
            for c, p in self.lineage_data.items()
            if c in pts[:, 0] and p in pts[:, 0]
        }

        if "LineageTracks" in self.viewer.layers:
            layer_tr = self.viewer.layers["LineageTracks"]
            layer_tr.data = pts
            layer_tr.graph = graph
        else:
            layer_tr = self.viewer.add_tracks(
                pts, graph=graph, name="LineageTracks", tail_length=30
            )

        layer_tr.display_id = False

        # 2. Text Label Data
        label_coords = []
        display_texts = []
        for (t, tid), (y, x) in self.centroids_cache.items():
            label_coords.append([t, y, x])
            tid_int = int(tid)
            parent_id = int(self.lineage_data.get(tid_int, 0))
            txt = f"{parent_id}-{tid_int}" if parent_id > 0 else f"{tid_int}"
            display_texts.append(txt)

        pt_props = {"label_text": np.array(display_texts)}

        # 3. Update or create Points Layer for labels
        if "ID_Labels" in self.viewer.layers:
            layer_lab = self.viewer.layers["ID_Labels"]
            layer_lab.data = np.array(label_coords)
            layer_lab.properties = pt_props
            layer_lab.text = {
                "string": "{label_text}",
                "color": "white",
                "size": 10,
                "anchor": "upper_left",
            }
        else:
            self.viewer.add_points(
                np.array(label_coords),
                properties=pt_props,
                text={
                    "string": "{label_text}",
                    "color": "white",
                    "size": 10,
                    "anchor": "upper_left",
                },
                name="ID_Labels",
                size=0,
                face_color="transparent",
            )

        layer_tr.refresh()
        if "ID_Labels" in self.viewer.layers:
            self.viewer.layers["ID_Labels"].refresh()

    def _save_history(self):
        if not self.labels_layer:
            return
        self.history.append(
            {
                "labels_data": self.labels_layer.data.copy(),
                "lineage_data": copy.deepcopy(self.lineage_data),
                "track_stats": copy.deepcopy(self.track_stats),
                "centroids_cache": copy.deepcopy(self.centroids_cache),
                "frame_to_ids": copy.deepcopy(self.frame_to_ids),
            }
        )
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo_action(self):
        if not self.history:
            self.log_message("No history to undo", "warning")
            return
        sn = self.history.pop()
        self.labels_layer.data = sn["labels_data"]
        self.labels_layer.refresh()
        self.lineage_data, self.track_stats, self.centroids_cache, self.frame_to_ids = (
            sn["lineage_data"],
            sn["track_stats"],
            sn["centroids_cache"],
            sn["frame_to_ids"],
        )
        self.update_info_table()
        self.update_tracks_layer()
        self.log_message("Action undone", "info")

    def merge_tracks_action(self):
        id_keep, id_src = self.spin_m_keep.value(), self.spin_m_del.value()
        if id_keep == 0 or id_src == 0 or id_keep == id_src:
            self.log_message("Merge failed: Invalid or duplicate IDs", "error")
            return

        self._save_history()

        # Modify pixels
        self.labels_layer.data[self.labels_layer.data == id_src] = id_keep
        self.labels_layer.refresh()

        # Inherit lineage
        for child, parent in list(self.lineage_data.items()):
            if parent == id_src:
                self.lineage_data[child] = id_keep
        if id_src in self.lineage_data:
            self.lineage_data.pop(id_src)

        self._refresh_cache_from_memory()
        self.update_info_table()
        self.update_tracks_layer()
        self.log_message(f"Merge successful: ID {id_src} into ID {id_keep}", "success")

    def split_track_action(self):
        old_id, t_start = self.spin_s_id.value(), self.spin_s_time.value()
        if old_id == 0 or old_id not in self.track_stats:
            self.log_message("Split failed: ID does not exist", "error")
            return

        self._save_history()

        # Generate new ID
        new_id = int(max(self.track_stats.keys()) + 1)
        data_view = self.labels_layer.data[t_start:, ...]
        mask_to_change = data_view == old_id

        if not np.any(mask_to_change):
            self.log_message(
                f"Warning: No pixels found for ID {old_id} at T >= {t_start}", "warning"
            )
            return

        data_view[mask_to_change] = new_id
        self.labels_layer.refresh()

        # Transfer children
        updated_children = []
        for child_id, parent_id in list(self.lineage_data.items()):
            if parent_id == old_id:
                child_start_frame = self.track_stats.get(child_id, [0, 0])[0]
                if child_start_frame >= t_start:
                    self.lineage_data[child_id] = new_id
                    updated_children.append(str(child_id))

        self._refresh_cache_from_memory()
        self.update_info_table()
        self.update_tracks_layer()

        msg = f"Split successful: ID {old_id} -> New ID {new_id} (at Frame {t_start})"
        if updated_children:
            msg += f" | Children transferred: {', '.join(updated_children)}"
        self.log_message(msg, "success")

    def _refresh_cache_from_memory(self):
        """Full re-calculation of caches from memory data"""
        self.log_message("Refreshing memory caches...", "info")

        mask_data = self.labels_layer.data
        num_frames = mask_data.shape[0]

        new_stats = {}
        new_cents = {}
        new_f2ids = {t: [] for t in range(num_frames)}

        for t in range(num_frames):
            frame = mask_data[t]
            uids = np.unique(frame)
            uids = uids[uids > 0]

            if len(uids) == 0:
                continue

            new_f2ids[t] = [int(u) for u in uids]
            centers = center_of_mass(frame, frame, uids)

            for idx, uid in enumerate(uids):
                uid = int(uid)
                y, x = centers[idx]
                new_cents[(t, uid)] = (y, x)
                if uid not in new_stats:
                    new_stats[uid] = [t, t]
                else:
                    new_stats[uid][1] = t

        self.track_stats = new_stats
        self.centroids_cache = new_cents
        self.frame_to_ids = new_f2ids

        # Cleanup lineage for removed IDs
        valid_ids = set(new_stats.keys())
        keys_to_remove = [k for k in self.lineage_data if k not in valid_ids]
        for k in keys_to_remove:
            del self.lineage_data[k]

        self.log_message("Memory cache refreshed", "info")

    def delete_track_globally(self):
        tid = self.spin_target_del.value()
        if tid == 0:
            return
        self._save_history()

        self.labels_layer.data[self.labels_layer.data == tid] = 0
        self.labels_layer.refresh()

        if tid in self.lineage_data:
            del self.lineage_data[tid]

        self._refresh_cache_from_memory()
        self.update_info_table()
        self.update_tracks_layer()
        self.log_message(f"Physically deleted ID {tid}", "warning")

    def delete_track_afterwards(self):
        tid, t_curr = self.spin_target_del.value(), self.viewer.dims.current_step[0]
        if tid == 0:
            return
        self._save_history()

        self.labels_layer.data[t_curr:][self.labels_layer.data[t_curr:] == tid] = 0
        self.labels_layer.refresh()

        self._refresh_cache_from_memory()
        self.update_info_table()
        self.update_tracks_layer()
        self.log_message(f"Truncated ID {tid} starting from Frame {t_curr}", "warning")

    def link_lineage_batch(self):
        p = self.spin_p.value()
        children = [self.spin_c1.value(), self.spin_c2.value()]

        if p == 0:
            show_info("❌ Parent ID cannot be 0")
            self.log_message("Lineage failed: Parent ID not specified", "error")
            return

        self._save_history()
        count = 0
        linked_children = []
        for c in children:
            if c > 0 and c != p:
                self.lineage_data[int(c)] = int(p)
                linked_children.append(str(c))
                count += 1

        if count > 0:
            msg = f"Lineage linked: Parent {p} -> Children {', '.join(linked_children)}"
            show_info(f"✅ {msg}")
            self.log_message(msg, "success")
            self.update_info_table()
            self.update_tracks_layer()
        else:
            self.log_message("Lineage failed: No valid child IDs specified", "warning")

    def _execute_save(self, output_dir):
        """Deep sync save: write images and L B E P text files"""
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            self.log_message(f"Saving data to: {output_dir} ...", "info")
            show_info("Executing deep sync save (with lineage)...")

            mask_data = self.labels_layer.data
            num_frames = mask_data.shape[0]

            new_stats = {}
            for t in range(num_frames):
                frame = mask_data[t]
                tiff.imwrite(
                    output_dir / f"man_seg{t:04d}.tif",
                    frame.astype(np.uint16),
                    compression="zlib",
                )

                uids = np.unique(frame)
                uids = uids[uids > 0]
                for uid in uids:
                    uid = int(uid)
                    if uid not in new_stats:
                        new_stats[uid] = [t, t]
                    else:
                        new_stats[uid][1] = t

            tlines = []
            div_count = 0
            for uid in sorted(new_stats.keys()):
                start, end = new_stats[uid]
                parent = int(self.lineage_data.get(uid, 0))
                if parent != 0 and parent not in new_stats:
                    self.log_message(
                        f"Warning: Orphan ID {uid} (Parent {parent} missing)", "warning"
                    )
                    parent = 0
                if parent > 0:
                    div_count += 1
                tlines.append([uid, start, end, parent])

            df = pd.DataFrame(tlines)
            df.to_csv(output_dir / "man_track.txt", sep=" ", index=False, header=False)

            self.track_stats = new_stats
            msg = (
                f"✅ Data Sync Successful!\n"
                f"Dir: {output_dir.name}\n"
                f"Total Tracks: {len(tlines)}\n"
                f"Divisions: {div_count}"
            )
            self.log_message(
                f"Save complete! Tracks: {len(tlines)}, Divisions: {div_count}",
                "success",
            )
            QMessageBox.information(self, "Save Confirmed", msg)

        except Exception as e:
            err_msg = f"Save failed: {str(e)}"
            self.log_message(err_msg, "error")
            QMessageBox.critical(self, "Error", err_msg)

    def save_overwrite(self):
        if not self.data_path:
            self.log_message("No data loaded, cannot overwrite.", "warning")
            return
        self._execute_save(self.data_path.parent / "RES_modified")

    def save_as(self):
        p = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if p:
            self._execute_save(Path(p))
