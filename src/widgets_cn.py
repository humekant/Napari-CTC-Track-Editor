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
    QPlainTextEdit,  # 新增：用于日志显示
)

import src.core_logic as core_logic


class CTCEditorWidget(QWidget):
    """针对 CTC 格式优化的 Napari 轨迹编辑器 - 带日志增强版"""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # --- 核心数据状态 ---
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

        # 初始化日志
        self.log_message("插件已启动，等待数据加载...", "info")

    def _init_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # 1. 数据导入
        self.main_layout.addWidget(QLabel("<b>[ 1. 数据导入 ]</b>"))
        path_layout = QGridLayout()
        path_layout.addWidget(QLabel("Mask 路径:"), 0, 0)
        self.edit_mask_path = QLineEdit()
        path_layout.addWidget(self.edit_mask_path, 0, 1)
        self.btn_browse_mask = QPushButton("浏览")
        path_layout.addWidget(self.btn_browse_mask, 0, 2)

        path_layout.addWidget(QLabel("Raw 路径:"), 1, 0)
        self.edit_raw_path = QLineEdit()
        self.edit_raw_path.setPlaceholderText("留空则自动寻找同级 01/02 文件夹")
        path_layout.addWidget(self.edit_raw_path, 1, 1)
        self.btn_browse_raw = QPushButton("浏览")
        path_layout.addWidget(self.btn_browse_raw, 1, 2)

        self.btn_load = QPushButton("🚀 全异步高速加载数据")
        self.btn_load.setStyleSheet(
            "font-weight: bold; height: 32px; background-color: #2c3e50; color: white;"
        )

        path_layout.addWidget(self.btn_load, 2, 0, 1, 3)
        self.main_layout.addLayout(path_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        self._add_line()

        # 2. 定位与状态
        self.main_layout.addWidget(QLabel("<b>[ 2. 状态查询与定位 ]</b>"))
        nav_layout = QGridLayout()
        nav_layout.addWidget(QLabel("🔍 定位 ID:"), 0, 0)
        self.spin_jump_id = QSpinBox()
        self.spin_jump_id.setRange(0, 99999)
        nav_layout.addWidget(self.spin_jump_id, 0, 1)

        self.btn_go_first = QPushButton("⏮ 回到首帧")
        nav_layout.addWidget(self.btn_go_first, 1, 0)
        self.btn_next_id = QPushButton("⏭ 下一个 ID")
        nav_layout.addWidget(self.btn_next_id, 1, 1)
        self.main_layout.addLayout(nav_layout)

        self.info_table = QTableWidget(0, 4)
        self.info_table.setHorizontalHeaderLabels(["ID", "Start", "End", "Parent"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.info_table.setFixedHeight(150)
        self.main_layout.addWidget(self.info_table)

        self._add_line()

        # 3 & 4. 双列编辑
        grid_container = QGridLayout()
        grid_container.addWidget(QLabel("<b>[ 3. 修改工具 ]</b>"), 0, 0)
        grid_container.addWidget(QLabel("保留 ID:"), 1, 0)
        self.spin_m_keep = QSpinBox()
        self.spin_m_keep.setRange(0, 99999)
        grid_container.addWidget(self.spin_m_keep, 1, 1)
        grid_container.addWidget(QLabel("被并 ID:"), 2, 0)
        self.spin_m_del = QSpinBox()
        self.spin_m_del.setRange(0, 99999)
        grid_container.addWidget(self.spin_m_del, 2, 1)
        self.btn_merge = QPushButton("🤝 合并轨迹")
        grid_container.addWidget(self.btn_merge, 3, 0, 1, 2)
        grid_container.addWidget(QLabel("拆分 ID:"), 4, 0)
        self.spin_s_id = QSpinBox()
        self.spin_s_id.setRange(0, 99999)
        grid_container.addWidget(self.spin_s_id, 4, 1)
        grid_container.addWidget(QLabel("起始帧:"), 5, 0)
        self.spin_s_time = QSpinBox()
        self.spin_s_time.setRange(0, 99999)
        grid_container.addWidget(self.spin_s_time, 5, 1)
        self.btn_split = QPushButton("✂️ 设为新细胞")
        grid_container.addWidget(self.btn_split, 6, 0, 1, 2)
        grid_container.addWidget(QLabel("父 P:"), 7, 0)
        self.spin_p = QSpinBox()
        self.spin_p.setRange(0, 99999)
        grid_container.addWidget(self.spin_p, 7, 1)
        grid_container.addWidget(QLabel("子 A:"), 8, 0)
        self.spin_c1 = QSpinBox()
        self.spin_c1.setRange(0, 99999)
        grid_container.addWidget(self.spin_c1, 8, 1)
        grid_container.addWidget(QLabel("子 B:"), 9, 0)
        self.spin_c2 = QSpinBox()
        self.spin_c2.setRange(0, 99999)
        grid_container.addWidget(self.spin_c2, 9, 1)
        self.btn_link = QPushButton("🔗 建立谱系")
        grid_container.addWidget(self.btn_link, 10, 0, 1, 2)

        grid_container.addWidget(QLabel("<b>[ 4. 系统与辅助 ]</b>"), 0, 2)
        grid_container.addWidget(QLabel("目标 ID:"), 1, 2)
        self.spin_target_del = QSpinBox()
        self.spin_target_del.setRange(0, 99999)
        grid_container.addWidget(self.spin_target_del, 1, 3)
        self.btn_del_all = QPushButton("❌ 物理全删")
        grid_container.addWidget(self.btn_del_all, 2, 2, 1, 2)
        self.btn_del_after = QPushButton("✂️ 截断删除")
        grid_container.addWidget(self.btn_del_after, 3, 2, 1, 2)
        self.btn_undo = QPushButton("↩️ 撤销上一步")
        grid_container.addWidget(self.btn_undo, 5, 2, 1, 2)
        self.btn_refresh = QPushButton("✨ 刷新 3D 轨迹")
        grid_container.addWidget(self.btn_refresh, 6, 2, 1, 2)
        grid_container.addWidget(QLabel("<b>[ 5. 保存 ]</b>"), 8, 2)
        self.btn_save_over = QPushButton("💾 覆盖保存")
        grid_container.addWidget(self.btn_save_over, 9, 2, 1, 2)
        self.btn_save_as = QPushButton("📁 另存为...")
        grid_container.addWidget(self.btn_save_as, 10, 2, 1, 2)

        self.main_layout.addLayout(grid_container)

        self._add_line()

        # --- 新增：日志控制台 ---
        self.main_layout.addWidget(QLabel("<b>[ 6. 系统日志 ]</b>"))
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(120)  # 固定高度
        # 设置样式：深色背景，等宽字体，像 Terminal
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
        """系统日志输出函数，支持 HTML 颜色标记"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 定义颜色
        colors = {
            "info": "#cccccc",  # 灰色/白色
            "success": "#4ec9b0",  # 青绿色
            "warning": "#ce9178",  # 橙色
            "error": "#f44747",  # 红色
        }
        color = colors.get(level, "#cccccc")

        # 构造 HTML 字符串
        html_msg = f'<span style="color:#666666;">[{timestamp}]</span> <span style="color:{color};">{message}</span>'

        self.log_console.appendHtml(html_msg)
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )
        # 同时打印到 Python 控制台作为备份
        print(f"[{level.upper()}] {message}")

    def _connect_signals(self):
        self.btn_browse_mask.clicked.connect(
            lambda: self.edit_mask_path.setText(
                QFileDialog.getExistingDirectory(self, "选择 Mask 文件夹")
            )
        )
        self.btn_browse_raw.clicked.connect(
            lambda: self.edit_raw_path.setText(
                QFileDialog.getExistingDirectory(self, "选择 Raw 文件夹")
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
        """核心读图功能：.tif 使用 tifffile 读取，其他格式使用 imageio"""
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
                    # 使用 tifffile 读取 (即你代码中 import 的 tiff)
                    img = tiff.imread(str(f))
                else:
                    # 使用 imageio 读取
                    img = iio.imread(str(f))

                img_list.append(img)
            except Exception as e:
                self.log_message(f"读取文件 {f.name} 失败: {e}", "error")  # Log error
                continue

        if not img_list:
            return None

        return np.stack(img_list)

    @thread_worker
    def _full_load_worker(self, mask_path, raw_path_input):
        # 1. 调用 core_logic 加载 Mask
        mask_stack = core_logic.read_image_folder(mask_path)
        if mask_stack is None:
            return None

        # 2. 推断 Raw (保持你原来的路径推断逻辑)
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

        # 3. 统计计算：调用 core_logic 处理每一帧
        stats, cents, f2ids = {}, {}, {t: [] for t in range(len(mask_stack))}
        for t in range(len(mask_stack)):
            # 调用 core_logic 的函数
            f2ids[t] = core_logic.scan_frame_for_stats(t, mask_stack[t], stats, cents)

            # --- 重点：必须在这里 yield 进度，否则会报 AttributeError ---
            yield ("progress", int((t + 1) / len(mask_stack) * 100))

        # 4. 读取 Lineage (保持你原来的 TXT 读取逻辑)
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
            self.log_message("未选择 Mask 路径，加载取消", "warning")
            return

        self.data_path = Path(p_str)
        raw_p = Path(self.edit_raw_path.text()) if self.edit_raw_path.text() else None

        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.log_message(f"开始从 {self.data_path.name} 加载数据...", "info")

        worker = self._full_load_worker(self.data_path, raw_p)
        worker.yielded.connect(lambda p: self.progress_bar.setValue(p[1]))
        worker.returned.connect(self._on_load_finished)
        worker.start()

    def _on_load_finished(self, result):
        if result is None:
            self.log_message("未找到有效图像文件！请检查路径。", "error")
            show_info("❌ 未找到有效图像文件！")
            self.btn_load.setEnabled(True)
            return

        mask, raw, stats, cents, f2ids, lin = result

        for name in ["RawImage", "SegLabels", "LineageTracks"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        if raw is not None:
            self.viewer.add_image(
                raw, name="RawImage", blending="additive", opacity=0.8
            )
            self.log_message("已加载 Raw 原始图像层", "info")
        else:
            self.log_message("未找到 Raw 图像，仅显示 Mask", "warning")

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
            f"数据加载完成: 共 {len(mask)} 帧, {len(stats)} 个细胞 ID", "success"
        )
        if lin:
            self.log_message(f"已加载 {len(lin)} 条分裂记录", "success")
        show_info("✅ 数据加载完成")

    # --- 辅助逻辑 (与上个版本保持一致) ---
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
        self.log_message("跳转至首帧", "info")

    def _jump_to_next_id(self):
        if not self.track_stats:
            return
        all_ids = sorted(self.track_stats.keys())
        curr = self.spin_jump_id.value()
        next_id = next((i for i in all_ids if i > curr), all_ids[0])
        self.spin_jump_id.setValue(next_id)
        self.viewer.dims.set_current_step(0, self.track_stats[next_id][0])

    def update_tracks_layer(self):
        """更新轨迹显示，并使用辅助点图层显示自定义 ID (父-子)"""
        if not self.centroids_cache:
            for name in ["LineageTracks", "ID_Labels"]:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            return

        # 1. 轨迹图层数据 (ID, T, Y, X)
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

        # 更新或创建轨迹层 (仅显示线条)
        if "LineageTracks" in self.viewer.layers:
            layer_tr = self.viewer.layers["LineageTracks"]
            layer_tr.data = pts
            layer_tr.graph = graph
        else:
            layer_tr = self.viewer.add_tracks(
                pts, graph=graph, name="LineageTracks", tail_length=30
            )

        layer_tr.display_id = False  # 彻底关掉那个带 .0 的默认 ID

        # 2. 构造文字标签数据
        label_coords = []
        display_texts = []
        for (t, tid), (y, x) in self.centroids_cache.items():
            label_coords.append([t, y, x])  # 坐标为 [T, Y, X]

            tid_int = int(tid)
            parent_id = int(self.lineage_data.get(tid_int, 0))
            # 拼接字符串：父-子 (如 8-21) 或 仅子 (如 8)
            txt = f"{parent_id}-{tid_int}" if parent_id > 0 else f"{tid_int}"
            display_texts.append(txt)

        pt_props = {"label_text": np.array(display_texts)}

        # 3. 更新或创建点图层 (专门负责文字显示)
        if "ID_Labels" in self.viewer.layers:
            layer_lab = self.viewer.layers["ID_Labels"]
            layer_lab.data = np.array(label_coords)
            layer_lab.properties = pt_props
            # 重新设置 text 字典确保刷新，移除 translation 避免 IndexError
            layer_lab.text = {
                "string": "{label_text}",
                "color": "white",
                "size": 10,
                "anchor": "upper_left",  # 使用锚点代替 translation 实现偏移
            }
        else:
            # 使用最保守的参数组合，避开 edge_color 等报错
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
                size=0,  # 点本身不可见
                face_color="transparent",
            )

        # 4. 强制重绘
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
            self.log_message("没有可撤销的历史记录", "warning")
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
        self.log_message("已撤销上一步操作", "info")

    # --- [修改 3]：更新合并操作，使用新的刷新函数 ---
    def merge_tracks_action(self):
        id_keep, id_src = self.spin_m_keep.value(), self.spin_m_del.value()
        if id_keep == 0 or id_src == 0 or id_keep == id_src:
            self.log_message("合并失败：ID 无效或相同", "error")
            return

        self._save_history()

        # 修改像素
        self.labels_layer.data[self.labels_layer.data == id_src] = id_keep
        self.labels_layer.refresh()

        # 继承分裂关系：如果被合并的 id_src 有子节点，现在归 id_keep
        for child, parent in list(self.lineage_data.items()):
            if parent == id_src:
                self.lineage_data[child] = id_keep

        # 如果 id_src 自己有父节点，逻辑比较复杂，通常合并后 id_keep 保持原父节点
        # 这里简单处理：移除 id_src 的记录
        if id_src in self.lineage_data:
            self.lineage_data.pop(id_src)

        # [重要] 调用新的全量刷新，确保 id_src 的标签消失，id_keep 的轨迹延长
        self._refresh_cache_from_memory()

        self.update_info_table()
        self.update_tracks_layer()
        self.log_message(f"合并成功: 将 ID {id_src} 合并入 ID {id_keep}", "success")

    # --- [修改 1]：修复拆分逻辑，增加子代继承转移 ---
    def split_track_action(self):
        old_id, t_start = self.spin_s_id.value(), self.spin_s_time.value()
        if old_id == 0 or old_id not in self.track_stats:
            self.log_message("拆分失败：ID 不存在", "error")
            return

        # 1. 保存历史用于撤销
        self._save_history()

        # 2. 生成新 ID 并修改像素
        new_id = int(max(self.track_stats.keys()) + 1)

        # 获取从 t_start 开始的数据切片
        # 注意：这里我们修改的是内存中的数据，不会立即影响磁盘文件
        data_view = self.labels_layer.data[t_start:, ...]
        mask_to_change = data_view == old_id

        if not np.any(mask_to_change):
            self.log_message(
                f"警告：在帧 {t_start} 之后未找到 ID {old_id} 的像素", "warning"
            )
            return

        data_view[mask_to_change] = new_id
        self.labels_layer.refresh()  # 刷新图层像素显示

        # 3. [核心修复]：转移子代关系 (Lineage Inheritance)
        # 遍历所有谱系关系，如果发现某个子细胞的父节点是 old_id
        # 且该子细胞出现的时间在拆分点 t_start 之后（或等于），则将其父节点改为 new_id
        updated_children = []
        for child_id, parent_id in list(self.lineage_data.items()):
            if parent_id == old_id:
                # 获取子细胞的起始时间
                child_start_frame = self.track_stats.get(child_id, [0, 0])[0]

                # 如果子细胞是在拆分时间点之后出现的，说明它应该属于新的一半
                if child_start_frame >= t_start:
                    self.lineage_data[child_id] = new_id
                    updated_children.append(str(child_id))

        # 4. [核心修复]：调用全量内存刷新，更新质心和统计，解决显示不更新的问题
        self._refresh_cache_from_memory()

        # 5. UI 反馈
        self.update_info_table()
        self.update_tracks_layer()

        msg = f"拆分成功: ID {old_id} -> 新 ID {new_id} (帧 {t_start})"
        if updated_children:
            msg += f" | 已转移子细胞: {', '.join(updated_children)}"
        self.log_message(msg, "success")

    # --- [修改 2]：新增全量内存刷新函数，替代简单的 _recompute_stats_simple ---
    def _refresh_cache_from_memory(self):
        """
        从当前的 labels_layer 内存数据中完全重新计算：
        1. track_stats (Start, End)
        2. centroids_cache (Frame, ID) -> (y, x) 用于绘图
        3. frame_to_ids 用于查询

        解决修改像素后，轨迹和文字标签不更新的问题。
        """
        self.log_message("正在刷新内存缓存...", "info")

        mask_data = self.labels_layer.data
        num_frames = mask_data.shape[0]

        new_stats = {}
        new_cents = {}
        new_f2ids = {t: [] for t in range(num_frames)}

        # 使用 scipy.ndimage 逐帧计算质心，速度尚可
        for t in range(num_frames):
            frame = mask_data[t]
            uids = np.unique(frame)
            uids = uids[uids > 0]  # 排除背景 0

            if len(uids) == 0:
                continue

            new_f2ids[t] = [int(u) for u in uids]

            # 计算该帧所有 ID 的质心
            # center_of_mass 返回 [(y1, x1), (y2, x2), ...]
            # index 参数传入 uids 列表，确保顺序对应
            centers = center_of_mass(frame, frame, uids)

            for idx, uid in enumerate(uids):
                uid = int(uid)
                y, x = centers[idx]

                # 更新质心缓存
                new_cents[(t, uid)] = (y, x)

                # 更新统计 (Start, End)
                if uid not in new_stats:
                    new_stats[uid] = [t, t]
                else:
                    # Start 保持不变 (第一次遇到就是Start)，End 更新为当前 t
                    new_stats[uid][1] = t

        # 更新类成员变量
        self.track_stats = new_stats
        self.centroids_cache = new_cents
        self.frame_to_ids = new_f2ids

        # 顺便清理一下 lineage_data，移除不存在的 ID
        valid_ids = set(new_stats.keys())
        keys_to_remove = [k for k in self.lineage_data if k not in valid_ids]
        for k in keys_to_remove:
            del self.lineage_data[k]

        self.log_message("内存缓存刷新完成", "info")

    # --- [修改 4]：更新删除操作，使用新的刷新函数 ---
    def delete_track_globally(self):
        tid = self.spin_target_del.value()
        if tid == 0:
            return
        self._save_history()

        self.labels_layer.data[self.labels_layer.data == tid] = 0
        self.labels_layer.refresh()

        # 清理 lineage
        if tid in self.lineage_data:
            del self.lineage_data[tid]

        # [重要] 全量刷新
        self._refresh_cache_from_memory()

        self.update_info_table()
        self.update_tracks_layer()
        self.log_message(f"已物理删除 ID {tid}", "warning")

    # --- [修改 5]：更新截断删除操作 ---
    def delete_track_afterwards(self):
        tid, t_curr = self.spin_target_del.value(), self.viewer.dims.current_step[0]
        if tid == 0:
            return
        self._save_history()

        # 将 t_curr 及之后的该 ID 像素置 0
        self.labels_layer.data[t_curr:][self.labels_layer.data[t_curr:] == tid] = 0
        self.labels_layer.refresh()

        # [重要] 全量刷新 (因为 End time 变了，且之后的质心需要移除)
        self._refresh_cache_from_memory()

        self.update_info_table()
        self.update_tracks_layer()
        self.log_message(f"已截断删除 ID {tid} (从帧 {t_curr} 开始)", "warning")

    def link_lineage_batch(self):
        p = self.spin_p.value()
        children = [self.spin_c1.value(), self.spin_c2.value()]

        if p == 0:
            show_info("❌ 父 ID 不能为 0")
            self.log_message("建立谱系失败：未指定父 ID", "error")
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
            msg = f"已建立分裂关系: 父 {p} -> 子 {', '.join(linked_children)}"
            show_info(f"✅ {msg}")
            self.log_message(msg, "success")
            self.update_info_table()
            self.update_tracks_layer()
        else:
            self.log_message(
                "建立谱系失败：未指定有效的子 ID 或子 ID 与父 ID 相同", "warning"
            )

    def _execute_save(self, output_dir):
        """
        深度同步保存逻辑：强制从 lineage_data 字典中提取父子关系
        """
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            self.log_message(f"正在保存数据至: {output_dir} ...", "info")
            show_info("正在执行深度同步保存（含分裂谱系）...")

            mask_data = self.labels_layer.data
            num_frames = mask_data.shape[0]

            # 1. 扫描图像重建统计，确保 ID 是存在的
            new_stats = {}
            for t in range(num_frames):
                frame = mask_data[t]
                # 保存图像 (CTC 标准命名)
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

            # 2. 构造 TXT 内容：L B E P
            tlines = []
            division_count = 0

            for uid in sorted(new_stats.keys()):
                start, end = new_stats[uid]
                parent = int(self.lineage_data.get(uid, 0))

                if parent != 0 and parent not in new_stats:
                    self.log_message(
                        f"警告: 孤儿 ID {uid} (父 {parent} 丢失)，已重置父节点为 0",
                        "warning",
                    )
                    parent = 0

                if parent > 0:
                    division_count += 1

                tlines.append([uid, start, end, parent])

            # 3. 写入 TXT
            df = pd.DataFrame(tlines)
            df.to_csv(output_dir / "man_track.txt", sep=" ", index=False, header=False)

            # 更新 UI 缓存
            self.track_stats = new_stats

            msg = (
                f"✅ 数据同步保存成功！\n"
                f"目录: {output_dir.name}\n"
                f"轨迹总数: {len(tlines)}\n"
                f"分裂事件: {division_count}"
            )

            # 日志输出
            self.log_message(
                f"保存完成! 轨迹数: {len(tlines)}, 分裂数: {division_count}", "success"
            )
            self.log_message(f"文件位置: {output_dir / 'res_track.txt'}", "info")

            QMessageBox.information(self, "保存确认", msg)

        except Exception as e:
            err_msg = f"保存失败: {str(e)}"
            self.log_message(err_msg, "error")
            QMessageBox.critical(self, "错误", err_msg)

    # 修改合并逻辑，确保 ID 彻底从记录中抹除
    def _recompute_stats_simple(self):
        """轻量级重新计算统计信息"""
        new_stats = {}
        f2ids = {t: [] for t in range(len(self.labels_layer.data))}
        for t, frame in enumerate(self.labels_layer.data):
            uids = np.unique(frame)
            uids = uids[uids > 0]
            f2ids[t] = [int(u) for u in uids]
            for u in uids:
                u = int(u)
                if u not in new_stats:
                    new_stats[u] = [t, t]
                else:
                    new_stats[u][1] = t
        self.track_stats = new_stats
        self.frame_to_ids = f2ids

    def save_overwrite(self):
        if not self.data_path:
            self.log_message("未加载数据，无法覆盖保存", "warning")
            return
        self._execute_save(self.data_path.parent / "RES_modified")

    def save_as(self):
        p = QFileDialog.getExistingDirectory(self, "选择保存位置")
        if p:
            self._execute_save(Path(p))
