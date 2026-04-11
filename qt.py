from PIL import Image, ImageDraw, ImageFont
import sys, nncam
# import matplotlib.pyplot as plt # 如不需要可注释
from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import os
import numpy as np
import time
import pyqtgraph as pg
from VS_inference import VirtualStainingModel
from preprocess import SDRNetPipeline
import analysis_core

if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH')


class ROILabel(QLabel):
    roi_finished = pyqtSignal(list)  # 用于 ROI 分析的多边形信号
    input_crop_selected = pyqtSignal(float, float)  # [New] 用于输入裁剪的信号 (norm_x, norm_y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.is_drawing = False
        self.mode = "analysis"  # [New] modes: 'analysis' (polygon) or 'input_crop' (rect)

        # [New] 用于绘制输入裁剪框的预览
        self.crop_center = None  # (x, y) in widget coordinates
        self.crop_rect_size_ratio = (0, 0)  # (w_ratio, h_ratio) relative to display

        self.setMouseTracking(True)
        self.displayed_pixmap = None
        self.offset_x = 0
        self.offset_y = 0

    def set_mode(self, mode):
        """ [New] 切换模式: 'analysis' 或 'input_crop' """
        self.mode = mode
        self.points = []
        self.crop_center = None
        self.update()

    def start_drawing(self):
        self.points = []
        self.is_drawing = True
        self.setCursor(Qt.CrossCursor)
        self.update()

    def reset_drawing(self):
        self.points = []
        self.crop_center = None
        self.update()

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()

        # === 模式 1: 输入裁剪 (Input Crop 924x924) ===
        if self.mode == 'input_crop':
            if event.button() == Qt.LeftButton:
                # 计算相对于图片的归一化坐标
                img_x = x - self.offset_x
                img_y = y - self.offset_y

                if self.displayed_pixmap:
                    w = self.displayed_pixmap.width()
                    h = self.displayed_pixmap.height()
                    if 0 <= img_x <= w and 0 <= img_y <= h:
                        norm_x = img_x / w
                        norm_y = img_y / h
                        self.input_crop_selected.emit(norm_x, norm_y)

                        # 更新UI显示点击位置
                        self.crop_center = (x, y)
                        self.update()
            return

        # === 模式 2: 结果分析 (ROI Polygon) ===
        if not self.is_drawing:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            self.points.append(QPoint(x, y))
            self.update()
        elif event.button() == Qt.RightButton:
            if len(self.points) > 2:
                norm_points = []
                for p in self.points:
                    img_x = p.x() - self.offset_x
                    img_y = p.y() - self.offset_y
                    if self.displayed_pixmap:
                        nx = max(0.0, min(1.0, img_x / self.displayed_pixmap.width()))
                        ny = max(0.0, min(1.0, img_y / self.displayed_pixmap.height()))
                        norm_points.append((nx, ny))
                self.is_drawing = False
                self.setCursor(Qt.ArrowCursor)
                self.roi_finished.emit(norm_points)
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制 ROI 多边形 (Analysis)
        if self.points:
            pen = QPen(Qt.green, 2)
            painter.setPen(pen)
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    painter.drawLine(self.points[i], self.points[i + 1])
                if not self.is_drawing:
                    painter.drawLine(self.points[-1], self.points[0])
            painter.setBrush(Qt.red)
            for p in self.points:
                painter.drawEllipse(p, 3, 3)

        # [New] 绘制输入裁剪框 (Input Crop)
        if self.mode == 'input_crop' and self.crop_center and self.displayed_pixmap:
            pen = QPen(Qt.yellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            # 计算屏幕上的矩形大小
            cx, cy = self.crop_center

            # 这里需要主程序传入的比例，或者简单的视觉反馈
            # 为了准确显示 924x924 在屏幕上的大小，需要 MainWidget 计算后传回，
            # 这里简化处理：画一个十字准星表示中心，具体的框由 MainWidget 的逻辑决定
            painter.drawLine(cx - 10, cy, cx + 10, cy)
            painter.drawLine(cx, cy - 10, cx, cy + 10)

            # 如果有了比例 (self.crop_rect_size_ratio)，可以画出框
            if self.crop_rect_size_ratio[0] > 0:
                disp_w = self.displayed_pixmap.width()
                disp_h = self.displayed_pixmap.height()

                rect_w = int(disp_w * self.crop_rect_size_ratio[0])
                rect_h = int(disp_h * self.crop_rect_size_ratio[1])

                rx = int(cx - rect_w / 2)
                ry = int(cy - rect_h / 2)
                painter.drawRect(rx, ry, rect_w, rect_h)

    # ... (setPixmap, resizeEvent, update_offsets 保持不变) ...
    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self.displayed_pixmap = pixmap
        self.update_offsets()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_offsets()

    def update_offsets(self):
        if self.displayed_pixmap:
            self.offset_x = (self.width() - self.displayed_pixmap.width()) / 2
            self.offset_y = (self.height() - self.displayed_pixmap.height()) / 2
        else:
            self.offset_x = 0
            self.offset_y = 0
# ==========================================
# 2. 结果弹窗类 (用于展示合成大图)
# ==========================================
class ResultWindow(QMainWindow):
    def __init__(self, image_np, title="Virtual Staining Result"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1000, 800)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        scroll_area.setWidget(self.lbl_image)

        self.display_image(image_np)



    def display_image(self, img_np):
        if len(img_np.shape) == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        h, w, c = img_rgb.shape
        qImg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qImg))


# ==========================================
# 3. 分析结果展示对话框
# ==========================================
class FeatureDialog(QDialog):
    def __init__(self, feature_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extracted Features")
        self.resize(400, 600)

        layout = QVBoxLayout()
        lbl_title = QLabel("Cellular Features")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(lbl_title)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Feature", "Value"])
        table.setRowCount(len(feature_dict))

        for i, (key, val) in enumerate(feature_dict.items()):
            table.setItem(i, 0, QTableWidgetItem(str(key)))
            table.setItem(i, 1, QTableWidgetItem(f"{val:.6f}"))

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(table)

        btn_next = QPushButton("Proceed to Prediction")
        btn_next.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_next.clicked.connect(self.accept)  # 点击后关闭窗口，返回 Accepted
        layout.addWidget(btn_next)

        self.setLayout(layout)


class FateResultDialog(QDialog):
    def __init__(self, stage_label, probs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fate Inference Result")
        self.resize(400, 300)

        layout = QVBoxLayout()

        # 结果显示
        lbl_title = QLabel("Prediction Result:")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_stage = QLabel(stage_label)
        lbl_stage.setAlignment(Qt.AlignCenter)
        # 根据不同阶段设置不同颜色
        color = "#2196F3"  # Blue
        if "Stage III" in stage_label:
            color = "#F44336"  # Red
        elif "Stage II" in stage_label:
            color = "#FF9800"  # Orange

        lbl_stage.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color}; margin: 20px;")
        layout.addWidget(lbl_stage)

        # 概率显示
        prob_group = QGroupBox("Probabilities")
        v_prob = QVBoxLayout()
        v_prob.addWidget(QLabel(f"Stage I (0):   {probs[0]:.4f}"))
        v_prob.addWidget(QLabel(f"Stage II (1):  {probs[1]:.4f}"))
        v_prob.addWidget(QLabel(f"Stage III (2): {probs[2]:.4f}"))
        prob_group.setLayout(v_prob)
        layout.addWidget(prob_group)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self.setLayout(layout)
# ==========================================
# 4. 后台工作线程 (核心流水线逻辑)
# ==========================================
class AIWorker(QThread):
    result_ready = pyqtSignal(np.ndarray, str, dict)

    def __init__(self, model_map, base_dir):
        super().__init__()
        self.model_map = model_map
        self.base_dir = base_dir

        self.inference_engine = None
        self.preprocess_engine = SDRNetPipeline()

        self.pending_image = None
        self.pending_model_name = None  # [新增] 用于区分单模型
        self.enable_preprocess = False
        self.is_running = True

        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()

    def run(self):
        print("Thread: Initializing Engines...")
        from VS_inference import VirtualStainingModel
        self.inference_engine = VirtualStainingModel()

        while self.is_running:
            self.mutex.lock()
            if self.pending_image is None:
                self.wait_condition.wait(self.mutex)

            # 获取数据
            img_data = self.pending_image
            model_name = self.pending_model_name
            do_preprocess = self.enable_preprocess

            # 重置状态
            self.pending_image = None
            self.pending_model_name = None
            self.mutex.unlock()

            if img_data is not None:
                try:
                    # === Step 1: 通用预处理 ===
                    current_img = img_data
                    if do_preprocess:
                        self.result_ready.emit(np.zeros((1, 1, 3), dtype=np.uint8), "preprocessing_start", {})
                        current_img = self.preprocess_engine.process(current_img)
                        self.result_ready.emit(current_img, "preprocess_result", {})

                    # === 分支判断 ===
                    if model_name is None:
                        # >>> 分支 A: 全自动流水线 (Nuclei -> Actin -> Yap) <<<
                        self._run_pipeline(current_img)
                    else:
                        # >>> 分支 B: 单模型运行 <<<
                        self._run_single_model(current_img, model_name)

                except Exception as e:
                    print(f"Error in AIWorker: {e}")
                    import traceback
                    traceback.print_exc()

    def _run_single_model(self, img, model_folder_name):
        """辅助函数：运行单个模型"""
        # 1. 通知 UI
        self.result_ready.emit(np.zeros((1, 1, 3), dtype=np.uint8), f"staining_single", {})

        # 2. 加载模型
        full_path = os.path.join(self.base_dir, "WSCON", "checkpoints", model_folder_name)
        print(f"Loading single model from: {full_path}")
        self.inference_engine.load_network(full_path)

        # 3. 推理
        pred_rgb = self.inference_engine.predict(img)

        # 4. 发送结果 (状态码 "single_result")
        self.result_ready.emit(pred_rgb, "single_result", {"model": model_folder_name})

    def _run_pipeline(self, img):
        """辅助函数：运行全自动流水线"""
        pipeline = [
            ("Nuclei", "Blue"),
            ("F-actin", "Green"),
            ("Yap", "Magenta")
        ]

        stain_results = {}
        save_dir = "./staining_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for model_key, color_name in pipeline:
            self.result_ready.emit(np.zeros((1, 1, 3), dtype=np.uint8), f"staining_{model_key}", {})

            folder_name = self.model_map[model_key]
            full_path = os.path.join(self.base_dir, "WSCON", "checkpoints", folder_name)
            self.inference_engine.load_network(full_path)

            pred_rgb = self.inference_engine.predict(img)
            # 转灰度用于合成
            if len(pred_rgb.shape) == 3:
                pred_gray = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2GRAY)
            else:
                pred_gray = pred_rgb

            stain_results[model_key] = pred_gray
            cv2.imwrite(f"{save_dir}/{model_key}.tif", pred_gray)

        # 合成 RGB (BGR: Nuclei, Actin, Yap)
        composite_img = cv2.merge([
            stain_results["Nuclei"],
            stain_results["F-actin"],
            stain_results["Yap"]
        ])
        cv2.imwrite(f"{save_dir}/Composite_RGB.tif", composite_img)

        self.result_ready.emit(composite_img, "final_result", {})

    def process_pipeline(self, image_data, enable_preprocess=False):
        self.mutex.lock()
        self.pending_image = image_data
        self.pending_model_name = None  # None 表示流水线模式
        self.enable_preprocess = enable_preprocess
        self.mutex.unlock()
        self.wait_condition.wakeAll()

    def process_image(self, image_data, model_name, enable_preprocess=False):
        self.mutex.lock()
        self.pending_image = image_data
        self.pending_model_name = model_name  # 有值 表示单模型模式
        self.enable_preprocess = enable_preprocess
        self.mutex.unlock()
        self.wait_condition.wakeAll()

    def stop(self):
        self.is_running = False
        self.wait_condition.wakeAll()
        self.wait()


class MainWidget(QWidget):
    evtCallback = pyqtSignal(int)

    @staticmethod

    def makeLayout(lbl1, sli1, val1, lbl2, sli2, val2):
        hlyt1 = QHBoxLayout()
        hlyt1.addWidget(lbl1)
        hlyt1.addStretch()
        hlyt1.addWidget(val1)
        hlyt2 = QHBoxLayout()
        hlyt2.addWidget(lbl2)
        hlyt2.addStretch()
        hlyt2.addWidget(val2)
        vlyt = QVBoxLayout()
        vlyt.addLayout(hlyt1)
        vlyt.addWidget(sli1)
        vlyt.addLayout(hlyt2)
        vlyt.addWidget(sli2)
        return vlyt

    def initUI(self):
        self.setGeometry(500, 1000, 1200, 900)
        self.setWindowTitle('Label-free computational microscope')

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1024, 768)
        self.hcam = None
        self.is_local_mode = False
        self.loaded_image = None
        self.current_stained_img = None
        self.current_roi_norm_points = None # 存储ROI坐标
        self.result_popup = None

        self.input_crop_rect = None  # (x, y, w, h)
        self.raw_input_image = None  # 始终保存原始的高分辨率图 (2448x2048)
        self.cropped_input_img = None  # 保存裁剪后的图 (924x924)
        self.is_cropped_view = False  # 标记当前界面显示的是全图还是裁剪图
        # 初始化预测器

        model_dir = './saved_fate_model'
        self.predictor = analysis_core.FatePredictor(
            model_path=os.path.join(model_dir, 'flexiblemlp_2000epochs.pth'),
            scaler_path=os.path.join(model_dir, 'scaler_params.pth')
        )

        self.timer = QTimer(self)
        self.snaptimer = QTimer(self)
        self.imgWidth = 0
        self.imgHeight = 0
        self.pData = None
        self.res = 0
        self.temp = nncam.NNCAM_TEMP_DEF
        self.tint = nncam.NNCAM_TINT_DEF
        self.count = 0
        self.interval = 0
        self.initUI()


        # 模型映射
        self.model_map = {
            "F-actin": "WF_actin",
            "Nuclei": "WF_dapi",
            "Yap": "WF_yap"
        }

        gbox_model = QGroupBox("Target Structure")
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Auto Pipeline (Nuclei->F-actin->Yap)"])
        self.cmb_model.addItems(list(self.model_map.keys()))


        vlyt_model = QVBoxLayout()
        vlyt_model.addWidget(self.cmb_model)
        gbox_model.setLayout(vlyt_model)

        gboxres = QGroupBox("Resolution")
        self.cmb_res = QComboBox()
        self.cmb_res.setEnabled(False)
        vlytres = QVBoxLayout()
        vlytres.addWidget(self.cmb_res)
        gboxres.setLayout(vlytres)
        self.cmb_res.currentIndexChanged.connect(self.onResolutionChanged)

        gboxexp = QGroupBox("Exposure")
        self.cbox_auto = QCheckBox("Auto exposure")
        self.cbox_auto.setEnabled(False)
        self.lbl_expoTime = QLabel("0")
        self.lbl_expoGain = QLabel("0")
        self.slider_expoTime = QSlider(Qt.Horizontal)
        self.slider_expoGain = QSlider(Qt.Horizontal)
        self.slider_expoTime.setEnabled(False)
        self.slider_expoGain.setEnabled(False)
        self.cbox_auto.stateChanged.connect(self.onAutoExpo)
        self.slider_expoTime.valueChanged.connect(self.onExpoTime)
        self.slider_expoGain.valueChanged.connect(self.onExpoGain)
        vlytexp = QVBoxLayout()
        vlytexp.addWidget(self.cbox_auto)
        vlytexp.addLayout(
            self.makeLayout(QLabel("Time(us):"), self.slider_expoTime, self.lbl_expoTime, QLabel("Gain(%):"),
                            self.slider_expoGain, self.lbl_expoGain))
        gboxexp.setLayout(vlytexp)

        gboxwb = QGroupBox("White balance")
        self.btn_autoWB = QPushButton("White balance")
        self.btn_autoWB.setEnabled(False)
        self.btn_autoWB.clicked.connect(self.onAutoWB)
        self.lbl_temp = QLabel(str(nncam.NNCAM_TEMP_DEF))
        self.lbl_tint = QLabel(str(nncam.NNCAM_TINT_DEF))
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(nncam.NNCAM_TEMP_MIN, nncam.NNCAM_TEMP_MAX)
        self.slider_temp.setValue(nncam.NNCAM_TEMP_DEF)
        self.slider_tint.setRange(nncam.NNCAM_TINT_MIN, nncam.NNCAM_TINT_MAX)
        self.slider_tint.setValue(nncam.NNCAM_TINT_DEF)
        self.slider_temp.setEnabled(False)
        self.slider_tint.setEnabled(False)
        self.slider_temp.valueChanged.connect(self.onWBTemp)
        self.slider_tint.valueChanged.connect(self.onWBTint)
        vlytwb = QVBoxLayout()
        vlytwb.addLayout(
            self.makeLayout(QLabel("Temperature:"), self.slider_temp, self.lbl_temp, QLabel("Tint:"), self.slider_tint,
                            self.lbl_tint))
        vlytwb.addWidget(self.btn_autoWB)
        gboxwb.setLayout(vlytwb)

        self.btn_open = QPushButton("Open Camara")
        self.btn_open.clicked.connect(self.onBtnOpen)
        self.btn_load_file = QPushButton("Load Image File")
        self.btn_load_file.clicked.connect(self.onBtnLoadFile)


        self.btn_snap = QPushButton("Snap")
        self.btn_snap.setEnabled(False)
        self.btn_snap.clicked.connect(self.onBtnSnap)
        self.btn_snapping = QPushButton("Video Capture")
        self.btn_snapping.setEnabled(False)
        self.btn_snapping.clicked.connect(self.beginsnapTimer)
        self.snaptimer.timeout.connect(self.snapTimer)
        self.btn_stopping = QPushButton("Stop Capture")
        self.btn_stopping.setEnabled(False)
        self.btn_stopping.clicked.connect(self.snapStop)

        self.btn_input_crop = QPushButton("Select Input Crop (924x924)")
        self.btn_input_crop.setToolTip("Crop a 924x924 area for AI input")
        self.btn_input_crop.clicked.connect(self.onBtnInputCrop)
        self.btn_input_crop.setEnabled(False)  # 加载图片后才启用

        self.btn_VS = QPushButton("2. Virtual Staining Pipeline")
        self.btn_VS.setEnabled(False)
        self.btn_VS.clicked.connect(self.onBtnVS)

        self.chk_preprocess = QCheckBox("Enable Preprocess: SDRNet")
        self.chk_preprocess.setChecked(False)
        self.chk_preprocess.setToolTip("Cellular image enhancement before staining")

        self.btn_VS = QPushButton("VFS-WSCON")
        self.btn_VS.setEnabled(False)
        self.btn_VS.clicked.connect(self.onBtnVS)

        # 分析控制
        self.gbox_crop = QGroupBox("Analysis & Fate")

        # 按钮1: 选择 ROI (开始绘制)
        self.btn_select_roi = QPushButton("Select ROI")
        self.btn_select_roi.setEnabled(False)
        self.btn_select_roi.clicked.connect(self.onBtnSelectROI)

        self.btn_reset_roi = QPushButton("Reset ROI")
        self.btn_reset_roi.setEnabled(False)
        self.btn_reset_roi.clicked.connect(self.onBtnResetROI)

        # 按钮3: Fate Identification (点击后才开始分析)
        self.btn_fate_id = QPushButton("Fate Inference")
        self.btn_fate_id.setEnabled(False)  # 默认禁用，等ROI选好后启用
        self.btn_fate_id.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_fate_id.clicked.connect(self.onBtnFateIdentification)

        vlyt_crop = QHBoxLayout()
        vlyt_crop.addWidget(self.btn_select_roi)
        vlyt_crop.addWidget(self.btn_reset_roi)
        vlyt_crop.addWidget(self.btn_fate_id)
        self.gbox_crop.setLayout(vlyt_crop)

        vlytctrl = QVBoxLayout()
        vlytctrl.addWidget(gboxres)
        vlytctrl.addWidget(gboxexp)
        vlytctrl.addWidget(gboxwb)
        vlytctrl.addWidget(gbox_model)
        vlytctrl.addWidget(self.btn_open)
        vlytctrl.addWidget(self.btn_load_file)
        vlytctrl.addWidget(self.btn_snap)
        vlytctrl.addWidget(self.btn_snapping)
        vlytctrl.addWidget(self.btn_stopping)
        vlytctrl.addWidget(self.chk_preprocess)
        vlytctrl.addWidget(self.btn_input_crop)
        vlytctrl.addWidget(self.btn_VS)
        vlytctrl.addWidget(self.gbox_crop)
        vlytctrl.addStretch()
        wgctrl = QWidget()
        wgctrl.setLayout(vlytctrl)

        self.lbl_frame = QLabel()
        self.lbl_video = ROILabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.roi_finished.connect(self.onROIFinished)
        self.lbl_video.input_crop_selected.connect(self.onInputCropSelected)

        vlytshow = QVBoxLayout()
        vlytshow.addWidget(self.lbl_video, 1)
        vlytshow.addWidget(self.lbl_frame)
        wgshow = QWidget()
        wgshow.setLayout(vlytshow)

        gmain = QGridLayout()
        gmain.setColumnStretch(0, 1)
        gmain.setColumnStretch(1, 4)
        gmain.addWidget(wgctrl)
        gmain.addWidget(wgshow)
        self.setLayout(gmain)

        self.timer.timeout.connect(self.onTimer)
        self.evtCallback.connect(self.onevtCallback)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ai_worker = AIWorker(self.model_map, base_dir)
        self.ai_worker.result_ready.connect(self.onVSResult)
        self.ai_worker.start()


    def onTimer(self):
        if self.hcam:
            nFrame, nTime, nTotalFrame = self.hcam.get_FrameRate()
            self.lbl_frame.setText("frame = {}, fps = {:.1f}".format(nTotalFrame, nFrame * 1000.0 / nTime))

    def dialogsetting(self):
        self.interval, ok = QInputDialog.getInt(self, 'Frequency', '拍摄的每张照片间隔时长:(s)')

    def beginsnapTimer(self):
        self.dialogsetting()
        self.snaptimer.start(self.interval*1000)
        if self.pData is not None:
            image = QImage(self.pData, self.imgWidth, self.imgHeight, QImage.Format_RGB888)
            image.save("./datasets/{}.jpg".format(self.count))

    def snapTimer(self):
        self.btn_snapping.setEnabled(False)
        self.btn_stopping.setEnabled(True)
        self.snap()

    def snap(self):
        if self.pData is not None:
            self.count += 1
            image = QImage(self.pData, self.imgWidth, self.imgHeight, QImage.Format_RGB888)
            image.save("./datasets/{}.jpg".format(self.count))

    def snapStop(self):
        self.btn_snapping.setText("Video Capture")
        self.snaptimer.stop()
        self.btn_snapping.setEnabled(True)
        self.btn_stopping.setEnabled(False)

    def closeCamera(self):
        if self.hcam:
            self.hcam.Close()
        self.hcam = None
        self.pData = None

        self.btn_open.setText("Open")
        self.timer.stop()
        self.lbl_frame.clear()
        self.cbox_auto.setEnabled(False)
        self.slider_expoGain.setEnabled(False)
        self.slider_expoTime.setEnabled(False)
        self.btn_autoWB.setEnabled(False)
        self.slider_temp.setEnabled(False)
        self.slider_tint.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.btn_snapping.setEnabled(False)
        self.btn_VS.setEnabled(False)
        self.cmb_res.setEnabled(False)
        self.cmb_res.clear()

    def closeEvent(self, event):
        self.closeCamera()
        self.snapStop()

    def onResolutionChanged(self, index):
        if self.hcam: #step 1: stop camera
            self.hcam.Stop()

        self.res = index
        self.imgWidth = self.cur.model.res[index].width
        self.imgHeight = self.cur.model.res[index].height

        if self.hcam: #step 2: restart camera
            self.hcam.put_eSize(self.res)
            self.startCamera()

    def onAutoExpo(self, state):
        if self.hcam:
            self.hcam.put_AutoExpoEnable(1 if state else 0)
            self.slider_expoTime.setEnabled(not state)
            self.slider_expoGain.setEnabled(not state)

    def onExpoTime(self, value):
        if self.hcam:
            self.lbl_expoTime.setText(str(value))
            if not self.cbox_auto.isChecked():
                self.hcam.put_ExpoTime(value)

    def onExpoGain(self, value):
        if self.hcam:
            self.lbl_expoGain.setText(str(value))
            if not self.cbox_auto.isChecked():
                self.hcam.put_ExpoAGain(value)

    def onAutoWB(self):
        if self.hcam:
            self.hcam.AwbOnce()

    def wbCallback(nTemp, nTint, self):
        self.slider_temp.setValue(nTemp)
        self.slider_tint.setValue(nTint)

    def onWBTemp(self, value):
        if self.hcam:
            self.temp = value
            self.hcam.put_TempTint(self.temp, self.tint)
            self.lbl_temp.setText(str(value))

    def onWBTint(self, value):
        if self.hcam:
            self.tint = value
            self.hcam.put_TempTint(self.temp, self.tint)
            self.lbl_tint.setText(str(value))

    def startCamera(self):
        self.is_local_mode = False
        self.pData = bytes(nncam.TDIBWIDTHBYTES(self.imgWidth * 24) * self.imgHeight)
        uimin, uimax, uidef = self.hcam.get_ExpTimeRange()
        self.slider_expoTime.setRange(uimin, uimax)
        self.slider_expoTime.setValue(uidef)
        usmin, usmax, usdef = self.hcam.get_ExpoAGainRange()
        self.slider_expoGain.setRange(usmin, usmax)
        self.slider_expoGain.setValue(usdef)
        self.handleExpoEvent()
        if self.cur.model.flag & nncam.NNCAM_FLAG_MONO == 0:
            self.handleTempTintEvent()
        try:
            self.hcam.StartPullModeWithCallback(self.eventCallBack, self)
        except nncam.HRESULTException:
            self.closeCamera()
            QMessageBox.warning(self, "Warning", "Failed to start camera.")
        else:
            self.cmb_res.setEnabled(True)
            self.cbox_auto.setEnabled(True)
            self.btn_autoWB.setEnabled(self.cur.model.flag & nncam.NNCAM_FLAG_MONO == 0)
            self.slider_temp.setEnabled(self.cur.model.flag & nncam.NNCAM_FLAG_MONO == 0)
            self.slider_tint.setEnabled(self.cur.model.flag & nncam.NNCAM_FLAG_MONO == 0)
            self.btn_open.setText("Close")
            self.btn_snap.setEnabled(True)
            self.btn_snapping.setEnabled(True)
            self.btn_VS.setEnabled(True)
            bAuto = self.hcam.get_AutoExpoEnable()
            self.cbox_auto.setChecked(1 == bAuto)
            self.timer.start(1000)

    def openCamera(self):
        self.hcam = nncam.Nncam.Open(self.cur.id)
        if self.hcam:
            self.res = self.hcam.get_eSize()
            self.imgWidth = self.cur.model.res[self.res].width
            self.imgHeight = self.cur.model.res[self.res].height
            with QSignalBlocker(self.cmb_res):
                self.cmb_res.clear()
                for i in range(0, self.cur.model.preview):
                    self.cmb_res.addItem("{}*{}".format(self.cur.model.res[i].width, self.cur.model.res[i].height))
                self.cmb_res.setCurrentIndex(self.res)
                self.cmb_res.setEnabled(True)
            self.hcam.put_Option(nncam.NNCAM_OPTION_BYTEORDER, 0) #Qimage use RGB byte order
            self.hcam.put_AutoExpoEnable(1)
            self.startCamera()

    def onBtnOpen(self):
        if self.hcam:
            self.closeCamera()
        else:
            arr = nncam.Nncam.EnumV2()
            if 0 == len(arr):
                QMessageBox.warning(self, "Warning", "No camera found.")
            elif 1 == len(arr):
                self.cur = arr[0]
                self.openCamera()
            else:
                menu = QMenu()
                for i in range(0, len(arr)):
                    action = QAction(arr[i].displayname, self)
                    action.setData(i)
                    menu.addAction(action)
                action = menu.exec(self.mapToGlobal(self.btn_open.pos()))
                if action:
                    self.cur = arr[action.data()]
                    self.openCamera()

    def onBtnSnap(self):
        menu = QMenu()
        if self.hcam:
            if 0 == self.cur.model.still:  # not support still image capture
                if self.pData is not None:
                    image = QImage(self.pData, self.imgWidth, self.imgHeight, QImage.Format_RGB888)
                    self.count += 1
                    image.save("./datasets/pyqt{}.jpg".format(self.count))

            else:
                for i in range(0, self.cur.model.still):
                    print(i)
                    action = QAction("2048*2048", self)
                    action.setData(i)
                    menu.addAction(action)
            action = menu.exec(self.mapToGlobal(self.btn_snap.pos()))
            print(action.data)
            self.hcam.Snap(action.data())

    def onBtnLoadFile(self):
        if self.hcam: self.closeCamera()
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Images (*.jpg *.png *.bmp *.tif)")
        if fname:
            self.loaded_image = QImage(fname)
            if self.loaded_image.isNull(): return

            self.is_local_mode = True
            # [New] 保存原始数据供裁剪使用
            self.raw_input_image = self.qimage_to_cv2(self.loaded_image)
            self.imgWidth = self.loaded_image.width()
            self.imgHeight = self.loaded_image.height()

            # 显示原图
            self.display_image(self.raw_input_image)

            self.btn_input_crop.setEnabled(True)  # 允许裁剪
            self.btn_VS.setEnabled(False)  # 强制用户先裁剪再染色 (或者你可以允许全图)
            self.lbl_frame.setText(
                f"Loaded: {os.path.basename(fname)} ({self.imgWidth}x{self.imgHeight}). Please select 924x924 crop.")

    def onBtnInputCrop(self):
        if self.raw_input_image is None: return

        # 1. 恢复显示全分辨率原图，以便用户选择
        self.display_image(self.raw_input_image)
        self.is_cropped_view = False

        # 2. 切换 ROILabel 到裁剪选择模式
        self.lbl_video.set_mode('input_crop')

        # 3. 计算 924x924 在全图上的比例，用于绘制黄色预览框
        h, w = self.raw_input_image.shape[:2]
        self.lbl_video.crop_rect_size_ratio = (924.0 / w, 924.0 / h)

        self.lbl_frame.setText("Full View Mode: Click on image to confirm 924x924 crop area.")

        # 禁用后续步骤，强迫用户重新走流程
        self.btn_VS.setEnabled(False)

    def onInputCropSelected(self, norm_x, norm_y):
        if self.raw_input_image is None: return

        h, w = self.raw_input_image.shape[:2]
        crop_size = 924

        # 1. 计算中心点和左上角坐标
        cx = int(norm_x * w)
        cy = int(norm_y * h)
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2

        # 2. 边界限制 (防止超出图片范围)
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x1 + crop_size > w: x1 = w - crop_size
        if y1 + crop_size > h: y1 = h - crop_size

        # 3. 保存裁剪参数
        self.input_crop_rect = (x1, y1, crop_size, crop_size)

        # 4. [核心] 执行裁剪，并保存到 self.cropped_input_img
        self.cropped_input_img = self.raw_input_image[y1:y1 + crop_size, x1:x1 + crop_size].copy()

        # 5. [核心] 立即更新主界面显示 -> 变成“特写视图”
        self.display_image(self.cropped_input_img)
        self.is_cropped_view = True

        # 6. 退出裁剪模式，停止画黄色框
        self.lbl_video.set_mode('none')

        self.lbl_frame.setText(f"Crop Confirmed ({x1},{y1}). Displaying Cropped View. Ready for Pipeline.")
        self.btn_VS.setEnabled(True)

    def onBtnVS(self):
        # 确定输入源：优先使用刚才确认的裁剪图
        target_input = None

        if self.is_cropped_view and self.cropped_input_img is not None:
            # 使用刚才裁剪好的图
            target_input = self.cropped_input_img
            print("Sending cropped view to AI Pipeline...")

        elif self.hcam and self.pData:
            # 相机模式直接取流
            image = QImage(self.pData, self.imgWidth, self.imgHeight, QImage.Format_RGB888)
            target_input = self.qimage_to_cv2(image)

        elif self.raw_input_image is not None:
            # 兜底：如果没有裁剪，就用全图（或者你可以禁止这种情况）
            target_input = self.raw_input_image

        if target_input is not None:
            selected_text = self.cmb_model.currentText()
            enable_preprocess = self.chk_preprocess.isChecked()

            # 锁定按钮
            self.btn_VS.setEnabled(False)
            self.btn_input_crop.setEnabled(False)
            self.btn_select_roi.setEnabled(False)
            self.btn_fate_id.setEnabled(False)

            if selected_text == 'Auto Pipeline (Nuclei->Actin->Yap)':
                if enable_preprocess:
                    self.btn_VS.setText(f"SDRNet + {selected_text}...")
                else:
                    self.btn_VS.setText(f"Loading {selected_text}...")
                self.ai_worker.process_pipeline(target_input, enable_preprocess)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_folder_name = self.model_map[selected_text]

                self.ai_worker.process_image(target_input, model_folder_name, enable_preprocess)


        else:
            QMessageBox.warning(self, "Warning", "No image source!")

    def onVSResult(self, result_np, status, info_dict):
        # 预处理状态
        if status == "preprocessing_start":
            self.btn_VS.setText("SDRNet...")
        elif status == "preprocess_result":
            # 如果是预处理中间结果，可以选择显示或不显示，这里暂不处理，等待最终结果
            pass

        # 染色中状态
        elif "staining_" in status:
            if status == "staining_single":
                self.btn_VS.setText("Running Single Model...")
            else:
                model = status.split("_")[1]
                self.btn_VS.setText(f"Staining {status.split('_')[1]}...")

        # === 情况 A: 全自动流水线完成 ===
        elif status == "final_result":
            self.btn_VS.setEnabled(True)
            self.btn_VS.setText("VFS-WSCON")

            # 1. 【关键】保存纯净的合成图数据 (用于后续 ROI 分析)
            # 千万不要把字写在 self.current_stained_img 上，否则分析时会把文字也算进去
            self.current_stained_img = result_np

            # 2. 创建一个副本用于弹窗显示 (Visualization Copy)
            vis_img = result_np.copy()

            # 3. 在副本上绘制通道说明 (文字颜色对应通道颜色 BGR)
            # 参数: 图片, 文本, 坐标, 字体, 大小, 颜色, 粗细
            # Blue: Nuclei
            cv2.putText(vis_img, "Nuclei", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 3)
            # Green: F-actin
            cv2.putText(vis_img, "F-actin", (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)
            # Red: YAP
            cv2.putText(vis_img, "YAP", (30, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            # 4. 弹出带标签的图像窗口
            self.result_popup = ResultWindow(vis_img, title="Merged VFS (RGB)")
            self.result_popup.show()

            # 5. 界面状态更新 (注意：主界面不调用 display_image，保持原图)
            self.lbl_video.set_mode('analysis')
            self.btn_select_roi.setEnabled(True)
            self.btn_reset_roi.setEnabled(True)
            self.lbl_frame.setText("Staining Done. Please Select ROI on original image.")

            if self.input_crop_rect and self.raw_input_image is not None:
                x, y, w, h = self.input_crop_rect
                cropped_input = self.raw_input_image[y:y + h, x:x + w]
                self.display_image(cropped_input)

        # === 情况 B: 单模型完成 [新增] ===
        elif status == "single_result":
            self.btn_VS.setEnabled(True)
            self.btn_VS.setText("VFS-WSCON")

            # 1. 弹窗显示单模型结果
            model_name = info_dict.get("model", "Single Model")
            self.result_popup = ResultWindow(result_np, title=f"Result: {model_name}")
            self.result_popup.show()

            # 2. 禁用分析功能 (因为缺其他通道，无法进行 Fate Identification)
            self.btn_select_roi.setEnabled(False)
            self.btn_reset_roi.setEnabled(False)
            self.btn_fate_id.setEnabled(False)
            self.current_stained_img = None  # 清空暂存，防止误用旧数据分析

            self.lbl_frame.setText(f"Single model {model_name} finished. (Analysis disabled for single channel)")

    def onBtnSelectROI(self):
        # 开始在主界面（原图）上绘制
        if self.current_stained_img is None: return
        self.lbl_video.set_mode('analysis')  # 确保切换回多边形模式
        self.lbl_video.start_drawing()
        self.btn_select_roi.setEnabled(False)
        self.lbl_frame.setText("Drawing Analysis ROI... Right Click to finish.")
    def onBtnResetROI(self):
        self.lbl_video.reset_drawing()
        self.btn_select_roi.setEnabled(True)
        self.btn_fate_id.setEnabled(False)  # 重置后禁用分析
        self.current_roi_norm_points = None
        self.lbl_frame.setText("ROI Reset.")

    def onROIFinished(self, norm_points):
        # ROI 绘制结束，只保存坐标，不自动分析
        self.btn_select_roi.setEnabled(True)
        self.current_roi_norm_points = norm_points
        self.btn_fate_id.setEnabled(True)  # 激活 Fate Identification 按钮
        self.lbl_frame.setText("ROI Selected. Click 'Fate Identification' to analyze.")

    def onBtnFateIdentification(self):
        if self.current_roi_norm_points is None or self.current_stained_img is None:
            return

        self.lbl_frame.setText("Analyzing Fate...")
        QApplication.processEvents()

        try:
            # 1. 坐标映射 (Normalized -> Real on Stained Image)
            h, w = self.current_stained_img.shape[:2]
            real_points = []
            for (nx, ny) in self.current_roi_norm_points:
                real_points.append([int(nx * w), int(ny * h)])

            if len(real_points) < 3: return

            # 2. 裁剪与通道分离
            roi_composite = self.crop_roi_img_logic(self.current_stained_img, real_points)
            roi_yap, roi_actin, roi_nuclei = cv2.split(roi_composite)
            _, whole_actin,whole_nuclei = cv2.split(self.current_stained_img)

            # 3. 特征提取
            all_features = {}
            all_features.update(analysis_core.analyze_actin(roi_actin))
            all_features.update(analysis_core.analyze_yap(roi_yap, roi_nuclei, roi_actin))
            all_features.update(analysis_core.analyze_mass(whole_nuclei, whole_actin))

            # 4. 预测
            pred_label_idx, probs = self.predictor.predict(all_features)  # 返回 '0','1','2'

            # 5. 结果映射
            stage_map = {"0": "Stage I", "1": "Stage II", "2": "Stage III"}
            final_stage = stage_map.get(str(pred_label_idx), "Unknown")

            # 6. 弹窗逻辑：先弹 Feature -> 确认后 -> 弹 Result
            self.lbl_frame.setText("Analysis Done.")

            # 弹窗 1: Feature
            feat_dlg = FeatureDialog(all_features, self)
            if feat_dlg.exec_() == QDialog.Accepted:
                # 弹窗 2: Fate Result
                res_dlg = FateResultDialog(final_stage, probs, self)
                res_dlg.exec_()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Analysis Error", str(e))

    def crop_roi_img_logic(self, img, points):
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        if len(img.shape) == 3:
            crop_img = cv2.bitwise_and(img, img, mask=mask)
        else:
            crop_img = cv2.bitwise_and(img, mask)
        x, y, w, h = cv2.boundingRect(pts)
        return crop_img[y:y + h, x:x + w]

    def display_image(self, img_np):
        if img_np is None: return
        if len(img_np.shape) == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        qImg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.lbl_video.setPixmap(
            QPixmap.fromImage(qImg).scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def qimage_to_cv2(self, qimg):
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        width = qimg.width();
        height = qimg.height();
        ptr = qimg.bits();
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ---------------- 相机回调 (保持不变) ----------------
    @staticmethod
    def eventCallBack(nEvent, self):
        self.evtCallback.emit(nEvent)

    def onevtCallback(self, nEvent):
        if self.hcam:
            if nEvent == nncam.NNCAM_EVENT_IMAGE:
                self.handleImageEvent()
            elif nEvent == nncam.NNCAM_EVENT_EXPOSURE:
                self.handleExpoEvent()
            elif nEvent == nncam.NNCAM_EVENT_TEMPTINT:
                self.handleTempTintEvent()
            elif nEvent == nncam.NNCAM_EVENT_STILLIMAGE:
                self.handleStillImageEvent()
            elif nEvent == nncam.NNCAM_EVENT_ERROR:
                self.closeCamera()
            elif nEvent == nncam.NNCAM_EVENT_DISCONNECTED:
                self.closeCamera()

    def handleImageEvent(self):
        try:
            self.hcam.PullImageV3(self.pData, 0, 24, 0, None)
        except:
            pass
        else:
            img = QImage(self.pData, self.imgWidth, self.imgHeight, QImage.Format_RGB888)
            self.lbl_video.setPixmap(
                QPixmap.fromImage(img.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.FastTransformation)))

    def handleExpoEvent(self):
        t, g = self.hcam.get_ExpoTime(), self.hcam.get_ExpoAGain()
        with QSignalBlocker(self.slider_expoTime): self.slider_expoTime.setValue(t)
        with QSignalBlocker(self.slider_expoGain): self.slider_expoGain.setValue(g)
        self.lbl_expoTime.setText(str(t));
        self.lbl_expoGain.setText(str(g))

    def handleTempTintEvent(self):
        t, ti = self.hcam.get_TempTint()
        with QSignalBlocker(self.slider_temp): self.slider_temp.setValue(t)
        with QSignalBlocker(self.slider_tint): self.slider_tint.setValue(ti)
        self.lbl_temp.setText(str(t));
        self.lbl_tint.setText(str(ti))

    def handleStillImageEvent(self):
        info = nncam.NncamFrameInfoV3()
        try:
            self.hcam.PullImageV3(None, 1, 24, 0, info)
            if info.width > 0:
                buf = bytes(nncam.TDIBWIDTHBYTES(info.width * 24) * info.height)
                self.hcam.PullImageV3(buf, 1, 24, 0, info)
                img = QImage(buf, info.width, info.height, QImage.Format_RGB888)
                self.count += 1;
                img.save(f"./datasets/{self.count}.jpg")
        except:
            pass



if __name__ == '__main__':
    # nncam.Nncam.GigeEnable(None, None)
    app = QApplication(sys.argv)
    mw = MainWidget()
    mw.show()
    sys.exit(app.exec_())