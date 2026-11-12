"""Application controls with automatic or custom spot search sizes."""
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QPushButton, QToolButton,
)

from src.spot_selection import SpotSearchSettings, SpotSizeLimits


class DetectionControls(QScrollArea):
    changed = Signal()
    view_changed = Signal()
    volume_changed = Signal()

    def __init__(self, file_widget, export_widget):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.setWidget(content)
        layout = QVBoxLayout(content)

        def section(title):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            layout.addWidget(box)
            return box, box_layout

        def number(value, minimum=0., maximum=1000.):
            field = QDoubleSpinBox()
            field.setDecimals(3)
            field.setRange(minimum, maximum)
            field.setSingleStep(.1)
            field.setValue(value)
            field.setKeyboardTracking(False)
            return field

        _, images = section('Images')
        images.addWidget(file_widget)

        self.nuclei_group, nuclei = section('Nuclei')
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.model = QComboBox()
        self.model.addItems(['cyto3', 'nuclei'])
        self.diameter = number(0.)
        self.diameter.setSpecialValueText('Automatic')
        self.diameter.setToolTip('Approximate nucleus diameter in µm. Zero estimates it automatically.')
        form.addRow('Cellpose model', self.model)
        form.addRow('Nucleus diameter (µm)', self.diameter)
        nuclei.addLayout(form)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText('Advanced nucleus settings')
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        nuclei.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        self.method = QComboBox()
        self.method.addItems(['3D volume', 'XY slices linked in Z'])
        self.minimum_volume = number(100., maximum=1000000.)
        self.minimum_volume.setSpecialValueText('Keep all objects')
        advanced_form.addRow('Cellpose method', self.method)
        advanced_form.addRow('Minimum nuclear volume (µm³)', self.minimum_volume)
        self.advanced.setVisible(False)
        nuclei.addWidget(self.advanced)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)

        self.spots_group, spots = section('Spots')
        mode_form = QFormLayout()
        self.size_mode = QComboBox()
        self.size_mode.addItems(['Automatic', 'Custom range'])
        mode_form.addRow('Spot sizes', self.size_mode)
        spots.addLayout(mode_form)
        self.automatic_note = QLabel('Uses the current automatic spot search.')
        self.automatic_note.setWordWrap(True)
        spots.addWidget(self.automatic_note)
        self.custom_range = QWidget()
        custom_layout = QVBoxLayout(self.custom_range)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        range_form = QFormLayout()
        self.minimum_size = number(.2, minimum=.001)
        self.maximum_size = number(10., minimum=.001)
        range_form.addRow('Minimum XY width (µm)', self.minimum_size)
        range_form.addRow('Maximum XY width (µm)', self.maximum_size)
        custom_layout.addLayout(range_form)
        search_note = QLabel('Run detection to search this approximate XY width range.')
        search_note.setWordWrap(True)
        custom_layout.addWidget(search_note)
        tooltip = ('Approximate XY spot sizes to search for, in micrometres. '
                   'Run detection after changing this range. '
                   'Search sizes are not exact watershed dimensions.')
        for field in (self.minimum_size, self.maximum_size):
            field.setToolTip(tooltip)
        spots.addWidget(self.custom_range)
        self.custom_range.setVisible(False)
        volume_form = QFormLayout()
        self.minimum_spot_volume = number(.2, maximum=1000000.)
        self.minimum_spot_volume.setSpecialValueText('Off (keep all)')
        self.minimum_spot_volume.setToolTip(
            'Count only spots with measured watershed volume strictly greater than this value. '
            'Updates existing results immediately. Zero disables this filter. '
            'Missing or invalid volumes are not counted while the filter is enabled.')
        volume_form.addRow('Minimum spot volume (µm³)', self.minimum_spot_volume)
        spots.addLayout(volume_form)
        self.volume_note = QLabel('Count spots with watershed volume above this limit. Zero keeps all.')
        self.volume_note.setWordWrap(True)
        spots.addWidget(self.volume_note)
        self.error = QLabel()
        self.error.setStyleSheet('color: #ff9b9b;')
        self.error.setWordWrap(True)
        spots.addWidget(self.error)
        self.pending = QLabel()
        self.pending.setWordWrap(True)
        self.pending.setStyleSheet('color: #ffd580;')
        spots.addWidget(self.pending)
        self.run_button = QPushButton('Run detection')
        spots.addWidget(self.run_button)

        _, review = section('View & Export')
        view_form = QFormLayout()
        self.image_view = QComboBox()
        self.image_view.addItems(['Processed', 'Raw'])
        view_form.addRow('Image view', self.image_view)
        review.addLayout(view_form)
        self.preview_button = QPushButton('View in 3D')
        self.preview_button.setToolTip(
            'Open a separate, smaller processed 3D preview. Drag to rotate and scroll to zoom. '
            'The slice view and all measurements stay unchanged.')
        review.addWidget(self.preview_button)
        self.counts = QLabel('Run detection to view spot counts.')
        self.counts.setWordWrap(True)
        review.addWidget(self.counts)
        review.addWidget(export_widget)
        self.export_widget = export_widget
        layout.addStretch()

        self.size_mode.currentTextChanged.connect(self._mode_changed)
        self.minimum_size.valueChanged.connect(lambda *_: self.changed.emit())
        self.maximum_size.valueChanged.connect(lambda *_: self.changed.emit())
        self.image_view.currentTextChanged.connect(lambda *_: self.view_changed.emit())
        self.minimum_spot_volume.valueChanged.connect(lambda *_: self.volume_changed.emit())
        self.refresh_state()

    def _toggle_advanced(self, expanded):
        self.advanced.setVisible(expanded)
        self.advanced_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _mode_changed(self, *_):
        custom = self.size_mode.currentText() == 'Custom range'
        self.custom_range.setVisible(custom)
        self.automatic_note.setVisible(not custom)
        self.changed.emit()

    def search_settings(self):
        if self.size_mode.currentText() == 'Automatic':
            return SpotSearchSettings()
        return SpotSearchSettings(self.minimum_size.value(), self.maximum_size.value())

    def segmentation_settings(self):
        return dict(backend='cellpose', model=self.model.currentText(),
                    diameter_um=self.diameter.value() or None,
                    mode='3d' if self.method.currentText() == '3D volume' else 'stitched',
                    min_volume_um3=self.minimum_volume.value())

    def size_limits(self):
        return SpotSizeLimits(min_volume_um3=self.minimum_spot_volume.value() or None)

    def refresh_state(self, completed_search=None, has_results=False, busy=False,
                      preview_loading=False, preview_open=False):
        try:
            selected = self.search_settings()
            error = ''
        except ValueError:
            selected = None
            error = 'Enter positive sizes, with minimum size no greater than maximum size.'
        self.error.setText(error)
        self.error.setVisible(bool(error))
        pending = (not error and has_results and completed_search is not None
                   and selected != completed_search)
        self.pending.setText('Search range changed. Run detection again to use it.'
                             if pending else '')
        self.pending.setVisible(bool(pending))
        self.run_button.setEnabled(not error and not busy)
        self.export_widget.setEnabled(not error and has_results and not busy)
        self.nuclei_group.setEnabled(not busy)
        self.spots_group.setEnabled(not busy)
        self.preview_button.setText('Preparing 3D...' if preview_loading else
                                    'Back to slices' if preview_open else 'View in 3D')
        self.preview_button.setEnabled(has_results and not busy and not preview_loading)
        return not bool(error)
