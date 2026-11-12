import traceback
import signal
from dataclasses import dataclass
from threading import Event
from pathlib import Path
from typing import Sequence

import napari
import numpy as np
from magicgui import magicgui
from napari.qt import get_qapp
from napari.qt.threading import thread_worker, FunctionWorker
from napari.utils.notifications import show_info, show_error, show_warning
from qtpy.QtCore import QObject, Signal, QTimer
from qtpy.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
    QVBoxLayout, QTextEdit, QWidget, QPushButton,
)

from src.controls import DetectionControls
from src.spot_detector import run_detection
from src.foci_measurement import focus_numbers_within_nuclei
from src.spot_selection import (
    SpotSearchSettings, apply_spot_selection, selection_warnings,
)
from src.utils import (
    save_counts_csv, save_focus_measurements_csv,
    SUPPORTED_IMAGE_EXTENSIONS, lazy_stack, image_voxel_size_um,
)
from src.memory import DetectionCancelled
from src.cellpose_backend import segmentation_session
from src.preview import prepare_3d_preview


# Create the Qt application before decorators construct widgets.
qt_app = get_qapp()
state = {}


class ProgressSignals(QObject):
    status = Signal(str)


progress_signals = ProgressSignals()


@dataclass
class BatchOutcome:
    results: list
    failures: list


def mask_needs_review(result):
    diagnostics = getattr(result, 'segmentation_diagnostics', None) or {}
    return bool(diagnostics.get('mask_review_required') or diagnostics.get('diameter_clamped_for_memory'))


class DetectionWorker(FunctionWorker):
    """Honor Napari's shutdown/cancel request inside the detection pipeline."""
    def __init__(self, function, *args, **kwargs):
        self.cancel_event = kwargs.setdefault('cancel_event', Event())
        super().__init__(function, *args, **kwargs)

    def quit(self):
        self.cancel_event.set()
        super().quit()


@thread_worker(progress=True, ignore_errors=True, worker_class=DetectionWorker)
@segmentation_session()
def run_batch_detection(file_paths, cancel_event,
                        voxel_size_overrides=None, segmentation_settings=None, spot_search_settings=None,
                        spot_size_limits=None):
    results = []
    failures = []
    voxel_size_overrides = voxel_size_overrides or {}
    segmentation_settings = dict(segmentation_settings or {})
    segmentation_settings["backend"] = "cellpose"
    for i, file_path in enumerate(file_paths):
        def progress(message):
            if cancel_event.is_set():
                raise DetectionCancelled('Detection cancelled.')
            progress_signals.status.emit(
                f"File {i + 1}/{len(file_paths)} ({Path(file_path).name}): {message}")
        try:
            progress('Starting detection')
            result = run_detection(str(file_path), progress_callback=progress,
                                   cancel_event=cancel_event,
                                   smart_downsample=True,
                                   segmentation_settings=segmentation_settings,
                                   spot_search_settings=spot_search_settings,
                                   spot_size_limits=spot_size_limits,
                                   voxel_size_override_um_zyx=voxel_size_overrides.get(
                                       str(Path(file_path).resolve())))
        except DetectionCancelled:
            # Normal cancellation is not a detection failure or a Qt error.
            return None
        except Exception as error:
            if cancel_event.is_set():
                return None
            reason = f"{type(error).__name__}: {error}"
            failures.append((str(Path(file_path).resolve()), reason))
            traceback.print_exception(type(error), error, error.__traceback__)
            progress(f"Skipped: {reason}")
            continue
        results.append(result)
    return BatchOutcome(results, failures)


def _native_points(blobs, result):
    points = np.asarray(blobs, dtype=float).reshape(-1, 4)[:, :3].copy()
    points[:, 0] += getattr(result, 'trimmed_z_planes', 0)
    scale = np.asarray(getattr(result, 'processing_scale_zyx', (1, 1, 1)))
    return (points + .5) * scale - .5


def _candidate_features(candidates):
    names = ('focus_id', 'nucleus_id', 'watershed_volume_um3',
             'watershed_equivalent_diameter_um', 'gaussian_fwhm_z_um',
             'gaussian_fwhm_y_um', 'gaussian_fwhm_x_um', 'gaussian_halfmax_volume_um3',
             'gaussian_fit_r2', 'measurement_status', 'qc_flags')
    features = {name: [c.measurement.get(name) for c in candidates] for name in names}
    features['candidate_id'] = [c.focus_id for c in candidates]
    features['focus_id'] = list(focus_numbers_within_nuclei(c.nucleus_id for c in candidates))
    features['typical_xy_width_um'] = [c.width_um for c in candidates]
    return features


def current_size_limits():
    return controls.size_limits()


def current_search_settings():
    return controls.search_settings()


def refresh_controls():
    return controls.refresh_state(state.get('completed_search'), bool(state.get('results')),
                                  state.get('busy', False),
                                  preview_loading=state.get('preview_worker') is not None,
                                  preview_open=state.get('preview_viewer') is not None)


def spot_controls_changed(*_):
    refresh_controls()


def update_selection_report():
    results = state.get('results', [])
    lines = [state.get('batch_report_base', '')]
    if results:
        lines.append('\nSpots after watershed-volume filtering:')
    for result in results:
        if getattr(result, 'spot_candidates', None) is None:
            continue
        statuses = result.spot_selection.statuses
        minimum = result.spot_selection.limits.min_volume_um3
        rule = f'volume > {minimum:g} µm³' if minimum is not None else 'volume filter off'
        lines.append(f'{result.file_stem}: {statuses.count("accepted")} counted / '
                     f'{len(statuses)} detected; {rule}; '
                     f'{statuses.count("below_volume_limit")} below or equal to limit; '
                     f'{statuses.count("volume_unavailable")} without valid volume.')
        lines.append('  ' + ', '.join(f'nucleus {i}: {count} spots'
                                     for i, count in result.nucleus_counts.items()))
        searched = result.spot_search_diagnostics.get('searched_width_um')
        if searched and searched[0] is not None:
            lines.append(f'  Completed approximate search: {searched[0]:.3g} to {searched[1]:.3g} µm')
        consolidation = result.spot_search_diagnostics.get('z_consolidation', {})
        if consolidation.get('enabled'):
            lines.append(f'  Cross-Z consolidation: {consolidation["removed_count"]} repeated peaks removed '
                         f'before measurement ({consolidation["input_candidates"]} to '
                         f'{consolidation["retained_candidates"]} candidate centers).')
        lines.extend('  ' + message for message in selection_warnings(result))
    batch_report.setPlainText('\n'.join(lines).strip())


def update_spot_selection(*_):
    try:
        limits = current_size_limits()
    except ValueError:
        refresh_controls()
        return
    results = state.get('results', [])
    for result in results:
        if getattr(result, 'spot_candidates', None) is not None:
            apply_spot_selection(result, limits)
    if not results:
        return
    result = results[-1]
    candidates = getattr(result, 'spot_candidates', None)
    if candidates is None:
        return
    for status, key in (('accepted', 'accepted_layer'), ('size_uncertain', 'uncertain_layer')):
        selected = [c for c, value in zip(candidates, result.spot_selection.statuses)
                    if status == 'accepted' and value == 'accepted']
        layer = state.get(key)
        if layer is not None:
            layer.data = _native_points([c.blob for c in selected], result)
            layer.features = _candidate_features(selected)
    accepted = len(result.blobs_filtered)
    controls.counts.setText(f'{result.file_stem}: {accepted} counted / {len(candidates)} detected.')
    viewer.text_overlay.text = (f'{result.file_stem}: {accepted} counted / {len(candidates)} detected across Z\n'
                               'Use the Z slider to inspect other slices')
    viewer.status = f'{result.file_stem}: {accepted} counted / {len(candidates)} detected'
    update_3d_spots()
    update_selection_report()


def update_display_view(*_):
    processed = controls.image_view.currentText() == 'Processed'
    for layer in state.get('raw_layers', []):
        layer.visible = not processed
    for layer in state.get('processed_layers', []):
        layer.visible = processed


@thread_worker(ignore_errors=True, worker_class=DetectionWorker)
def build_3d_preview(result, cancel_event):
    return prepare_3d_preview(result, cancel_event)


def update_3d_spots():
    preview = state.get('preview_viewer')
    result = state.get('preview_source')
    layer = state.get('preview_spots')
    if preview is None or result is None or layer is None:
        return
    candidates = getattr(result, 'spot_candidates', None)
    if candidates is not None:
        selected = [c for c, status in zip(candidates, result.spot_selection.statuses)
                    if status == 'accepted']
        points = _native_points([c.blob for c in selected], result)
        features = _candidate_features(selected)
    else:
        points = _native_points(result.blobs_filtered, result)
        features = {'focus_id': np.arange(1, len(points) + 1)}
    layer.data = points * np.asarray(result.voxel_size_um_zyx)
    layer.features = features
    preview.text_overlay.text = (f'{result.file_stem}: {len(points)} counted spots\n'
                                 '3D preview. Drag to rotate; scroll to zoom.')


def close_3d_preview():
    worker = state.pop('preview_worker', None)
    if worker is not None:
        worker.quit()
    preview = state.pop('preview_viewer', None)
    state.pop('preview_spots', None)
    state.pop('preview_source', None)
    if preview is not None:
        try:
            preview.close()
        except RuntimeError:
            pass
    try:
        refresh_controls()
        main_window = getattr(viewer.window, '_qt_window', None)
        if main_window is not None:
            main_window.raise_()
            main_window.activateWindow()
    except RuntimeError:
        pass


def toggle_3d_preview():
    if state.get('preview_viewer') is not None:
        close_3d_preview()
        return
    if state.get('busy') or state.get('preview_worker') is not None or not state.get('results'):
        return
    result = state['results'][-1]
    worker = build_3d_preview(result, cancel_event=Event())
    state['preview_worker'] = worker
    refresh_controls()
    viewer.status = 'Preparing the processed 3D preview...'

    def on_ready(data):
        if (worker.cancel_event.is_set() or not state.get('results')
                or state['results'][-1] is not result):
            return
        preview = None
        try:
            preview = napari.Viewer(title=f'{result.file_stem}: 3D preview', ndisplay=3,
                                    axis_labels=('Z (µm)', 'Y (µm)', 'X (µm)'))
            state['preview_viewer'] = preview
            state['preview_source'] = result
            preview.add_image(data.nuclei, name='Nuclei (3D preview)', colormap='green',
                              contrast_limits=(0., 1.), rendering='mip', blending='additive',
                              scale=data.spacing_um, translate=data.origin_um)
            preview.add_image(data.foci, name='Foci (3D preview)', colormap='magenta',
                              contrast_limits=result.processed_foci_limits or (0., 1.),
                              rendering='mip', blending='additive',
                              scale=data.spacing_um, translate=data.origin_um)
            preview.add_labels(data.labels, name='Nuclear masks (3D preview)', opacity=.08,
                               scale=data.spacing_um, translate=data.origin_um)
            state['preview_spots'] = preview.add_points(
                np.empty((0, 3)), ndim=3, name='Counted spots', size=.25,
                face_color='red', border_color='white', canvas_size_limits=(5, 15),
                features={'focus_id': np.empty(0, dtype=int)},
                text={'string': '   {focus_id:.0f}', 'color': 'white', 'size': 12})
            preview.text_overlay.visible = True
            preview.scale_bar.visible = True
            preview.scale_bar.unit = 'um'
            update_3d_spots()
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.addWidget(QLabel('Display preview only. Measurements use the original image.'))
            back = QPushButton('Back to slices')
            back.clicked.connect(close_3d_preview)
            layout.addWidget(back)
            preview.window.add_dock_widget(panel, name='3D view', area='right')
            preview.reset_view()

            def on_closed(*_):
                if state.get('preview_viewer') is preview:
                    state.pop('preview_viewer', None)
                    state.pop('preview_spots', None)
                    state.pop('preview_source', None)
                    try:
                        refresh_controls()
                    except RuntimeError:
                        pass
            preview.window._qt_window.destroyed.connect(on_closed)
        except Exception as error:
            close_3d_preview()
            show_error(f'Could not open 3D preview: {error}')

    def on_error(error):
        if not worker.cancel_event.is_set():
            show_error(f'Could not prepare 3D preview: {error}')

    def on_finished():
        if state.get('preview_worker') is worker:
            state.pop('preview_worker', None)
            try:
                refresh_controls()
            except RuntimeError:
                pass

    worker.returned.connect(on_ready)
    worker.errored.connect(on_error)
    worker.finished.connect(on_finished)
    worker.start()


def show_results(results):
    close_3d_preview()
    # Clear old layers before releasing their disk mappings.
    viewer.layers.clear()
    state['results'] = results
    state['raw_layers'], state['processed_layers'] = [], []
    last = results[-1]
    channel_names = ['DNA', 'Foci', 'AF488', 'Ch4']
    colormaps = ['gray', 'magenta', 'green', 'yellow']
    for c in range(last.img.shape[1]):
        layer = viewer.add_image(
            lazy_stack(last.img[:, c], last._workspace),
            name='Raw ' + (channel_names[c] if c < len(channel_names) else f'Channel {c}'),
            colormap=colormaps[c] if c < len(colormaps) else 'gray',
            rendering='mip', contrast_limits=last.contrast_limits[c], visible=False,
            blending='additive',
        )
        state['raw_layers'].append(layer)
    scale = np.asarray(getattr(last, 'processing_scale_zyx', (1, 1, 1)), dtype=float)
    # Apply the same block-center transform to images, labels and focus coordinates.
    trim = getattr(last, 'trimmed_z_planes', 0)
    translate = (scale - 1) / 2 + np.array([trim, 0, 0]) * scale
    for attr, name, color, limits in (
        ('processed_nuclei', 'Processed nuclei', 'green', (0., 1.)),
        ('processed_foci', 'Processed foci', 'magenta', getattr(last, 'processed_foci_limits', (0., 1.))),
    ):
        stack = getattr(last, attr, None)
        if stack is not None:
            layer = viewer.add_image(lazy_stack(stack, last._workspace), name=name,
                                     colormap=color, contrast_limits=limits, rendering='mip',
                                     scale=tuple(scale), translate=tuple(translate), blending='additive')
            state['processed_layers'].append(layer)
    viewer.add_labels(lazy_stack(last.roi_labels_3d, last._workspace),
                      name=f'Accepted nuclei ({last.nucleus_count})', opacity=.12, visible=False,
                      scale=tuple(scale), translate=tuple(translate))
    state.pop('uncertain_layer', None)
    for key, name, color in (('accepted_layer', 'Detected Spots', 'red'),):
        state[key] = viewer.add_points(
            np.empty((0, 3)), name=name, size=10, face_color=color,
            border_color='white', border_width=.1, canvas_size_limits=(10, 30),
            out_of_slice_display=False, ndim=3, visible=key == 'accepted_layer',
            features={'focus_id': np.empty(0, dtype=int)},
            text={'string': '   {focus_id:.0f}', 'color': 'white', 'size': 12,
                  'anchor': 'upper_left', 'scaling': False, 'visible': True})
    if getattr(last, 'spot_candidates', None) is not None:
        update_spot_selection()
    else:
        state['accepted_layer'].data = _native_points(last.blobs_filtered, last)
        measurements = getattr(last, 'focus_measurements', None) or []
        if len(measurements) == len(last.blobs_filtered):
            records = [m.to_record() for m in measurements]
            names = ('focus_id', 'nucleus_id', 'watershed_volume_um3')
            features = {name: [r.get(name) for r in records] for name in names}
            features['focus_id'] = list(focus_numbers_within_nuclei(r['nucleus_id'] for r in records))
            state['accepted_layer'].features = features
        else:
            state['accepted_layer'].features = {
                'focus_id': np.arange(1, len(last.blobs_filtered) + 1, dtype=int)}
        viewer.text_overlay.text = f'{last.file_stem}: {len(last.blobs_filtered)} spots across all Z slices'
    update_display_view()
    if not state['processed_layers']:
        for layer in state['raw_layers']:
            layer.visible = True
    points = state['accepted_layer'].data
    if len(points):
        slices, counts = np.unique(points[:, 0], return_counts=True)
        initial_z = float(slices[np.argmax(counts)])
    else:
        initial_z = last.img.shape[0] // 2
    viewer.dims.set_point(0, initial_z)


def request_voxel_size(file_path):
    """Ask for physical spacing only when a file has no usable metadata."""
    dialog = QDialog(viewer.window._qt_window)
    dialog.setWindowTitle('Missing voxel-size metadata')
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(
        f'{Path(file_path).name} does not contain valid physical voxel sizes.\n'
        'Enter the Z, Y, and X size of one voxel in micrometres.'
    ))
    form = QFormLayout()
    fields = []
    for axis in 'ZYX':
        field = QDoubleSpinBox()
        field.setDecimals(6)
        field.setRange(0.0, 1000.0)
        field.setSingleStep(0.01)
        field.setSpecialValueText('Required')
        form.addRow(f'Voxel {axis} (µm)', field)
        fields.append(field)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    ok_button = buttons.button(QDialogButtonBox.Ok)
    ok_button.setEnabled(False)

    def update_ok_button():
        ok_button.setEnabled(all(field.value() > 0 for field in fields))

    for field in fields:
        field.valueChanged.connect(update_ok_button)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec_() != QDialog.Accepted:
        return None
    return tuple(field.value() for field in fields)


def collect_voxel_size_overrides(paths):
    """Return per-file manual calibration, or None when the user cancels."""
    overrides = {}
    for file_path in paths:
        try:
            spacing = image_voxel_size_um(file_path)
        except Exception:
            # Let the worker report unreadable files and continue the batch.
            continue
        if spacing is None:
            manual = request_voxel_size(file_path)
            if manual is None:
                return None
            overrides[str(Path(file_path).resolve())] = manual
    return overrides


def start_detection(paths, segmentation_settings=None, spot_search_settings=None):
    if state.get("busy"):
        show_info("Detection is already running. See the status bar for progress.")
        return
    if not paths:
        show_info("Please select LSM or CZI files.")
        return
    unsupported = [str(p) for p in paths
                   if Path(p).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS]
    if unsupported:
        show_error(f"Unsupported file type: {unsupported[0]}")
        return
    if not refresh_controls():
        return
    try:
        search = (SpotSearchSettings(**spot_search_settings) if isinstance(spot_search_settings, dict)
                  else spot_search_settings or current_search_settings())
        segmentation_settings = dict(segmentation_settings or controls.segmentation_settings())
        segmentation_settings["backend"] = "cellpose"
    except ValueError as error:
        show_error(str(error))
        return
    voxel_size_overrides = collect_voxel_size_overrides(paths)
    if voxel_size_overrides is None:
        show_info('Detection cancelled: voxel-size calibration was not provided.')
        return
    state["busy"] = True
    close_3d_preview()
    detect_spots.enabled = False
    refresh_controls()
    viewer.status = f"Processing {len(paths)} file(s)..."
    cancel_event = Event()
    state['cancel_event'] = cancel_event
    worker = run_batch_detection(paths, cancel_event=cancel_event,
                                 segmentation_settings=segmentation_settings,
                                 spot_search_settings=search,
                                 spot_size_limits=current_size_limits(),
                                 voxel_size_overrides=voxel_size_overrides)
    state["worker"] = worker

    def on_done(outcome):
        if cancel_event.is_set() or outcome is None:
            return
        try:
            state['failures'] = outcome.failures
            if outcome.results:
                state['completed_search'] = search
                show_results(outcome.results)
            else:
                viewer.layers.clear()
                state['raw_layers'], state['processed_layers'] = [], []
                state.pop('accepted_layer', None)
                state.pop('uncertain_layer', None)
                state.pop('results', None)
                state.pop('completed_search', None)
                controls.counts.setText('No images processed successfully.')
                viewer.text_overlay.text = 'No images processed successfully. See Batch report.'
            summary = (f"Batch complete: {len(outcome.results)}/{len(paths)} images processed; "
                       f"{len(outcome.failures)} skipped.")
            reviews = [result for result in outcome.results
                       if mask_needs_review(result)]
            if reviews:
                summary += f' {len(reviews)} mask(s) need review.'
            details = [summary]
            for result in outcome.results:
                diagnostics = getattr(result, 'segmentation_diagnostics', None)
                if diagnostics:
                    details.append(
                        f"{result.file_stem}: {result.nucleus_count} nuclei, "
                        f"Cellpose {diagnostics['model']} ({diagnostics.get('mode', '3d')}), "
                        f"diameter {diagnostics['diameter_um']:.1f} µm, "
                        f"{diagnostics['total_seconds']:.1f} s, "
                        f"helper peak {diagnostics['helper_peak_memory_bytes']/1024**3:.2f} GiB")
                    removed = diagnostics.get('rejected_small_instances', [])
                    if removed:
                        details.append(f"  Excluded {len(removed)} small objects below "
                                       f"{diagnostics['min_volume_um3']:g} µm³: " +
                                       ', '.join(f"{item['volume_um3']:.1f} µm³" for item in removed))
                    unsupported = diagnostics.get('rejected_low_support_instances', [])
                    if unsupported:
                        details.append(f'  Excluded {len(unsupported)} masks without sufficient local nuclear signal.')
                    if diagnostics.get('diameter_clamped_for_memory'):
                        details.append('  Review mask: the automatic diameter was increased to '
                                       'keep CPU inference within the memory budget.')
                    if diagnostics.get('signal_support_unavailable'):
                        details.append('  Review mask: local background was unavailable for some instances.')
            if outcome.failures:
                details.append('\nSkipped files:')
                details.extend(f"{path}\n  {reason}" for path, reason in outcome.failures)
            state['batch_report_base'] = '\n'.join(details)
            update_selection_report()
            refresh_controls()
            viewer.status = summary
            if not outcome.results:
                show_error(summary + '\nSee Batch report for filenames and reasons.')
            elif outcome.failures or reviews:
                show_warning(summary + '\nSee Batch report for filenames and reasons.')
            else:
                show_info(summary)
        except Exception as error:
            on_error(error)

    def on_error(error):
        if cancel_event.is_set():
            return
        viewer.status = "Detection failed"
        traceback.print_exception(type(error), error, error.__traceback__)
        show_error(f"Detection failed: {error}")

    def on_finished():
        state["busy"] = False
        try:
            detect_spots.enabled = True
            refresh_controls()
        except RuntimeError:
            pass  # The widget may already have been deleted during shutdown.
        state.pop("worker", None)

    worker.returned.connect(on_done)
    worker.errored.connect(on_error)
    worker.finished.connect(on_finished)
    worker.start()


@magicgui(file_path={"label": "LSM/CZI Files", "mode": "rm"}, call_button=False)
def detect_spots(file_path: Sequence[Path]):
    start_detection(file_path)


def handle_file_drop(viewer, paths):
    images = [p for p in paths if Path(p).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    if not images:
        show_info("No LSM or CZI files detected.")
        return
    start_detection(images)


@magicgui(call_button="Export CSV")
def export_csv():
    if "results" not in state:
        show_info("Run detection first.")
        return
    if state.get("busy") or not refresh_controls():
        return
    try:
        update_spot_selection()
        counts_path = save_counts_csv(state["results"])
        measurements_path = save_focus_measurements_csv(state["results"])
        viewer.status = "Count and focus-measurement CSV files saved"
        detected = counted = 0
        for result in state['results']:
            selection = getattr(result, 'spot_selection', None)
            if selection is not None:
                statuses = selection.statuses
                detected += len(statuses)
                counted += statuses.count('accepted')
            else:
                count = len(result.blobs_filtered)
                detected += count
                counted += count
        message = (f"CSV files saved at:\n{counts_path}\n{measurements_path}\n"
                   f"Count CSV: {counted} spots after volume filtering.\n"
                   f"Measurement CSV: the same {counted} spots shown in the viewer.")
        if detected and not counted:
            message += '\nNo candidates passed the current volume filter. Lower the limit or set it to zero to keep all.'
        reviews = [result.file_stem for result in state['results']
                   if mask_needs_review(result)]
        if reviews:
            message += '\nThese exported images have masks flagged for review: ' + ', '.join(reviews)
            show_warning(message)
        else:
            show_info(message)
    except Exception as error:
        show_error(f"Export failed: {error}")


# Slice viewing requests only the selected native-resolution Z plane.
viewer = napari.Viewer(ndisplay=2)
def update_status(message):
    event = state.get('cancel_event')
    if event is None or not event.is_set():
        viewer.status = message


progress_signals.status.connect(update_status)
viewer.text_overlay.visible = True
viewer.text_overlay.text = "Drop .lsm or .czi files here or use Run Detection"
qt_window = viewer.window._qt_window
qt_window.setAcceptDrops(True)


def dropEvent(event):
    handle_file_drop(viewer, [url.toLocalFile() for url in event.mimeData().urls()])
    event.accept()


qt_window.dropEvent = dropEvent
controls = DetectionControls(detect_spots.native, export_csv.native)
controls.changed.connect(spot_controls_changed)
controls.volume_changed.connect(update_spot_selection)
controls.view_changed.connect(update_display_view)
controls.preview_button.clicked.connect(toggle_3d_preview)
controls.run_button.clicked.connect(lambda: start_detection(detect_spots.file_path.value))
viewer.window.add_dock_widget(controls, name='Detection', area='right')
batch_report = QTextEdit()
batch_report.setReadOnly(True)
batch_report.setPlaceholderText('Batch results and skipped-file reasons will appear here.')
viewer.window.add_dock_widget(batch_report, name='Batch report', area='bottom')


def request_stop():
    close_3d_preview()
    event = state.get('cancel_event')
    if event is not None:
        event.set()
    worker = state.get('worker')
    if worker is not None:
        worker.quit()


def main():
    # Qt's native event loop otherwise delays Python's Ctrl+C handling.
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(100)
    previous_handler = signal.getsignal(signal.SIGINT)

    def interrupt(signum, frame):
        request_stop()
        viewer.close()

    signal.signal(signal.SIGINT, interrupt)
    try:
        napari.run()
    finally:
        request_stop()
        heartbeat.stop()
        signal.signal(signal.SIGINT, previous_handler)
        state.pop('results', None)


qt_window.destroyed.connect(lambda *_: request_stop())

if __name__ == "__main__":
    main()
