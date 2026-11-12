"""Non-OpenGL startup check for the headless macOS build runner."""
import importlib


def check_runtime():
    import aicspylibczi
    import imagecodecs
    import magicgui
    import napari
    import vispy
    from napari.components import ViewerModel
    from napari.qt import get_qapp
    from qtpy.QtWidgets import QWidget

    application = get_qapp()
    widget = QWidget()
    model = ViewerModel()
    for name in (
        'src.cellpose_backend', 'src.controls', 'src.display',
        'src.foci_measurement', 'src.focus_deduplication', 'src.memory',
        'src.nuclei_segmenter', 'src.preview', 'src.spot_detector',
        'src.spot_selection', 'src.utils',
    ):
        importlib.import_module(name)
    application.processEvents()
    model.layers.clear()
    widget.close()
