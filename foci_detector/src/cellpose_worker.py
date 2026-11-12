"""JSON-lines helper, executed ONLY by the isolated Cellpose 3 interpreter."""
import json
import sys
import time
import traceback
import os
from pathlib import Path

if getattr(sys, 'frozen', False):
    # Models ship alongside the helper; never depend on a developer's cache.
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(Path(sys.executable).resolve().parent / 'models')


def bounded_diameter(shape, anisotropy, diameter, model_diameter, manual=False, max_xy=128):
    """Bound Cellpose's internal rescaling, not just its input array."""
    max_rescale = min(1., max_xy/max(shape[1:]), 64/(shape[0]*anisotropy))
    minimum = float(model_diameter)/max_rescale
    clamped = diameter < minimum
    if clamped:
        if manual:
            raise ValueError('Requested nucleus diameter would exceed the CPU memory budget. '
                             'Use a larger diameter or a smaller field of view.')
        diameter = minimum
    if min(shape[1:])*float(model_diameter)/diameter < 8:
        raise ValueError('Estimated diameter is too large for this image; supply a diameter override.')
    return diameter, clamped


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Library progress must never corrupt the protocol.
    import numpy as np
    import psutil
    import torch
    from scipy.ndimage import gaussian_filter
    from cellpose import models

    torch.set_num_threads(4)
    cached = {}
    for line in sys.stdin:
        started = time.perf_counter()
        try:
            request = json.loads(line)
            name = request['model']
            if name not in ('cyto3', 'nuclei'):
                raise ValueError('Supported Cellpose models: cyto3, nuclei.')
            model_was_cached = name in cached
            if name not in cached:
                print(f'Loading Cellpose {name} weights', file=sys.stderr, flush=True)
                cached[name] = models.Cellpose(gpu=False, model_type=name)
            model = cached[name]
            if request.get('check_runtime'):
                protocol.write(json.dumps({'ok': True, 'model': name, 'weights_loaded': True}) + '\n')
                protocol.flush()
                continue
            image = np.load(request['input'], allow_pickle=False)
            diameter = request.get('diameter_px')
            if diameter is None:
                # The size estimator operates on 2D images, not whole volumes.
                # AF488 has fine granular structure. Smooth only the sizing
                # projection so the size model measures nuclei, not granules.
                sizing = gaussian_filter(image.mean(axis=0), request['sizing_sigma_px'])
                diameter, _ = model.sz.eval(sizing, channels=[0, 0])
                diameter = float(np.asarray(diameter).item())
            print(f'Cellpose {name}: grid {image.shape}, diameter {diameter:.2f} px, '
                  f'anisotropy {request["anisotropy"]:.3f}', file=sys.stderr, flush=True)
            if not np.isfinite(diameter) or diameter <= 0:
                raise ValueError('Cellpose could not estimate a positive nucleus diameter; supply an override.')
            estimated_diameter = diameter
            # Cellpose rescales again internally. A 512-pixel input is NOT a
            # memory bound when a tiny predicted diameter causes upsampling.
            # Bound the network grid as well, including anisotropic Z expansion.
            diameter, diameter_was_clamped = bounded_diameter(
                image.shape, request['anisotropy'], diameter, model.cp.diam_mean,
                manual=request.get('diameter_px') is not None, max_xy=request['network_max_xy'])
            masks, _, _, _ = model.eval(
                image ** request['intensity_gamma'], channels=[0, 0], channel_axis=None, z_axis=0,
                diameter=diameter, do_3D=request['mode'] == '3d', anisotropy=request['anisotropy'],
                stitch_threshold=.25 if request['mode'] == 'stitched' else 0.,
                normalize=False, resample=False, batch_size=4,
                flow3D_smooth=1.,
            )
            np.save(request['output'], masks.astype(np.int32), allow_pickle=False)
            memory = psutil.Process().memory_info()
            response = {'ok': True, 'diameter_px': diameter,
                        'model_was_cached': model_was_cached,
                        'flow3d_smooth': 1.,
                        'mode': request['mode'],
                        'intensity_gamma': request['intensity_gamma'],
                        'estimated_diameter_px': estimated_diameter,
                        'diameter_clamped_for_memory': diameter_was_clamped,
                        'network_max_xy': request['network_max_xy'], 'network_max_z': 64,
                        'inference_seconds': time.perf_counter() - started,
                        'helper_peak_memory_bytes': getattr(memory, 'peak_wset', memory.rss)}
        except Exception as error:
            traceback.print_exc()
            response = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        protocol.write(json.dumps(response) + '\n')
        protocol.flush()


if __name__ == '__main__':
    main()
