"""Conservative consolidation of repeated axial peaks before focus measurement."""
import math
import numpy as np

from src.memory import check_cancelled, filter_volume, release


def consolidate_z_duplicates(candidates, spot_stack, labels, spacing_um_zyx,
                             xy_tolerance_um=.35, min_bridge_ratio=.8):
    """Merge aligned Z peaks only when their raw-signal profile has no deep dip.

    Uses physical spacing, a local background estimate, and the original foci
    processing channel rather than the Z-normalized detector/display images.
    Every member must match every existing member of a group; proximity alone
    and chains of individually close detections cannot join distinct groups.
    """
    spacing = np.asarray(spacing_um_zyx, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError('Z duplicate consolidation requires positive calibrated Z/Y/X spacing.')
    if not math.isfinite(xy_tolerance_um) or xy_tolerance_um <= 0:
        raise ValueError('Z duplicate XY tolerance must be positive micrometres.')
    if not math.isfinite(min_bridge_ratio) or not 0 < min_bridge_ratio <= 1:
        raise ValueError('Z duplicate bridge ratio must be greater than zero and at most one.')
    diagnostics = dict(enabled=True, method='axial_signal_continuity',
                       xy_tolerance_um=xy_tolerance_um, min_bridge_ratio=min_bridge_ratio,
                       smoothing_um=[.2, .08, .08], input_candidates=len(candidates),
                       retained_candidates=len(candidates), removed_count=0, groups=[])
    if len(candidates) < 2:
        return candidates, diagnostics
    xy_spacing = float(np.sqrt(np.prod(spacing[1:])))

    def aligned(a, b):
        if a['label_id'] != b['label_id'] or a['z'] == b['z']:
            return False
        distance = np.linalg.norm((np.array([a['y'], a['x']]) - [b['y'], b['x']]) * spacing[1:])
        # Only nearly coincident centers are eligible, even for large features.
        radius_um = min(a['radius'], b['radius']) * xy_spacing
        tolerance = min(xy_tolerance_um, max(.15, .6 * radius_um))
        return distance <= tolerance

    # Avoid an extra volume/filter when there are no aligned axial proposals.
    pairs = set()
    for i, a in enumerate(candidates):
        check_cancelled()
        for j in range(i):
            if aligned(a, candidates[j]):
                pairs.add((j, i))
    if not pairs:
        return candidates, diagnostics
    smooth = filter_volume(spot_stack, 'gaussian', sigma=tuple(np.array([.2, .08, .08]) / spacing))
    cache = {}

    def bridge(i, j):
        key = tuple(sorted((i, j)))
        if key not in pairs:
            return None
        if key in cache:
            return cache[key]
        a, b = candidates[i], candidates[j]
        if a['z'] > b['z']:
            a, b = b, a
        radius_um = min(a['radius'], b['radius']) * xy_spacing
        core_radius = min(.2, max(.12, .4 * radius_um))
        shell_inner = max(.4, 1.8 * radius_um)
        shell_outer = max(.8, 3. * radius_um)
        pad_y, pad_x = np.ceil(shell_outer / spacing[1:]).astype(int)
        profile = []
        for z in range(int(a['z']), int(b['z']) + 1):
            check_cancelled()
            fraction = (z - a['z']) / (b['z'] - a['z'])
            y = a['y'] + fraction * (b['y'] - a['y'])
            x = a['x'] + fraction * (b['x'] - a['x'])
            cy, cx = int(round(y)), int(round(x))
            if (not 0 <= cy < labels.shape[1] or not 0 <= cx < labels.shape[2]
                    or labels[z, cy, cx] != a['label_id']):
                cache[key] = None
                return None
            y0, y1 = max(0, cy-pad_y), min(labels.shape[1], cy+pad_y+1)
            x0, x1 = max(0, cx-pad_x), min(labels.shape[2], cx+pad_x+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance2 = ((yy-y)*spacing[1])**2 + ((xx-x)*spacing[2])**2
            patch = smooth[z, y0:y1, x0:x1]
            core = patch[distance2 <= core_radius**2]
            shell = patch[(distance2 >= shell_inner**2) & (distance2 <= shell_outer**2)]
            if (not core.size or shell.size < 8 or not np.all(np.isfinite(core))
                    or not np.all(np.isfinite(shell))):
                cache[key] = None
                return None
            profile.append(max(0., float(np.percentile(core, 95) - np.median(shell))))
        endpoint = min(profile[0], profile[-1])
        ratio = min(profile) / endpoint if endpoint > 0 else 0.
        cache[key] = ratio if ratio >= min_bridge_ratio else None
        return cache[key]

    try:
        groups = []
        # Respect the existing priority of earlier scale passes; use the strongest
        # peak within that pass as the representative measurement center.
        order = sorted(range(len(candidates)), key=lambda i: (
            candidates[i].get('pass_order', 0), -candidates[i]['score'],
            candidates[i]['z'], candidates[i]['y'], candidates[i]['x']))
        for index in order:
            check_cancelled()
            for group in groups:
                if all(bridge(index, other) is not None for other in group):
                    group.append(index)
                    break
            else:
                groups.append([index])
        kept = [candidates[group[0]] for group in groups]
        for group in groups:
            if len(group) < 2:
                continue
            representative = candidates[group[0]]
            diagnostics['groups'].append(dict(
                nucleus_id=int(representative['label_id']),
                representative_zyx=[int(representative[k]) for k in ('z', 'y', 'x')],
                merged_centers_zyx=[[int(candidates[i][k]) for k in ('z', 'y', 'x')] for i in group[1:]],
                coordinate_grid='processing',
            ))
        diagnostics['retained_candidates'] = len(kept)
        diagnostics['removed_count'] = len(candidates) - len(kept)
        return kept, diagnostics
    finally:
        release(smooth)
