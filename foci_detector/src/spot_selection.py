"""Physical search settings and reversible selection of immutable candidates."""
from dataclasses import dataclass
from types import MappingProxyType
import math

import numpy as np
from src.foci_measurement import watershed_export_record, focus_numbers_within_nuclei

FWHM_FACTOR = 2 * math.sqrt(2 * math.log(2))


@dataclass(frozen=True)
class SpotSearchSettings:
    min_width_um: float | None = None
    max_width_um: float | None = None

    def __post_init__(self):
        if self.min_width_um is None and self.max_width_um is None:
            return
        if any(v is None or not math.isfinite(v) or v <= 0
               for v in (self.min_width_um, self.max_width_um)):
            raise ValueError('Search requires positive minimum and maximum widths in µm.')
        if self.min_width_um > self.max_width_um:
            raise ValueError('Minimum search width must not exceed maximum search width.')

    @property
    def manual(self):
        return self.min_width_um is not None

    def widths(self):
        if not self.manual:
            return ()
        steps = max(1, math.ceil(math.log(self.max_width_um / self.min_width_um)
                                 / math.log(math.sqrt(2))))
        if self.min_width_um == self.max_width_um:
            return (self.min_width_um,)
        return tuple(np.geomspace(self.min_width_um, self.max_width_um, steps + 1))


@dataclass(frozen=True)
class SpotSizeLimits:
    min_width_um: float | None = None
    max_width_um: float | None = None
    min_volume_um3: float | None = None

    def __post_init__(self):
        if self.min_volume_um3 is not None and (
                not math.isfinite(self.min_volume_um3) or self.min_volume_um3 < 0):
            raise ValueError('Minimum spot volume must be finite, nonnegative cubic micrometres.')
        for value in (self.min_width_um, self.max_width_um):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError('Spot width limits must be finite, nonnegative µm values.')
        if (self.min_width_um is not None and self.max_width_um is not None
                and self.min_width_um > self.max_width_um):
            raise ValueError('Minimum spot width must not exceed maximum spot width.')

    @property
    def active(self):
        return self.width_active or self.min_volume_um3 is not None

    @property
    def width_active(self):
        return self.min_width_um is not None or self.max_width_um is not None


@dataclass(frozen=True)
class SpotCandidate:
    focus_id: int
    nucleus_id: int
    blob: tuple
    measurement: object

    def __post_init__(self):
        object.__setattr__(self, 'blob', tuple(self.blob))
        object.__setattr__(self, 'measurement', MappingProxyType(dict(self.measurement)))

    @classmethod
    def create(cls, nucleus_id, blob, measurement):
        record = measurement.to_record()
        return cls(int(record['focus_id']), int(nucleus_id), tuple(map(float, blob)),
                   MappingProxyType(record))

    @property
    def width_um(self):
        x, y = (self.measurement.get(f'gaussian_fwhm_{axis}_um') for axis in 'xy')
        if any(v is None or not math.isfinite(v) or v <= 0 for v in (x, y)):
            return None
        return math.sqrt(x) * math.sqrt(y)

    @property
    def reliable_width(self):
        r = self.measurement
        r2 = r.get('gaussian_fit_r2')
        gaussian_flags = any(f.startswith('gaussian_') for f in (r.get('qc_flags') or '').split(';'))
        return (self.width_um is not None and r.get('gaussian_fit_status') == 'ok'
                and r2 is not None and math.isfinite(r2) and r2 >= .8
                and not gaussian_flags)


@dataclass(frozen=True)
class SpotSelection:
    limits: SpotSizeLimits
    statuses: tuple[str, ...]


def select_candidates(candidates, limits=None):
    limits = limits or SpotSizeLimits()
    if isinstance(limits, dict):
        limits = SpotSizeLimits(**limits)
    statuses = []
    for candidate in candidates:
        volume = candidate.measurement.get('watershed_volume_um3')
        if limits.min_volume_um3 is not None and (
                volume is None or not math.isfinite(volume) or volume <= 0):
            status = 'volume_unavailable'
        elif limits.min_volume_um3 is not None and volume <= limits.min_volume_um3:
            status = 'below_volume_limit'
        elif not limits.width_active:
            status = 'accepted'
        elif not candidate.reliable_width:
            status = 'size_uncertain'
        elif ((limits.min_width_um is not None
               and candidate.width_um < limits.min_width_um
               and not math.isclose(candidate.width_um, limits.min_width_um, rel_tol=1e-12))
              or (limits.max_width_um is not None
                  and candidate.width_um > limits.max_width_um
                  and not math.isclose(candidate.width_um, limits.max_width_um, rel_tol=1e-12))):
            status = 'outside_range'
        else:
            status = 'accepted'
        statuses.append(status)
    return SpotSelection(limits, tuple(statuses))


def apply_spot_selection(result, limits=None):
    """Update compatibility counts/arrays; never modify the cached candidates."""
    candidates = result.spot_candidates
    selection = select_candidates(candidates, limits)
    ids = sorted(result.nucleus_blobs)  # includes nuclei with zero candidates
    grouped = {i: [] for i in ids}
    uncertain = {i: 0 for i in ids}
    for candidate, status in zip(candidates, selection.statuses):
        if status == 'accepted':
            grouped[candidate.nucleus_id].append(candidate.blob)
        elif status in ('size_uncertain', 'volume_unavailable'):
            uncertain[candidate.nucleus_id] += 1
    result.nucleus_blobs = {i: np.asarray(b, dtype=np.float32).reshape(-1, 4)
                            for i, b in grouped.items()}
    result.nucleus_counts = {i: len(b) for i, b in grouped.items()}
    result.uncertain_counts = uncertain
    result.blobs_filtered = np.asarray([c.blob for c, status in zip(candidates, selection.statuses)
                                       if status == 'accepted'], dtype=np.float32).reshape(-1, 4)
    for i in (1, 2):
        setattr(result, f'nucleus{i}_blobs', result.nucleus_blobs.get(i, np.empty((0, 4), np.float32)))
        setattr(result, f'nucleus{i}_count', result.nucleus_counts.get(i, 0))
    result.spot_selection = selection
    return selection


def selection_warnings(result):
    diagnostics = result.spot_search_diagnostics or {}
    warnings = list(diagnostics.get('warnings', ()))
    selection = result.spot_selection
    if selection is not None and selection.limits.width_active:
        lo, hi = diagnostics.get('searched_width_um', (None, None))
        limits = selection.limits
        if lo is not None and ((limits.min_width_um is None or limits.min_width_um < lo)
                               or (limits.max_width_um is not None and limits.max_width_um < lo)):
            warnings.append('Selection extends below the searched widths; rerun with a smaller search range.')
        if hi is not None and (limits.max_width_um is None
                              or limits.max_width_um > hi
                              or (limits.min_width_um is not None and limits.min_width_um > hi)):
            warnings.append('Selection extends above the searched widths; larger spots may be missing. '
                            'Expand the search range and rerun detection.')
    return warnings


def candidate_export_records(result, include_excluded=False):
    selection = result.spot_selection or select_candidates(result.spot_candidates)
    selected = [(candidate, status) for candidate, status in zip(result.spot_candidates, selection.statuses)
                if status == 'accepted' or include_excluded]
    numbers = focus_numbers_within_nuclei(candidate.nucleus_id for candidate, _ in selected)
    for (candidate, status), number in zip(selected, numbers):
        record = watershed_export_record(candidate.measurement, status)
        record['focus_id'] = number
        yield record
