"""Render MRI and CT comparisons on the native image grid."""
from pathlib import Path
import numpy as np
from rsam.data import load_image


def visualize(mr_path, prediction_path, output, slice_index, ct_path=None, ct_window=(-1000, 2000)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if ct_window[0] >= ct_window[1]:
        raise ValueError('CT display minimum must be smaller than maximum')
    image, mr = load_image(mr_path, {'orientation': 'native'})
    pi, prediction = load_image(prediction_path, {'orientation': 'native'})
    def aligned(other):
        if other.shape != image.shape or not np.allclose(other.affine, image.affine, atol=1e-4, rtol=0):
            raise ValueError('Comparison requires matching physical grids')
    aligned(pi)
    if not 0 <= slice_index < mr.shape[2]:
        raise ValueError('Slice index is outside the volume')
    values = mr[mr != 0]
    lo, hi = np.percentile(values if values.size else mr, [.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    panels = [('MRI', mr, 'gray', lo, hi), ('Synthetic CT', prediction, 'gray', *ct_window)]
    if ct_path:
        ci, ct = load_image(ct_path, {'orientation': 'native'})
        aligned(ci)
        panels.extend([('Reference CT', ct, 'gray', *ct_window),
                       ('Absolute error (HU)', np.abs(prediction-ct), 'magma', 0, 500)])
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    aspect = float(image.header.get_zooms()[1] / image.header.get_zooms()[0])
    for ax, (title, volume, cmap, vmin, vmax) in zip(axes, panels):
        artist = ax.imshow(volume[:, :, slice_index].T, origin='lower', cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation='nearest', aspect=aspect)
        ax.set_title(title);ax.set_axis_off()
        if title != 'MRI':
            fig.colorbar(artist, ax=ax, shrink=.65, label='HU')
    fig.suptitle(f'Native slice {slice_index}; CT window [{ct_window[0]}, {ct_window[1]}] HU')
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
