"""Floor patch registration and measurement of visible wall-floor junctions."""
import numpy as np
from scipy import ndimage, signal

FOCAL = 128 / np.tan(np.deg2rad(47))
BOOM = .46


def patches(rgb):
    rgb = rgb.astype(np.float32)
    gray = rgb @ np.array([.30, .59, .11], np.float32)
    # Unpainted, textured material, excluding bright panels, the load and robot.
    natural = (np.ptp(rgb, axis=2) < 42) & (gray > 32) & (gray < 177)
    return gray, natural


def translation(old, new, prior, radius):
    """Match independent, normalized floor patches. Return old->new image motion."""
    a, ma = old
    b, mb = new
    mean = ndimage.uniform_filter(b, 19)
    energy = np.maximum(ndimage.uniform_filter(b*b, 19)-mean*mean, .01)
    predicted = np.rint(prior).astype(int)
    shifts = []
    for y in (35, 81, 128, 174, 220):
        for x in (35, 81, 128, 174, 220):
            u, v = np.array([x, y])+predicted
            l, r = max(9, u-radius), min(246, u+radius)
            t, d = max(9, v-radius), min(246, v+radius)
            if l > r or t > d or ma[y-9:y+10, x-9:x+10].mean() < .86:
                continue
            templ = a[y-9:y+10, x-9:x+10]
            kernel = templ-templ.mean()
            std = kernel.std()
            if std < 3.8:
                continue
            corr = signal.fftconvolve(b[t-9:d+10, l-9:r+10], kernel[::-1, ::-1], mode="valid")
            corr /= (kernel.size*std)*np.sqrt(energy[t:d+1, l:r+1])
            j, i = np.unravel_index(corr.argmax(), corr.shape)
            if corr[j, i] < .68 or not mb[t+j, l+i]:
                continue
            def sub(k, values):
                if k == 0 or k == len(values)-1:
                    return 0.
                lo, mid, hi = values[k-1:k+2]
                return float(np.clip(.5*(hi-lo)/max(2*mid-lo-hi, .001), -.5, .5))
            shifts.append((l+i-x+sub(i, corr[j]), t+j-y+sub(j, corr[:, i])))
    if len(shifts) < 2:
        return None
    shifts = np.asarray(shifts)
    votes = np.linalg.norm(shifts[:, None]-shifts[None, :], axis=2) < 2.2
    group = shifts[votes[np.argmax(votes.sum(axis=1))]]
    if len(group) < 2 or len(group)*2 < len(shifts):
        return None
    return np.median(group, axis=0), len(group)


def wall_face(rgb):
    """Range to the actual skirting-floor joint; optionally the real jamb midpoint."""
    r, g, b = rgb.astype(int).transpose(2, 0, 1)
    wood = (r > g*1.29) & (g > b*1.25) & (r < 183) & (b > 23)
    bottom = wood[:-1] & ~wood[1:]
    score = bottom[:, 4:-4].sum(axis=1)
    # Central rooms are about 2.7 m across; exclude distant walls through doors.
    scale = (1.30-.145)/FOCAL
    ahead = BOOM+(127.5-np.arange(len(score)))*scale
    score[(ahead < .75) | (ahead > 1.68)] = 0
    v = int(np.argmax(score))
    if score[v] < 12:
        return None, None
    line = np.any(bottom[max(0, v-1):v+2], axis=0)
    us, = np.where(line)
    spans = np.split(us, np.flatnonzero(np.diff(us) > 5)+1)
    runs = [s for s in spans if len(s) >= 5]
    centers = []
    for left, right in zip(runs, runs[1:]):
        gap = (right[0]-left[-1])*scale
        if .88 < gap < 1.57 and left[-1] < 127 < right[0]:
            centers.append(((right[0]+left[-1])/2-127.5)*scale)
    return float(ahead[v]+scale*.5), centers[0] if len(centers) == 1 else None
