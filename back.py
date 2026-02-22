# backend.py
# Core imaging: load + preprocess, bone signatures, diff maps, overlays, rigid registration, PDF.
from __future__ import annotations
import os, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Image I/O (DICOM optional)
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:
    pydicom = None

import cv2
from PIL import Image
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# Data types
# =========================
@dataclass
class ImageBundle:
    array: np.ndarray  # grayscale float32 in [0,1]
    path: str


# =========================
# I/O + preprocessing
# =========================

def load_image(path: str) -> ImageBundle:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".dcm", ".dicom") and pydicom is not None:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        arr -= arr.min()
        mx = arr.max()
        if mx > 0:
            arr /= mx
        return ImageBundle(arr, path)
    else:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32)/255.0
        return ImageBundle(arr, path)


def preprocess(img: np.ndarray) -> np.ndarray:
    a8 = (img*255).astype(np.uint8)
    a8 = cv2.bilateralFilter(a8, 5, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    a8 = clahe.apply(a8)
    return a8.astype(np.float32)/255.0


# =========================
# Geometry: bone mask, centerline, normals, features
# =========================

def bone_mask(a: np.ndarray) -> np.ndarray:
    edges = cv2.Canny((a*255).astype(np.uint8), 40, 120)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
    fill = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    cnts,_ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(a, dtype=np.uint8)
    if not cnts:
        return mask
    cmax = max(cnts, key=cv2.contourArea)
    cv2.drawContours(mask, [cmax], -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return (mask>0).astype(np.uint8)


def centerline_from_mask(mask: np.ndarray, n_points: int = 100) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros((n_points,2), np.float32)
    skel = morphology.skeletonize(mask.astype(bool)).astype(np.uint8)
    ys, xs = np.nonzero(skel)
    if len(xs) < 5:
        coords = np.column_stack([xs,ys]).astype(np.float32)
        if len(coords)==0:
            return np.zeros((n_points,2), np.float32)
        mean = coords.mean(0)
        _,_,vt = np.linalg.svd(coords-mean, full_matrices=False)
        axis = vt[0]
        proj = (coords-mean)@axis
        order = np.argsort(proj)
        path = coords[order]
    else:
        coords = np.column_stack([xs,ys]).astype(np.float32)
        start = int(np.argmin(xs+ys))
        used = np.zeros(len(xs), bool); used[start]=True
        order=[start]
        for _ in range(len(xs)-1):
            last = order[-1]
            p = coords[last]
            d = np.sum((coords-p)**2, axis=1)
            d[used] = 1e9
            nxt = int(np.argmin(d)); order.append(nxt); used[nxt]=True
        path = coords[order]
    dif = np.diff(path, axis=0)
    s = np.r_[0, np.cumsum(np.hypot(dif[:,0], dif[:,1]))]
    if s[-1]==0:
        return np.repeat(path[:1], n_points, axis=0)
    sn = np.linspace(0, s[-1], n_points)
    x = np.interp(sn, s, path[:,0]); y = np.interp(sn, s, path[:,1])
    return np.column_stack([x,y]).astype(np.float32)


def normals_from_centerline(cl: np.ndarray) -> np.ndarray:
    d = np.gradient(cl, axis=0)
    t = d/(np.linalg.norm(d, axis=1, keepdims=True)+1e-6)
    n = np.column_stack([-t[:,1], t[:,0]])
    return n.astype(np.float32)


def width_profile(mask: np.ndarray, cl: np.ndarray, normals: np.ndarray, max_ray:int=200) -> np.ndarray:
    h,w = mask.shape; prof=[]
    for (x,y),(nx,ny) in zip(cl, normals):
        lp=0; ln=0
        for s in range(1, max_ray):
            xi=int(round(x+nx*s)); yi=int(round(y+ny*s))
            if xi<0 or yi<0 or xi>=w or yi>=h or mask[yi,xi]==0: lp=s-1; break
        for s in range(1, max_ray):
            xi=int(round(x-nx*s)); yi=int(round(y-ny*s))
            if xi<0 or yi<0 or xi>=w or yi>=h or mask[yi,xi]==0: ln=s-1; break
        prof.append(lp+ln)
    return np.asarray(prof, np.float32)


def curvature_and_dir(cl: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    dx=np.gradient(cl[:,0]); dy=np.gradient(cl[:,1])
    ddx=np.gradient(dx); ddy=np.gradient(dy)
    denom=(dx*dx+dy*dy)**1.5+1e-6
    kappa=(dx*ddy - dy*ddx)/denom
    theta=np.arctan2(dy,dx)
    dtheta=np.diff(theta, prepend=theta[:1]); dtheta=(dtheta+np.pi)%(2*np.pi)-np.pi
    return kappa.astype(np.float32), dtheta.astype(np.float32)


def _minmax(a: np.ndarray) -> np.ndarray:
    a=a.astype(np.float32); rng=np.ptp(a)
    return (a-a.min())/(rng+1e-6)


def signature_vector(mask: np.ndarray, img: np.ndarray, n_segments=100) -> Dict[str,np.ndarray]:
    cl=centerline_from_mask(mask, n_segments)
    nrm=normals_from_centerline(cl)
    width=width_profile(mask, cl, nrm)
    kappa,dtheta=curvature_and_dir(cl)
    # density along a thin ribbon
    h,w=img.shape; dens=[]
    for (x,y),(nx,ny) in zip(cl,nrm):
        vals=[]
        for s in range(-5,6):
            xi=int(round(x+nx*s)); yi=int(round(y+ny*s))
            if 0<=xi<w and 0<=yi<h: vals.append(img[yi,xi])
        dens.append(float(np.mean(vals)) if vals else 0.0)
    dens=np.asarray(dens, np.float32)
    width_n=_minmax((width-np.median(width))/(np.std(width)+1e-6))
    kappa_n=_minmax((kappa-np.median(kappa))/(np.std(kappa)+1e-6))
    dtheta_n=_minmax((dtheta-np.median(dtheta))/(np.std(dtheta)+1e-6))
    sig=np.concatenate([width_n,kappa_n,dtheta_n],0).astype(np.float32)
    return {"centerline":cl,"normals":nrm,"width":width,"kappa":kappa,"dtheta":dtheta,
            "density_curve":dens,"signature":sig}


# =========================
# Difference maps (edge + SSIM), overlays, registration
# =========================

def _norm01(a: np.ndarray) -> np.ndarray:
    a=a.astype(np.float32); mn=float(a.min()); mx=float(a.max())
    if mx-mn<1e-6: return np.zeros_like(a, np.float32)
    return (a-mn)/(mx-mn)

def _sobel_mag(a: np.ndarray) -> np.ndarray:
    a8=(a*255).astype(np.uint8)
    gx=cv2.Sobel(a8, cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(a8, cv2.CV_32F,0,1,ksize=3)
    return _norm01(np.sqrt(gx*gx+gy*gy))

def difference_map(test_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    e1=_sobel_mag(test_img); e2=_sobel_mag(ref_img)
    ed=np.abs(e1-e2)
    try:
        _, ssim_map = ssim(test_img, ref_img, data_range=1.0, full=True)
        sd=1.0-ssim_map.astype(np.float32)
    except Exception:
        sd=np.zeros_like(ed, np.float32)
    d=0.7*ed+0.3*sd
    d=cv2.GaussianBlur(d,(0,0),1.2)
    return _norm01(d)


def overlays_from_diff(test_img: np.ndarray, ref_img: np.ndarray, threshold_pct: float):
    diff=difference_map(test_img, ref_img)
    tval=np.percentile(diff.ravel()*100.0, threshold_pct)
    mask=((diff*100.0)>=tval).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),1)
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),1)
    base=cv2.cvtColor((test_img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay=base.copy()
    if cnts:
        cv2.drawContours(overlay, cnts, -1, (0,0,255), 2)
    heat_u8=(np.clip(diff,0,1)*255).astype(np.uint8)
    cm=cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat=cv2.addWeighted(base, 0.35, cm, 0.65, 0)
    # legend
    h,w=heat.shape[:2]; bar_w=max(30,int(0.04*w)); grad=np.linspace(255,0,h,dtype=np.uint8)[:,None]
    bar=np.repeat(grad, bar_w, 1); bar=cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    cv2.putText(bar, "More\nDifference", (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bar, "Less\nDifference", (4,h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1,cv2.LINE_AA)
    heat=np.concatenate([heat, bar], 1)
    diff_percent=float(diff.mean()*100.0)
    return overlay, heat, diff_percent


def rigid_register(mov: np.ndarray, ref: np.ndarray) -> np.ndarray:
    m8=(mov*255).astype(np.uint8); r8=(ref*255).astype(np.uint8)
    orb=cv2.ORB_create(800)
    kp1,des1=orb.detectAndCompute(m8,None); kp2,des2=orb.detectAndCompute(r8,None)
    if des1 is None or des2 is None: return mov
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches=bf.match(des1,des2)
    if len(matches)<10: return mov
    src=np.float32([kp1[m.queryIdx].pt for m in matches])
    dst=np.float32([kp2[m.trainIdx].pt for m in matches])
    M,_=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None: return mov
    h,w=ref.shape
    aligned=cv2.warpAffine(m8, M, (w,h), flags=cv2.INTER_LINEAR)
    return aligned.astype(np.float32)/255.0


def combined_score(sig_test: np.ndarray, sig_ref: np.ndarray,
                   dens_test: np.ndarray, dens_ref: np.ndarray,
                   w_shape=0.6, w_dens=0.4) -> Tuple[float,float,float]:
    """Return (shape%, density%, unified%) in 0..100, clipped.
    shape: cosine distance; density: Bhattacharyya distance.
    """
    # --- shape distance in [0,1]
    na = float(np.linalg.norm(sig_test) + 1e-9)
    nb = float(np.linalg.norm(sig_ref) + 1e-9)
    cos_sim = float(np.dot(sig_test, sig_ref) / (na * nb))  # [-1,1]
    d_shape01 = np.clip(1.0 - (cos_sim + 1.0) * 0.5, 0.0, 1.0)

    # --- density distance in [0,1] via Bhattacharyya coefficient
    ha, _ = np.histogram(np.clip(dens_test, 0, 1), bins=64, range=(0,1), density=True)
    hb, _ = np.histogram(np.clip(dens_ref,  0, 1), bins=64, range=(0,1), density=True)
    ha = ha / (ha.sum() + 1e-9)
    hb = hb / (hb.sum() + 1e-9)
    bc = float(np.sum(np.sqrt(ha * hb)))  # 0..1 (1 = identical)
    d_dens01 = np.clip(1.0 - bc, 0.0, 1.0)

    unified01 = float(np.clip(w_shape * d_shape01 + w_dens * d_dens01, 0.0, 1.0))

    shape = float(np.clip(d_shape01 * 100.0, 0.0, 100.0))
    dens  = float(np.clip(d_dens01 * 100.0, 0.0, 100.0))
    unif  = float(np.clip(unified01 * 100.0, 0.0, 100.0))
    return shape, dens, unif


# =========================
# Extra visualization renderers (Signature, DensityHistogram, Checkerboard)
# =========================

def _fig_to_bgr(fig, dpi=120):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb  = rgba[..., :3]
    bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return bgr


def render_signature_graph(sig_test: dict, sig_ref: dict | None) -> np.ndarray:
    x = np.arange(len(sig_test['width']))
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), constrained_layout=True)
    # width (z-norm)
    w_t = (sig_test['width'] - np.median(sig_test['width']))/(np.std(sig_test['width'])+1e-6)
    ax[0].plot(x, w_t, label='Test')
    if sig_ref is not None:
        w_r = (sig_ref['width'] - np.median(sig_ref['width']))/(np.std(sig_ref['width'])+1e-6)
        ax[0].plot(x, w_r, label='Ref', alpha=0.85)
    ax[0].set_title('Width (z-normalized)'); ax[0].grid(True, alpha=0.3); ax[0].legend(loc='upper right')
    # curvature
    k_t = (sig_test['kappa'] - np.median(sig_test['kappa']))/(np.std(sig_test['kappa'])+1e-6)
    ax[1].plot(x, k_t)
    if sig_ref is not None:
        k_r = (sig_ref['kappa'] - np.median(sig_ref['kappa']))/(np.std(sig_ref['kappa'])+1e-6)
        ax[1].plot(x, k_r, alpha=0.85)
    ax[1].set_title('Curvature κ'); ax[1].grid(True, alpha=0.3)
    # direction change
    d_t = (sig_test['dtheta'] - np.median(sig_test['dtheta']))/(np.std(sig_test['dtheta'])+1e-6)
    ax[2].plot(x, d_t)
    if sig_ref is not None:
        d_r = (sig_ref['dtheta'] - np.median(sig_ref['dtheta']))/(np.std(sig_ref['dtheta'])+1e-6)
        ax[2].plot(x, d_r, alpha=0.85)
    ax[2].set_title('Δθ (wrapped)'); ax[2].grid(True, alpha=0.3)
    return _fig_to_bgr(fig)


def render_density_hist(dens_test: np.ndarray, dens_ref: np.ndarray | None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(np.clip(dens_test,0,1), bins=64, range=(0,1), histtype='step', linewidth=1.5, label='Test', density=True)
    if dens_ref is not None:
        ax.hist(np.clip(dens_ref,0,1), bins=64, range=(0,1), histtype='step', linewidth=1.5, label='Ref', density=True)
    ax.set_xlabel('Normalized intensity'); ax.set_ylabel('Density')
    ax.set_title('Radiodensity Histogram (along centerline ribbon)')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
    return _fig_to_bgr(fig)


def render_checkerboard(test_img: np.ndarray, ref_img: np.ndarray, tiles=8) -> np.ndarray:
    h, w = test_img.shape
    th, tw = max(1,h//tiles), max(1,w//tiles)
    base = np.zeros((h, w), np.uint8)
    t8 = (test_img*255).astype(np.uint8)
    r8 = (ref_img*255).astype(np.uint8)
    for i in range(tiles):
        for j in range(tiles):
            ys, ye = i*th, (i+1)*th if i<tiles-1 else h
            xs, xe = j*tw, (j+1)*tw if j<tiles-1 else w
            if (i + j) % 2 == 0:
                base[ys:ye, xs:xe] = t8[ys:ye, xs:xe]
            else:
                base[ys:ye, xs:xe] = r8[ys:ye, xs:xe]
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


# =========================
# PDF export
# =========================

def export_pdf(pdf_path: str, test_path: str, ref_path: Optional[str],
               contour_bgr: np.ndarray, heat_bgr: np.ndarray,
               scores: Dict[str,Any], meta: Dict[str,Any]):
    """Create a 6-panel report: Test, Reference, Contours, Heatmap, Signature, Density."""
    # grayscale previews
    def _preview(p):
        if not p or not os.path.exists(p):
            return None
        try:
            arr = load_image(p).array
            return (np.clip(arr,0,1) * 255).astype(np.uint8)
        except Exception:
            return None

    test_preview = _preview(test_path)
    ref_preview  = _preview(ref_path) if ref_path else None

    # compute signature + histogram images if both sides are available
    sig_img = None
    hist_img = None
    try:
        if test_preview is not None and ref_preview is not None:
            t = preprocess(test_preview.astype(np.float32)/255.0)
            r = preprocess(ref_preview.astype(np.float32)/255.0)
            st = signature_vector(bone_mask(t), t, n_segments=100)
            sr = signature_vector(bone_mask(r), r, n_segments=100)
            sig_img  = render_signature_graph(st, sr)
            hist_img = render_density_hist(st['density_curve'], sr['density_curve'])
    except Exception:
        pass

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.0, 8.0))
        fig.suptitle('X-ray Comparison Report', fontsize=16)

        def axat(r, c, rs=1, cs=1):
            return plt.subplot2grid((2, 3), (r, c), rowspan=rs, colspan=cs)

        # Row 1: Test | Reference | Contours
        ax1 = axat(0,0); ax1.set_title('Test Image'); ax1.axis('off')
        if test_preview is not None:
            ax1.imshow(test_preview, cmap='gray', vmin=0, vmax=255)
        else:
            ax1.text(0.5,0.5,'(not available)', ha='center', va='center')

        ax2 = axat(0,1); ax2.set_title('Reference Image'); ax2.axis('off')
        if ref_preview is not None:
            ax2.imshow(ref_preview, cmap='gray', vmin=0, vmax=255)
        else:
            ax2.text(0.5,0.5,'(not available)', ha='center', va='center')

        ax3 = axat(0,2); ax3.set_title('Contours (red)'); ax3.axis('off')
        ax3.imshow(cv2.cvtColor(contour_bgr, cv2.COLOR_BGR2RGB))

        # Row 2: Heatmap | Signature | Density
        ax4 = axat(1,0); ax4.set_title('Heatmap'); ax4.axis('off')
        ax4.imshow(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))

        ax5 = axat(1,1); ax5.set_title('Signature'); ax5.axis('off')
        if sig_img is not None:
            ax5.imshow(cv2.cvtColor(sig_img, cv2.COLOR_BGR2RGB))
        else:
            ax5.text(0.5,0.5,'(not available)', ha='center', va='center')

        ax6 = axat(1,2); ax6.set_title('Radiodensity Histogram'); ax6.axis('off')
        if hist_img is not None:
            ax6.imshow(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB))
        else:
            ax6.text(0.5,0.5,'(not available)', ha='center', va='center')

        # Scores & settings box
        txt = '\n'.join([f'{k}: {v}' for k,v in (scores or {}).items()] + [''] +
                         [f'{k}: {v}' for k,v in (meta or {}).items()])
        fig.text(0.80, 0.05, txt, fontsize=9, family='monospace',
                 bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
