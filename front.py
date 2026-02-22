# frontend.py
# Tk UI (run this file). Auto-inits DB, ingests ./data tree, strict side + gender/age fallback, live threshold.
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

import database as db
import backend as be

APP_TITLE = "X-ray Compare – Contours first, click to Heatmap"
DATA_DIR = "data"

BODY_DISPLAY = ['chest','femur','foot','forearm','hand','humerus','shoulder','tibia and fibula','wrist']
BODY_KEYS = {'chest':'chest','femur':'femur','foot':'foot','forearm':'forearm','hand':'hand','humerus':'humerus','shoulder':'shoulder','tibia and fibula':'tibia_fibula','wrist':'wrist'}
SIDES = ['left','right','midline','unknown']
GENDERS = ['male','female','unknown']

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE); self.geometry('1360x860')
        self.state_dict = {}
        self._build_ui()
        self.after(50, self._startup)

    def _startup(self):
        try:
            db.init_db()
            stats = db.ingest_tree(DATA_DIR)
        except Exception as e:
            messagebox.showerror('DB', f'Init/Ingest failed: {e}')

    # ---------------- UI -----------------
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill='x', padx=10, pady=6)
        ttk.Button(top, text='Upload X-ray', command=self.on_upload).pack(side='left', padx=4)
        ttk.Label(top, text='Body Part').pack(side='left');
        self.body_cb = ttk.Combobox(top, values=BODY_DISPLAY, state='readonly'); self.body_cb.set('femur'); self.body_cb.pack(side='left', padx=4)
        ttk.Label(top, text='Side').pack(side='left');
        self.side_cb = ttk.Combobox(top, values=SIDES, state='readonly'); self.side_cb.set('right'); self.side_cb.pack(side='left', padx=4)
        ttk.Label(top, text='Gender').pack(side='left');
        self.gender_cb = ttk.Combobox(top, values=GENDERS, state='readonly'); self.gender_cb.set('female'); self.gender_cb.pack(side='left', padx=4)
        ttk.Label(top, text='Age').pack(side='left');
        self.age_sp = ttk.Spinbox(top, from_=0, to=120, width=5); self.age_sp.set('45'); self.age_sp.pack(side='left', padx=4)
        self.rigid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='Rigid registration', variable=self.rigid_var).pack(side='left', padx=8)
        ttk.Button(top, text='Compare', command=self.on_compare).pack(side='left', padx=8)
        ttk.Button(top, text='Export PDF', command=self.on_export_pdf).pack(side='left', padx=8)
        # View selector
        ttk.Label(top, text='View').pack(side='left', padx=(16,4))
        self.view_cb = ttk.Combobox(top, state='readonly',
            values=['Contours','Heatmap','Signature','DensityHistogram','Checkerboard','Flicker'])
        self.view_cb.set('Contours')
        self.view_cb.pack(side='left')
        self.view_cb.bind('<<ComboboxSelected>>', lambda e: self.render_view())

        thr_fr = ttk.Frame(self); thr_fr.pack(fill='x', padx=10, pady=4)
        ttk.Label(thr_fr, text='Threshold (%)').pack(side='left')
        self.thr = tk.DoubleVar(value=90)
        self.thr_scale = ttk.Scale(thr_fr, from_=0, to=100, orient='horizontal', variable=self.thr, command=self.on_threshold_change)
        self.thr_scale.pack(side='left', fill='x', expand=True, padx=8)
        self.thr_lbl = ttk.Label(thr_fr, text='90'); self.thr_lbl.pack(side='left')

        img_fr = ttk.Frame(self); img_fr.pack(fill='both', expand=True)
        self.canvas_test = tk.Canvas(img_fr, bg='black'); self.canvas_ref = tk.Canvas(img_fr, bg='black'); self.canvas_out = tk.Canvas(img_fr, bg='black')
        for c in (self.canvas_test, self.canvas_ref, self.canvas_out):
            c.configure(width=440, height=700)
        self.canvas_test.pack(side='left', fill='both', expand=True, padx=6, pady=6)
        self.canvas_ref.pack(side='left',  fill='both', expand=True, padx=6, pady=6)
        self.canvas_out.pack(side='left',  fill='both', expand=True, padx=6, pady=6)

        bot = ttk.Frame(self); bot.pack(fill='x', padx=10, pady=8)
        ttk.Button(bot, text='Zoom cursor (100×)', command=self.on_zoom_cursor).pack(side='left', padx=4)
        ttk.Button(bot, text='Download Output', command=self.on_download).pack(side='left', padx=8)
        self.status = ttk.Label(self, text='Ready'); self.status.pack(fill='x', padx=10, pady=4)

        # clicking on output toggles contour <-> heatmap
        self.canvas_out.bind('<Button-1>', self.on_output_click)

    # ----------- helpers ----------
    def _render_to_canvas(self, canvas: tk.Canvas, bgr: np.ndarray):
        if bgr is None: return
        h, w = bgr.shape[:2]
        cw = max(10, canvas.winfo_width()); ch = max(10, canvas.winfo_height())
        scale = min(cw/w, ch/h); out = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.image = imgtk
        canvas.delete('all'); canvas.create_image(cw//2, ch//2, image=imgtk)

    # ----------- actions ----------
    def on_upload(self):
        fp = filedialog.askopenfilename(title='Select X-ray', filetypes=[('Images','*.dcm *.dicom *.png *.jpg *.jpeg *.tif *.tiff *.bmp')])
        if not fp: return
        self.state_dict['test_path'] = fp
        b = be.load_image(fp)
        self.state_dict['test_raw'] = b.array
        self._render_to_canvas(self.canvas_test, cv2.cvtColor((b.array*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        self.status.config(text=f'Loaded: {fp}')

    def on_compare(self):
        if not self.state_dict.get('test_path'):
            messagebox.showwarning('Missing', 'Upload an X-ray first.'); return
        body = BODY_KEYS[self.body_cb.get()]; side = self.side_cb.get(); gender = self.gender_cb.get()
        try: age = int(self.age_sp.get())
        except Exception: age = None
        self.status.config(text='Comparing…'); self.update_idletasks()

        test = be.preprocess(self.state_dict['test_raw'])
        row = db.find_reference(body, side, gender, age)
        ref_img = None; ref_path=None
        if row is not None:
            _id, ref_path, _age = row
            ref_img = be.preprocess(be.load_image(ref_path).array)
            if self.rigid_var.get():
                test = be.rigid_register(test, ref_img)
            self._render_to_canvas(self.canvas_ref, cv2.cvtColor((ref_img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        else:
            self.canvas_ref.delete('all'); self.canvas_ref.create_text(200,200, fill='white', text='No image to compare', font=('Arial',12))
        # keep reference path for PDF export
        self.state_dict['ref_path'] = ref_path

        sig_test = be.signature_vector(be.bone_mask(test), test, n_segments=100)
        sig_ref  = be.signature_vector(be.bone_mask(ref_img), ref_img, n_segments=100) if ref_img is not None else None
        # keep signatures for advanced views
        self.state_dict['sig_test'] = sig_test
        self.state_dict['sig_ref']  = sig_ref

        if sig_ref is not None:
            s,d,t = be.combined_score(sig_test['signature'], sig_ref['signature'], sig_test['density_curve'], sig_ref['density_curve'])
        else:
            s=d=t=float('nan')

        # store for live recompute and toggle
        self.state_dict['test_img'] = test; self.state_dict['ref_img'] = ref_img
        overlay_bgr, heat_bgr, diff_pct = be.overlays_from_diff(test, ref_img if ref_img is not None else test, threshold_pct=self.thr.get())
        self.state_dict['overlay_bgr'] = overlay_bgr; self.state_dict['heat_bgr'] = heat_bgr
        self.state_dict['view'] = 'contour'
        self.state_dict['scores'] = {'Shape(0-100)':f"{s:.1f}", 'Density(0-100)':f"{d:.1f}", 'Unified(0-100)':f"{t:.1f}", 'Diff%':f"{diff_pct:.2f}"}
        self.state_dict['meta'] = {'body_part':body,'side':side,'gender':gender,'age':age,'threshold_pct':float(self.thr.get()),'rigid':self.rigid_var.get()}

        # Default to Contours view via renderer
        self.view_cb.set('Contours')
        self.render_view()
        self.status.config(text='Done (click output to toggle Heatmap)')

    def on_output_click(self, _evt=None):
        # Only toggle when in image-comparison views
        cur = self.view_cb.get()
        if cur in ('Contours','Heatmap'):
            nxt = 'Heatmap' if cur == 'Contours' else 'Contours'
            self.view_cb.set(nxt)
            self.render_view()

    def on_threshold_change(self, _evt=None):
        self.thr_lbl.config(text=f"{self.thr.get():.0f}")
        test = self.state_dict.get('test_img'); ref = self.state_dict.get('ref_img')
        if test is None or ref is None:
            return
        overlay_bgr, heat_bgr, diff_pct = be.overlays_from_diff(test, ref, threshold_pct=self.thr.get())
        self.state_dict['overlay_bgr'] = overlay_bgr
        self.state_dict['heat_bgr'] = heat_bgr
        try:
            self.state_dict['scores']['Diff%'] = f"{diff_pct:.2f}"
        except Exception:
            pass
        # re-render current view
        self.render_view()

    def on_download(self):
        if 'overlay_bgr' not in self.state_dict: return messagebox.showinfo('Nothing','Run Compare first.')
        key = 'heat_bgr' if self.state_dict.get('view','contour')=='heatmap' else 'overlay_bgr'
        fp = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
        if not fp: return
        cv2.imwrite(fp, self.state_dict[key]); messagebox.showinfo('Saved', fp)

    def on_export_pdf(self):
        if 'overlay_bgr' not in self.state_dict: return messagebox.showinfo('Nothing','Run Compare first.')
        fp = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[('PDF','*.pdf')])
        if not fp: return
        be.export_pdf(fp, self.state_dict.get('test_path'), self.state_dict.get('ref_path'),
                      self.state_dict['overlay_bgr'], self.state_dict['heat_bgr'],
                      self.state_dict.get('scores',{}), self.state_dict.get('meta',{}))
        messagebox.showinfo('Exported', fp)

    def on_zoom_cursor(self):
        if 'test_raw' not in self.state_dict:
            return messagebox.showwarning('No image','Upload an image first.')
        top = tk.Toplevel(self); top.title('100× Zoom'); lbl = ttk.Label(top); lbl.pack()
        def on_click(ev):
            img=(self.state_dict['test_raw']*255).astype(np.uint8)
            h,w=img.shape
            # simple centered crop (could map canvas coords → image coords later)
            cx=max(2,min(w-3,w//2)); cy=max(2,min(h-3,h//2))
            crop=img[cy-2:cy+2, cx-2:cx+2]
            big=cv2.resize(crop,(400,400),interpolation=cv2.INTER_NEAREST)
            imgtk=ImageTk.PhotoImage(Image.fromarray(big)); lbl.configure(image=imgtk); lbl.image=imgtk
        self.canvas_test.bind('<Button-1>', on_click)
        messagebox.showinfo('Zoom', 'Click on the left image to open a 100× zoom view.')

    def render_view(self):
        test = self.state_dict.get('test_img'); ref = self.state_dict.get('ref_img')
        if test is None:
            return
        view = self.view_cb.get()
        if view == 'Contours':
            self._render_to_canvas(self.canvas_out, self.state_dict.get('overlay_bgr'))
        elif view == 'Heatmap':
            self._render_to_canvas(self.canvas_out, self.state_dict.get('heat_bgr'))
        elif view == 'Signature':
            sig_t = self.state_dict.get('sig_test'); sig_r = self.state_dict.get('sig_ref')
            if sig_t is None:
                messagebox.showwarning('No signature','Run Compare first.'); return
            try:
                img = be.render_signature_graph(sig_t, sig_r)
            except Exception as e:
                messagebox.showerror('Signature view error', str(e)); return
            self._render_to_canvas(self.canvas_out, img)
        elif view == 'DensityHistogram':
            sig_t = self.state_dict.get('sig_test'); sig_r = self.state_dict.get('sig_ref')
            if sig_t is None:
                messagebox.showwarning('No signature','Run Compare first.'); return
            try:
                img = be.render_density_hist(sig_t['density_curve'], None if sig_r is None else sig_r['density_curve'])
            except Exception as e:
                messagebox.showerror('Histogram view error', str(e)); return
            self._render_to_canvas(self.canvas_out, img)
        elif view == 'Checkerboard':
            if ref is None:
                messagebox.showinfo('No reference','Need a reference image for Checkerboard'); return
            img = be.render_checkerboard(test, ref, tiles=8)
            self._render_to_canvas(self.canvas_out, img)
        elif view == 'Flicker':
            if ref is None:
                messagebox.showinfo('No reference','Need a reference image for Flicker'); return
            # toggle on/off
            self._flicker_on = not getattr(self, '_flicker_on', False)
            self.status.config(text=f'Flicker: {"ON" if self._flicker_on else "OFF"}')
            if self._flicker_on:
                self.after(0, self._flicker_loop)
        
    def _flicker_loop(self):
        if not getattr(self, '_flicker_on', False):
            return
        test = self.state_dict.get('test_img'); ref = self.state_dict.get('ref_img')
        if test is None or ref is None:
            self._flicker_on = False; return
        # show test then schedule ref
        self._render_to_canvas(self.canvas_out, cv2.cvtColor((test*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        self.after(120, self._flicker_step2)

    def _flicker_step2(self):
        if not getattr(self, '_flicker_on', False):
            return
        ref = self.state_dict.get('ref_img')
        self._render_to_canvas(self.canvas_out, cv2.cvtColor((ref*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        # loop back
        self.after(120, self._flicker_loop)

if __name__ == '__main__':
    App().mainloop()
