def on_threshold_change(self, _evt=None):
        self.thr_lbl.config(text=f"{self.thr.get():.0f}")
        test=self.state_dict.get('test_img'); ref=self.state_dict.get('ref_img')
        if test is None or ref is None: return
        overlay_bgr, heat_bgr, diff_pct = be.overlays_from_diff(test, ref, threshold_pct=self.thr.get())
        self.state_dict['overlay_bgr']=overlay_bgr; self.state_dict['heat_bgr']=heat_bgr; self.state_dict['scores']['Diff%']=f"{diff_pct:.2f}"
        # re-render current view
        self.render_view():
        self.thr_lbl.config(text=f"{self.thr.get():.0f}")
        test=self.state_dict.get('test_img'); ref=self.state_dict.get('ref_img')
        if test is None or ref is None: return
        overlay_bgr, heat_bgr, diff_pct = be.overlays_from_diff(test, ref, threshold_pct=self.thr.get())
        self.state_dict['overlay_bgr']=overlay_bgr; self.state_dict['heat_bgr']=heat_bgr; self.state_dict['scores']['Diff%']=f"{diff_pct:.2f}"
        key = 'heat_bgr' if self.state_dict.get('view','contour')=='heatmap' else 'overlay_bgr'
        self._render_to_canvas(self.canvas_out, self.state_dict[key])

