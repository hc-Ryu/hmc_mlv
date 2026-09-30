---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
Cell In[2], line 944
    924 weights = {
    925     'w_phys':      10.0,
    926     'w_smooth':    0.1,
    927     'w_mass':      1.0,
    928     'w_collision': 10.0,
    929 }
    931 history, base_coords, result_coords, part_labels = run_training(
    932     data,
    933     target_mps=target_mps,
   (...)
    941     snapshot_interval=10,
    942 )
--> 944 visualize_training(history, base_coords, result_coords, TARGET_MP, part_labels=part_labels)
    946 final_pred = float(np.sum(history['pred_mp'][-1]))
    947 final_err  = abs(final_pred - TARGET_MP) / TARGET_MP * 100

Cell In[2], line 896
    894 plt.tight_layout()
    895 out_path = 'uni-section/uni_section_v4_result.png'
--> 896 plt.savefig(out_path, dpi=120, bbox_inches='tight')
    897 plt.show()
    898 print(f"\n결과 저장: {out_path}")

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\pyplot.py:1250, in savefig(*args, **kwargs)
   1247 fig = gcf()
   1248 # savefig default implementation has no return, so mypy is unhappy
   1249 # presumably this is here because subclasses can return?
-> 1250 res = fig.savefig(*args, **kwargs)  # type: ignore[func-returns-value]
   1251 fig.canvas.draw_idle()  # Need this if 'transparent=True', to reset colors.
   1252 return res

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\figure.py:3490, in Figure.savefig(self, fname, transparent, **kwargs)
   3488     for ax in self.axes:
   3489         _recursively_make_axes_transparent(stack, ax)
-> 3490 self.canvas.print_figure(fname, **kwargs)

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\backend_bases.py:2186, in FigureCanvasBase.print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, pad_inches, bbox_extra_artists, backend, **kwargs)
   2182 try:
   2183     # _get_renderer may change the figure dpi (as vector formats
   2184     # force the figure dpi to 72), so we need to set it again here.
   2185     with cbook._setattr_cm(self.figure, dpi=dpi):
-> 2186         result = print_method(
   2187             filename,
   2188             facecolor=facecolor,
   2189             edgecolor=edgecolor,
   2190             orientation=orientation,
   2191             bbox_inches_restore=_bbox_inches_restore,
   2192             **kwargs)
   2193 finally:
   2194     if bbox_inches and restore_bbox:

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\backend_bases.py:2042, in FigureCanvasBase._switch_canvas_and_return_print_method.<locals>.<lambda>(*args, **kwargs)
   2038     optional_kws = {  # Passed by print_figure for other renderers.
   2039         "dpi", "facecolor", "edgecolor", "orientation",
   2040         "bbox_inches_restore"}
   2041     skip = optional_kws - {*inspect.signature(meth).parameters}
-> 2042     print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
   2043         *args, **{k: v for k, v in kwargs.items() if k not in skip}))
   2044 else:  # Let third-parties do as they see fit.
   2045     print_method = meth

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\backends\backend_agg.py:481, in FigureCanvasAgg.print_png(self, filename_or_obj, metadata, pil_kwargs)
    434 def print_png(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
    435     """
    436     Write the figure to a PNG file.
    437 
   (...)
    479         *metadata*, including the default 'Software' key.
    480     """
--> 481     self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\backends\backend_agg.py:430, in FigureCanvasAgg._print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata)
    425 """
    426 Draw the canvas, then save it using `.image.imsave` (to which
    427 *pil_kwargs* and *metadata* are forwarded).
    428 """
    429 FigureCanvasAgg.draw(self)
--> 430 mpl.image.imsave(
    431     filename_or_obj, self.buffer_rgba(), format=fmt, origin="upper",
    432     dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\matplotlib\image.py:1657, in imsave(fname, arr, vmin, vmax, cmap, format, origin, dpi, metadata, pil_kwargs)
   1655 pil_kwargs.setdefault("format", format)
   1656 pil_kwargs.setdefault("dpi", (dpi, dpi))
-> 1657 image.save(fname, **pil_kwargs)

File c:\Users\user\anaconda3\envs\pytorch\Lib\site-packages\PIL\Image.py:2566, in Image.save(self, fp, format, **params)
   2564         fp = builtins.open(filename, "r+b")
   2565     else:
-> 2566         fp = builtins.open(filename, "w+b")
   2567 else:
   2568     fp = cast(IO[bytes], fp)

FileNotFoundError: [Errno 2] No such file or directory: 'uni-section/uni_section_v4_result.png'