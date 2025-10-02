# mixins.py
from tkinter import ttk, filedialog
from typing import Optional

class ThemingMixin:
    def apply_theme(self):
        try:
            style = ttk.Style(self)
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass

class FileDialogMixin:
    def ask_image(self) -> Optional[str]:
        return filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )
