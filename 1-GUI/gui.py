import os
import io
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from typing import Optional, Dict
from PIL import Image, ImageTk

from hf_client import HFClient
from adapters import (
    BaseAdapter,
    ModelInfo,
    TextToImageAdapter,
    ImageClassificationAdapter,
)
from mixins import ThemingMixin, FileDialogMixin
from decorators import run_in_thread, catch_errors

class TkAIApp(tk.Tk, ThemingMixin, FileDialogMixin):
    """
    OOP concepts:
      • Multiple Inheritance: Tk + ThemingMixin + FileDialogMixin
      • Encapsulation: token stored as _token with @property; setter refreshes client/adapters
      • Polymorphism & Overriding: BaseAdapter.run() overridden by task-specific adapters
      • Multiple Decorators: @run_in_thread + @catch_errors on handlers; @require_token on adapters
    """

    MODELS: Dict[str, ModelInfo] = {
        "Text-to-Image": ModelInfo(
            model_id="stabilityai/sdxl-turbo",
            task="Vision (Text-to-Image)",
            description="SDXL-Turbo: fast text-to-image. Input: prompt → Output: image."
        ),
        "Image Classification": ModelInfo(
            model_id="google/vit-base-patch16-224",
            task="Vision (Image Classification)",
            description="ViT Base (16×224) image classifier. Input: image → Output: top labels."
        ),
    }

    def __init__(self):
        super().__init__()
        self.title("Tkinter AI GUI")
        self.geometry("1000x650")
        self.minsize(820, 560)
        self.apply_theme()

        # Encapsulation
        self._token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        self._client = HFClient(self._token)
        self._last_image: Optional[Image.Image] = None

        self._build_menubar()
        self._build_layout()
        self._init_adapters()
        self._populate_explanations()
        self._load_selected_model_info()

    # ---- Encapsulation via property ----
    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, value: str):
        self._token = (value or "").strip()
        self._client = HFClient(self._token)
        for k in getattr(self, "_adapters", {}):
            self._adapters[k].client = self._client

    # ---- Menus ----
    def _build_menubar(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Set Hugging Face Token…", command=self._set_token_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        models_menu = tk.Menu(menubar, tearoff=0)
        models_menu.add_command(label="Run Model 1 (Text-to-Image)", command=self.on_run_model1)
        models_menu.add_command(label="Run Model 2 (Image Classification)", command=self.on_run_model2)
        menubar.add_cascade(label="Models", menu=models_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "Tkinter AI GUI • OOP demo using Hugging Face Inference API"))
        menubar.add_cascade(label="Help", menu=help_menu)

    # ---- Layout (grid + scrollbars, fits screen) ----
    def _build_layout(self):
        # Top: model selection
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Model Selection:").pack(side=tk.LEFT)
        self.var_model_choice = tk.StringVar(value="Text-to-Image")
        self.combo_models = ttk.Combobox(
            top, state="readonly", values=list(self.MODELS.keys()),
            textvariable=self.var_model_choice, width=24
        )
        self.combo_models.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load Model", command=self._load_selected_model_info).pack(side=tk.LEFT, padx=6)

        # Main body
        body = ttk.Frame(self, padding=(10, 5))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: input
        left = ttk.LabelFrame(body, text="User Input Section", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.rowconfigure(1, weight=1)

        self.var_input_kind = tk.StringVar(value="Text")
        radios = ttk.Frame(left)
        radios.grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(radios, text="Text", variable=self.var_input_kind, value="Text",
                        command=self._switch_input).pack(side=tk.LEFT)
        ttk.Radiobutton(radios, text="Image", variable=self.var_input_kind, value="Image",
                        command=self._switch_input).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(radios, text="Browse", command=self._browse_image).pack(side=tk.LEFT, padx=(8, 0))

        txt_frame = ttk.Frame(left)
        txt_frame.grid(row=1, column=0, sticky="nsew", pady=4)
        self.txt_input = tk.Text(txt_frame, height=6, wrap=tk.WORD)
        self.txt_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_in = ttk.Scrollbar(txt_frame, command=self.txt_input.yview)
        sb_in.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_input.config(yscrollcommand=sb_in.set)

        self.img_name_var = tk.StringVar(value="No image selected.")
        ttk.Label(left, textvariable=self.img_name_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.lbl_preview = ttk.Label(left)
        self.lbl_preview.grid(row=3, column=0, sticky="w", pady=(4, 4))

        btnrow = ttk.Frame(left)
        btnrow.grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Button(btnrow, text="Run Model 1", command=self.on_run_model1).pack(side=tk.LEFT)
        ttk.Button(btnrow, text="Run Model 2", command=self.on_run_model2).pack(side=tk.LEFT, padx=6)
        ttk.Button(btnrow, text="Clear", command=self._clear_io).pack(side=tk.LEFT)

        # Right: output
        right = ttk.LabelFrame(body, text="Model Output Section", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        out_frame = ttk.Frame(right)
        out_frame.pack(fill=tk.BOTH, expand=True)
        self.txt_output = tk.Text(out_frame, wrap=tk.WORD, height=12)
        self.txt_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_out = ttk.Scrollbar(out_frame, command=self.txt_output.yview)
        sb_out.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_output.config(yscrollcommand=sb_out.set)

        # Lower: info + OOP explanation
        lower = ttk.Frame(self, padding=8)
        lower.pack(fill=tk.BOTH, expand=False)

        info_frame = ttk.LabelFrame(lower, text="Selected Model Info", padding=8)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.txt_model_info = tk.Text(info_frame, width=40, height=8, wrap=tk.WORD)
        self.txt_model_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_info = ttk.Scrollbar(info_frame, command=self.txt_model_info.yview)
        sb_info.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_model_info.config(yscrollcommand=sb_info.set)

        oop_frame = ttk.LabelFrame(lower, text="OOP Concepts Explanation", padding=8)
        oop_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.txt_oop = tk.Text(oop_frame, width=40, height=8, wrap=tk.WORD)
        self.txt_oop.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_oop = ttk.Scrollbar(oop_frame, command=self.txt_oop.yview)
        sb_oop.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_oop.config(yscrollcommand=sb_oop.set)

        notes = ttk.LabelFrame(self, text="Notes")
        notes.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        self.txt_notes = tk.Text(notes, height=2)
        self.txt_notes.pack(fill=tk.X)

        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.var_status, anchor="e", foreground="#666").pack(
            side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6)
        )

        self._switch_input()

    # ---- Adapters ----
    def _init_adapters(self):
        self._adapters: Dict[str, BaseAdapter] = {
            "Text-to-Image": TextToImageAdapter(self._client, self.MODELS["Text-to-Image"]),
            "Image Classification": ImageClassificationAdapter(self._client, self.MODELS["Image Classification"]),
        }

    # ---- Helpers/handlers ----
    def _set_status(self, msg: str):
        self.var_status.set(msg)
        self.after(30, self.update_idletasks)

    def _clear_io(self):
        self.txt_input.delete("1.0", tk.END)
        self.txt_output.delete("1.0", tk.END)
        self.img_name_var.set("No image selected.")
        self.lbl_preview.configure(image="")
        self.lbl_preview.image = None
        self._last_image = None

    def _switch_input(self):
        self.txt_input.configure(state=("normal" if self.var_input_kind.get() == "Text" else "disabled"))

    def _browse_image(self):
        path = self.ask_image()
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 360))
            self._last_image = img
            self.img_name_var.set(os.path.basename(path))
            tkimg = ImageTk.PhotoImage(img)
            self.lbl_preview.configure(image=tkimg)
            self.lbl_preview.image = tkimg
            self._set_status("Image loaded.")
        except Exception as e:
            self._set_status(f"Image error: {e}")
            messagebox.showerror("Image Error", str(e))

    def _set_token_dialog(self):
        value = simpledialog.askstring("Hugging Face Token",
                                       "Paste your Hugging Face Access Token:",
                                       show="•", initialvalue=self.token)
        if value is not None:
            self.token = value
            self._set_status("Token saved for this session.")

    def _load_selected_model_info(self):
        key = self.var_model_choice.get()
        self.txt_model_info.delete("1.0", tk.END)
        self.txt_model_info.insert("1.0", self._adapters[key].info_text())
        self._set_status(f"Loaded: {key}")

    def _populate_explanations(self):
        text = (
            "• Multiple Inheritance:\n"
            "  TkAIApp(tk.Tk, ThemingMixin, FileDialogMixin)\n\n"
            "• Encapsulation:\n"
            "  Token stored in _token with @property; setter refreshes HF client + adapters\n\n"
            "• Polymorphism & Overriding:\n"
            "  BaseAdapter.run() overridden in TextToImageAdapter/ImageClassificationAdapter\n\n"
            "• Multiple Decorators:\n"
            "  @run_in_thread & @catch_errors on UI handlers; @require_token on adapters\n"
        )
        self.txt_oop.delete("1.0", tk.END)
        self.txt_oop.insert("1.0", text)

    # ---- Run buttons ----
    @catch_errors
    @run_in_thread
    def on_run_model1(self):
        self._set_status("Running Text-to-Image…")
        adapter = self._adapters["Text-to-Image"]
        prompt = self.txt_input.get("1.0", tk.END).strip() or "A cozy reading nook with plants"
        img = adapter.run(prompt=prompt)
        tkimg = ImageTk.PhotoImage(img.resize((min(512, img.width), min(384, img.height))))
        self.lbl_preview.configure(image=tkimg)
        self.lbl_preview.image = tkimg
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert("1.0", "Text-to-Image completed. Preview updated.")
        self._set_status("Done (Model 1).")

    @catch_errors
    @run_in_thread
    def on_run_model2(self):
        self._set_status("Running Image Classification…")
        adapter = self._adapters["Image Classification"]
        if self._last_image is None:
            raise RuntimeError("Please select/load an image (Browse).")
        buf = io.BytesIO()
        self._last_image.save(buf, format="PNG")
        result = adapter.run(image_bytes=buf.getvalue())
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert("1.0", result)
        self._set_status("Done (Model 2).")

if __name__ == "__main__":
    TkAIApp().mainloop()
