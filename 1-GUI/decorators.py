# decorators.py
import threading
from typing import Callable
from tkinter import messagebox

def run_in_thread(fn: Callable) -> Callable:
    """Run a blocking handler in a daemon thread so the UI remains responsive."""
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start()
    return wrapper

def catch_errors(fn: Callable) -> Callable:
    """Catch exceptions, toast a dialog, and write to the status line when available."""
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, "_set_status"):
                self._set_status(f"Error: {e}")
            messagebox.showerror("Error", str(e))
    return wrapper

def require_token(fn):
    def wrapper(self, *args, **kwargs):
        # allow empty token but warn in the status line
        if not getattr(self, "token", "") and hasattr(self, "_set_status"):
            self._set_status("Warning: no HF token set; request may be rate-limited or fail.")
        return fn(self, *args, **kwargs)
    return wrapper

