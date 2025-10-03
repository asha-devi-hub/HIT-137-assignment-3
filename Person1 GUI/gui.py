import tkinter as tk

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HIT137 Group Project - GUI (Person 1)")
        self.root.geometry("400x200")

        # Simple test label + button
        tk.Label(self.root, text="Hello! GUI is working").pack(pady=20)
        tk.Button(self.root, text="Exit", command=self.root.quit).pack()

    def run(self):
        self.root.mainloop()
