import tkinter as tk
from gui import DigitalSignalAnalyzerGUI

def main():
    root = tk.Tk()
    app = DigitalSignalAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
