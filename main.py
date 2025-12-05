import tkinter as tk
from gui import MainGUI

# 설정값
VIDEO_ROOT = "raw_videos"
SAVE_ROOT = "analyzed"
CLASS_MAP = {
    "neutral": 0,
    "movement": 1,
    "threat": 2
}

def main():
    root = tk.Tk()

    app = MainGUI(
        root=root,
        video_root=VIDEO_ROOT,
        save_root=SAVE_ROOT,
        class_map=CLASS_MAP
    )

    root.mainloop()

if __name__ == "__main__":
    main()