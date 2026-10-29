"""ASCII startup progress that stays active between loading callbacks."""

import shutil
import threading


class StartupProgress:
    def __init__(self, console):
        self.console = console
        self.width = max(32, shutil.get_terminal_size((120, 20)).columns - 1)
        self.animated = console.isatty()
        self.percent = 0
        self.label = "Loading"
        self.frame = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _draw(self):
        filled = round(24 * self.percent / 100)
        bar = "#" * filled + "-" * (24 - filled)
        dots = (".  ", ".. ", "...", " ..", "  .", "   ")[self.frame % 6]
        marker = f"[{dots}] " if self.animated and self.percent < 100 else ""
        line = f"{marker}[{bar}] {self.percent:3d}%  {self.label}"
        if self.animated:
            self.console.write("\r" + line[:self.width].ljust(self.width))
        else:
            self.console.write(line + "\n")
        self.console.flush()

    def __call__(self, percent, label):
        with self._lock:
            self.percent = min(100, max(0, int(percent)))
            self.label = label
            self._draw()

    def _animate(self):
        while not self._stop.wait(0.2):
            with self._lock:
                self.frame += 1
                self._draw()

    def __enter__(self):
        if self.animated:
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if exc_type is not None and self.animated:
            self.console.write("\n")
            self.console.flush()
