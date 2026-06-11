import ctypes
import datetime
import os
import runpy
import sys
import traceback


def make_startup_log_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(logs_dir, f"app_startup_{timestamp}.log")


LOG_PATH = make_startup_log_path()


class TeeFile:
    def __init__(self, original_stream, log_handle):
        self.original_stream = original_stream
        self.log_handle = log_handle
        self.encoding = getattr(original_stream, "encoding", "utf-8")

    def write(self, value):
        text = str(value)
        self.log_handle.write(text)
        self.log_handle.flush()
        if self.original_stream is not None:
            try:
                self.original_stream.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        self.log_handle.flush()
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except Exception:
                pass

    def isatty(self):
        if self.original_stream is None:
            return False
        return bool(getattr(self.original_stream, "isatty", lambda: False)())


def show_startup_error(log_path, exc_text):
    message = (
        "Chatterbox TTS UI failed to start.\n\n"
        f"Log file:\n{log_path}\n\n"
        "Last error:\n"
        f"{exc_text}"
    )
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Chatterbox TTS UI Startup Error", 0x10)
    except Exception:
        pass


def main():
    with open(LOG_PATH, "a", encoding="utf-8") as log_handle:
        sys.stdout = TeeFile(sys.__stdout__, log_handle)
        sys.stderr = TeeFile(sys.__stderr__, log_handle)

        print(f"Launching main.py at {datetime.datetime.now().isoformat()}")
        print(f"Python executable: {sys.executable}")
        print(f"Working directory: {os.getcwd()}")

        try:
            runpy.run_path(os.path.join(os.path.dirname(__file__), "main.py"), run_name="__main__")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            show_startup_error(LOG_PATH, tb.splitlines()[-1] if tb.splitlines() else "Unknown startup error")
            raise


if __name__ == "__main__":
    main()
