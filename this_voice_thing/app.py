import ctypes
import datetime
import os
import runpy
import sys
import traceback

from this_voice_thing import paths


def make_startup_log_path():
    logs_dir = paths.LOGS_DIR
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
        "This Voice Thing failed to start.\n\n"
        f"Log file:\n{log_path}\n\n"
        "Last error:\n"
        f"{exc_text}"
    )
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "This Voice Thing: Startup Error", 0x10)
    except Exception:
        pass


def main():
    with open(LOG_PATH, "a", encoding="utf-8") as log_handle:
        sys.stdout = TeeFile(sys.__stdout__, log_handle)
        sys.stderr = TeeFile(sys.__stderr__, log_handle)

        print(f"Launching This Voice Thing at {datetime.datetime.now().isoformat()}")
        print(f"Python executable: {sys.executable}")
        print(f"Working directory: {os.getcwd()}")

        try:
            runpy.run_module("this_voice_thing.ui.main_window", run_name="__main__", alter_sys=True)
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            show_startup_error(LOG_PATH, tb.splitlines()[-1] if tb.splitlines() else "Unknown startup error")
            raise


if __name__ == "__main__":
    main()
