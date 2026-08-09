# logger.py
from __future__ import annotations
import sys, atexit, traceback, os, stat
from typing import Optional, TextIO

class Logger:
    def __init__(self, suppress: bool = False):
        self.msgs: list[str] = []
        self.suppress = suppress
        self._tee_fd: Optional[TextIO] = None
        self._tee_path: Optional[str] = None
        atexit.register(self.close)
        
    def __getstate__(self):
        # Copy state and remove unpicklable members
        state = self.__dict__.copy()
        state['_tee_fd'] = None   # don't try to pickle an open file handle
        return state

    def __setstate__(self, state):
        # Restore; _tee_fd stays None in worker processes
        self.__dict__.update(state)

    # ---- one-line API you can call from summit.py once you know --out ----
    def attach_file(self, path: str, mode: str = "a"):
        """Start teeing all future logs to file, and dump the backlog immediately."""
        if mode not in {"a", "w"}:
            raise ValueError("Logger file mode must be 'a' or 'w'.")
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if mode == "a":
            flags |= os.O_APPEND
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Secure log creation requires O_NOFOLLOW support.")
        flags |= nofollow
        descriptor = os.open(path, flags, 0o600)
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError(f"Log target must be a regular file: {path}.")
            if observed.st_uid != os.geteuid() or observed.st_nlink != 1:
                raise PermissionError(
                    f"Log target must be an owner-controlled, singly linked file: {path}."
                )
            os.fchmod(descriptor, 0o600)
            if mode == "w":
                os.ftruncate(descriptor, 0)
            new_tee = os.fdopen(descriptor, mode, buffering=1)
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if self._tee_fd:
            try: self._tee_fd.close()
            except Exception: pass
        self._tee_path = path
        self._tee_fd = new_tee
        # replay backlog
        for line in self.msgs:
            try:
                self._tee_fd.write(line)
            except Exception:
                pass
        try: self._tee_fd.flush()
        except Exception: pass

    def install_excepthook(self):
        """Write uncaught exceptions (with traceback) to the log file and terminal."""
        prev = sys.excepthook
        def _hook(exc_type, exc, tb):
            tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
            self._log("[fatal] Uncaught exception:\n" + tb_text.rstrip())
            try: self._tee_fd.flush() if self._tee_fd else None
            except Exception: pass
            # still show the normal traceback
            try:
                (prev or sys.__excepthook__)(exc_type, exc, tb)
            except Exception:
                pass
        sys.excepthook = _hook

    # ---- existing API, unchanged in your other modules ----
    def _log(self, msg: str, end: str = "\n"):
        line = msg + end
        self.msgs.append(line)
        if not self.suppress:
            print(line, end="")  # same behavior as before
        if self._tee_fd is not None:
            try:
                self._tee_fd.write(line); self._tee_fd.flush()
            except Exception:
                pass

    def _save_log(self, path: str):
        with open(path, "w") as fd:
            for msg in self.msgs:
                fd.write(msg)

    def close(self):
        try:
            if self._tee_fd is not None:
                self._tee_fd.flush()
                self._tee_fd.close()
        finally:
            self._tee_fd = None
