import time
import threading
import signal
from app import app  # your Flask/FastAPI app

START_TIME = time.time()
STOP = threading.Event()

def uptime_logger(interval_sec: int = 5):
    """Periodically log total uptime until STOP is set."""
    while not STOP.wait(interval_sec):
        elapsed = time.time() - START_TIME
        print(f"[uptime] {elapsed:,.1f}s")

def _graceful_stop(*_):
    STOP.set()

if __name__ == "__main__":
    # Handle Ctrl+C / container stop -> stop the logger cleanly
    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    # Start the logger thread
    t = threading.Thread(target=uptime_logger, kwargs={"interval_sec": 5}, daemon=True)
    t.start()

    # IMPORTANT for Flask: avoid double-spawning the thread due to the reloader
    # If you're on Flask, pass use_reloader=False; FastAPI/Uvicorn doesn't need this line.
    try:
        app.run(use_reloader=False)  # remove/use_reloader depending on your server
    finally:
        STOP.set()
        t.join(timeout=1)
