import time
import numpy as np

# 50th percentile: 2.3056132793426514
# 90th percentile: 3.40212574005127
# 95th percentile: 4.306429862976074
percentiles = {50: 2.3056132793426514, 90: 3.40212574005127, 95: 4.306429862976074}

def tempo_to_b(tempo):
    tempo = np.array(tempo)
    b_f = np.clip(tempo, percentiles[50], percentiles[95]) - percentiles[50]
    b_f = b_f / (percentiles[95] - percentiles[50]) * 100
    return b_f

class Logger:
    def __init__(self, quiet: bool = False, debug: bool = False):
        self.quiet = quiet
        self.debug = debug

    def log(self, msg: str, level: str = "INFO", err=None):
        """Central logger. Suppresses output when quiet=True or non-debug messages
        when debug=False and level=="DEBUG"."""
        if not self.quiet or level in ["ERROR", "WARN"]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}][{level}] {msg}")
        if err and self.debug:
            raise err
