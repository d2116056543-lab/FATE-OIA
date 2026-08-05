from __future__ import annotations

import subprocess
import sys


def main() -> None:
    # Intentionally foreground: no Start-Process, nohup, daemon or hidden child process.
    raise SystemExit(subprocess.call([sys.executable, "-m", "fate_oia.engine.train_lens_oia", *sys.argv[1:]]))


if __name__=="__main__": main()
