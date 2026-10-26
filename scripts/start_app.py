"""Do not accept user sessions until the bundled model passes verification."""

import os
from pathlib import Path
import sys

from verify_model import verify_model


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    os.environ["LASTRADA_SAMPLEID_CHECKPOINT"] = str(verify_model())
    os.environ["LASTRADA_REQUIRE_PREINSTALLED_MODEL"] = "1"
    # exec releases preflight model memory before Streamlit loads user analyses.
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", "simple_app.py",
        "--server.address=0.0.0.0", f"--server.port={os.environ.get('PORT', '8080')}",
    ])


if __name__ == "__main__":
    main()
