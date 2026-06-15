from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wound_rt.data.bootstrap_demo_data import main


if __name__ == "__main__":
    main()
