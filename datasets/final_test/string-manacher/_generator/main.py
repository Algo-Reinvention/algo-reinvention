from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.generate_final_test_cases import run_single_algorithm_cli


if __name__ == "__main__":
    raise SystemExit(run_single_algorithm_cli(Path(__file__).resolve().parent))
