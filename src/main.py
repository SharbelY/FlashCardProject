import subprocess
import sys


if __name__ == "__main__":
    script_path = "test.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])