import os
import sys
import subprocess
import shutil
import argparse

VENV_DIR = ".venv"

def run(cmd, **kwargs):
    print(f"▶️ Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)

def create_venv(force=False):
    if force and os.path.exists(VENV_DIR):
        print("🗑 Removing existing virtual environment...")
        shutil.rmtree(VENV_DIR)

    if not os.path.exists(VENV_DIR):
        print("📦 Creating virtual environment...")
        run([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print(f"✅ Virtual environment '{VENV_DIR}' already exists.")

def install_deps(upgrade=False):
    pip = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "pip")

    if upgrade:
        print("⬆️ Upgrading pip...")
        run([pip, "install", "--upgrade", "pip"])

    print("📦 Installing dependencies...")
    run([pip, "install", "-r", "requirements.txt"])

def main():
    parser = argparse.ArgumentParser(description="Project environment setup")
    parser.add_argument("--force", action="store_true", help="Recreate the virtual environment")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade pip and packages")
    args = parser.parse_args()

    create_venv(force=args.force)
    install_deps(upgrade=args.upgrade)

    print("\n🎉 Setup complete.")
    print("➡️ To activate the virtual environment, run:")
    if os.name == "nt":
        print(r".venv\Scripts\activate")
    else:
        print("source .venv/bin/activate")

if __name__ == "__main__":
    main()
