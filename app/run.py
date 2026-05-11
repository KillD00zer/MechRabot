#!/usr/bin/env python3
import subprocess
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run the MechRabot project via Modal")
    parser.add_argument(
        "query", 
        nargs="?", 
        default="How do I replace the timing belt?", 
        help="The query to ask MechRabot"
    )
    parser.add_argument("--serve", action="store_true", help="Serve the FastAPI endpoint")
    parser.add_argument("--deploy", action="store_true", help="Deploy the app to Modal")
    
    args = parser.parse_args()

    # Find the project root (one directory up from the 'app' folder)
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    
    # Path relative to project root
    modal_app_path = os.path.join("app", "modal_app.py")

    if args.serve:
        print("🌐 Serving MechRabot endpoint on Modal...")
        cmd = ["modal", "serve", modal_app_path]
    elif args.deploy:
        print("🚀 Deploying MechRabot to Modal...")
        cmd = ["modal", "deploy", modal_app_path]
    else:
        print(f"🤖 Running MechRabot query: '{args.query}'")
        cmd = ["modal", "run", modal_app_path, "--text", args.query]

    print(f"Executing: {' '.join(cmd)}\n")
    
    try:
        # Run from project root so .add_local_python_source("app") in modal_app.py works correctly
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
