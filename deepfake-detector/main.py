"""Main entry point CLI wrapper for Deepfake Detection pipeline."""
import os
import sys
import argparse

# Insert root to system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.prepare_dataset import run_prepare

def main() -> None:
    """Configures subcommand parser and triggers script endpoints."""
    parser = argparse.ArgumentParser(
        description="Hybrid Deepfake Detection using Transfer Learning CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Workflow commands")
    
    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", 
        help="Preprocess raw videos into face crop datasets"
    )
    prepare_parser.add_argument(
        "--config-dir", 
        type=str, 
        default="config",
        help="Path to configuration split directory (default: config)"
    )
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        try:
            run_prepare(config_dir=args.config_dir)
        except Exception:
            sys.exit(1)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Error: Unknown command '{args.command}'")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
