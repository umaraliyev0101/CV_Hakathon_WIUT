import argparse, sys

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pulse")
    p.add_subparsers(dest="cmd")
    args = p.parse_args(argv)
    p.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
