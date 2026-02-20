"""CLI entry point for aiipython."""

import argparse

from aiipython import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aiipython",
        description="A reactive AI chat assistant running inside IPython.",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=(
            "LM model string. If omitted: remembered last model > "
            "AIIPYTHON_MODEL env > gemini/gemini-3-flash-preview"
        ),
    )
    args = parser.parse_args()

    from aiipython import chat
    chat(model=args.model)


if __name__ == "__main__":
    main()
