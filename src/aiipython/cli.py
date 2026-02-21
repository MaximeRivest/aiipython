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
    parser.add_argument(
        "--ui",
        default=None,
        choices=["pi-native", "pi-tui", "textual"],
        help="Frontend UI: pi-native (default), pi-tui, or textual (legacy).",
    )
    parser.add_argument(
        "--lm-backend",
        default=None,
        choices=["auto", "pi", "litellm"],
        help="LM backend routing: auto (default), pi gateway, or litellm.",
    )
    args = parser.parse_args()

    from aiipython import chat
    chat(model=args.model, ui=args.ui, lm_backend=args.lm_backend)


if __name__ == "__main__":
    main()
