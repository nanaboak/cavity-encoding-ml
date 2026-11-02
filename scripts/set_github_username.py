"""Replace the placeholder YOUR-GITHUB-USERNAME in the project files.

Usage, from the repository folder:

    python scripts/set_github_username.py my-username
"""

import pathlib
import sys

FILES = ["README.md", "pyproject.toml", "CITATION.cff", "CONTRIBUTING.md"]


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/set_github_username.py YOUR_USERNAME")
    name = sys.argv[1]
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in FILES:
        p = root / f
        text = p.read_text(encoding="utf-8")
        p.write_text(text.replace("YOUR-GITHUB-USERNAME", name), encoding="utf-8")
        print(f"updated {f}")


if __name__ == "__main__":
    main()
