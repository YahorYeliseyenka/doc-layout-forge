import sys
from pathlib import Path

import requests


def send_image(in_path: Path, out_path: Path, url: str) -> None:
    with in_path.open("rb") as f:
        files = {"image": (in_path.name, f, "image/png")}
        resp = requests.post(url, files=files, timeout=10)

    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:  # noqa: PLR2004
        print("Usage: python send_image.py input.png output.png [url]")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    endpoint = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8089/detect"  # noqa: PLR2004

    send_image(Path(in_file), Path(out_file), endpoint)
