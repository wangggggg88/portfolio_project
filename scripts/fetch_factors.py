"""Download the two Fama-French daily factor files used by src/run_factors.py.

Source: Kenneth R. French Data Library (Tuck School of Business, Dartmouth).
Files are saved unmodified under data/factors/raw/ with a manifest of URL,
download time and SHA-256. Run manually; analysis never downloads.

    python scripts/fetch_factors.py
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = ["F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"]


def main():
    out = ROOT / "data" / "factors" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Kenneth R. French Data Library",
        "library_page": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "use": "Local personal educational research; cite the Data Library; no redistribution license asserted",
        "files": [],
    }
    for name in FILES:
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "portfolio_project research script"})
        data = urllib.request.urlopen(req, timeout=60).read()
        (out / name).write_bytes(data)
        manifest["files"].append(
            {
                "file": name,
                "url": BASE + name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        print(f"saved {name} ({len(data):,} bytes)")
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
