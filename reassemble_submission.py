"""
reassemble_submission.py — Reconstructs CognitiveAI_submission.zip (264.19 MB)
from the 90MB GitHub-compatible parts in submission_parts/.
"""

import os
import glob

def reassemble():
    parts = sorted(glob.glob(os.path.join("submission_parts", "CognitiveAI_submission.zip.part*")))
    if not parts:
        print("No split parts found in submission_parts/")
        return

    out_zip = "CognitiveAI_submission.zip"
    print(f"Reassembling {out_zip} from {len(parts)} parts...")
    with open(out_zip, "wb") as out_f:
        for p in parts:
            print(f"  Appending {p}...")
            with open(p, "rb") as in_f:
                out_f.write(in_f.read())

    size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"[OK] Successfully reassembled {out_zip} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    reassemble()
