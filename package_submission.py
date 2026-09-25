"""
package_submission.py — Packages the final submission zip archive according to hackathon specifications.

Directory structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_submission_zip(team_name="CognitiveAI", zip_out=None):
    if zip_out is None:
        zip_out = os.path.join(RESOURCE_DIR, f"{team_name}_submission.zip")

    output_dir = os.path.join(RESOURCE_DIR, "output")
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    code_dir = os.path.join(RESOURCE_DIR, "code", "business_entity_resolution")
    doc_file = os.path.join(RESOURCE_DIR, "Documentation_template.md")

    # Verification checks
    missing = []
    if not os.path.exists(matching_file):
        missing.append(matching_file)
    if not os.path.exists(candidate_file):
        missing.append(candidate_file)
    if not os.path.exists(code_dir):
        missing.append(code_dir)
    if not os.path.exists(doc_file):
        missing.append(doc_file)

    if missing:
        print("ERROR: Missing required submission files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"Creating submission package: {zip_out}...")
    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # 1. output/
        z.write(matching_file, "output/matching_results.tsv")
        z.write(candidate_file, "output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        for root, dirs, files in os.walk(code_dir):
            for f in files:
                if f.endswith(('.py', '.md', '.txt')) and not f.startswith('.'):
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, RESOURCE_DIR)
                    z.write(full_p, rel_p.replace('\\', '/'))

        # 3. Documentation_template.md
        z.write(doc_file, "Documentation_template.md")

    print(f"[OK] Successfully created {zip_out} ({os.path.getsize(zip_out) / (1024*1024):.2f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=str, default='CognitiveAI', help='Team name')
    parser.add_argument('--out', type=str, default=None, help='Output zip path')
    args = parser.parse_args()
    create_submission_zip(team_name=args.team, zip_out=args.out)
