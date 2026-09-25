"""
deep_error_analysis.py — Identify exact failure modes & quantify improvement potential.
"""

import sys, os, time
from collections import defaultdict, Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import config
from normalizer import normalize_business_name, normalize_address, extract_postal_code, extract_street_numbers
from blocking import CandidateBlocker

print("="*75)
print("DEEP ERROR ANALYSIS — Finding Exactly Where We Lose Points")
print("="*75)

# Load 3000 GT rows
gt = {}
needed_ids = set()
with open(config.TRAIN_GT, encoding='utf-8') as f:
    f.readline()
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        matches = [m.strip() for m in parts[1].split(',') if m.strip()] if len(parts) > 1 else []
        gt[parts[0]] = matches
        needed_ids.update(matches)
        if len(gt) >= 3000:
            break

# Load S1
s1 = {}
with open(config.TRAIN_S1, encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in gt:
            s1[p[0]] = {'name': p[1] if len(p)>1 else '', 'addr': p[2] if len(p)>2 else '', 'country': p[3] if len(p)>3 else 'US'}

# Load S2/S3 raw records (target matches only)
cand_raw = {}
for path in [config.TRAIN_S2, config.TRAIN_S3]:
    with open(path, encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.strip().split('\t')
            if p[0] in needed_ids or i < 100000:
                cand_raw[p[0]] = {'name': p[1] if len(p)>1 else '', 'addr': p[2] if len(p)>2 else '', 'country': p[3] if len(p)>3 else 'US'}

# Build blocker with same candidates
blocker = CandidateBlocker(max_candidates=25)
for eid, rec in cand_raw.items():
    blocker.add_candidate(eid, rec['name'], rec['addr'], rec['country'])

print(f"Loaded {len(gt)} GT queries, {len(cand_raw)} candidate records")

# Classify failures
failure_cats = Counter()
detail_examples = defaultdict(list)

def has_non_latin(text):
    """Check for non-Latin scripts (Hindi, Tamil, Gujarati, etc.)"""
    for ch in text:
        cp = ord(ch)
        if cp > 0x024F and cp < 0xFFFF and not (0x2000 <= cp <= 0x206F):
            return True
    return False

def detect_script(text):
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F: return 'Devanagari'
        if 0x0B80 <= cp <= 0x0BFF: return 'Tamil'
        if 0x0A80 <= cp <= 0x0AFF: return 'Gujarati'
        if 0x0A00 <= cp <= 0x0A7F: return 'Gurmukhi'
        if 0x0980 <= cp <= 0x09FF: return 'Bengali'
        if 0x0C00 <= cp <= 0x0C7F: return 'Telugu'
        if 0x0C80 <= cp <= 0x0CFF: return 'Kannada'
        if 0x0D00 <= cp <= 0x0D7F: return 'Malayalam'
    return None

blocking_miss_total = 0
blocking_miss_reasons = Counter()
total_true = 0

for s1_id, true_matches in gt.items():
    s1_rec = s1.get(s1_id)
    if not s1_rec or not true_matches:
        continue

    present_matches = [m for m in true_matches if m in cand_raw]
    if not present_matches:
        continue

    total_true += len(present_matches)
    cands = blocker.retrieve_candidates(s1_id, s1_rec['name'], s1_rec['addr'], s1_rec['country'])
    cand_set = set(cands)

    for m_id in present_matches:
        if m_id not in cand_set:
            blocking_miss_total += 1
            m_rec = cand_raw[m_id]

            # Categorize the miss
            s1_name_norm = normalize_business_name(s1_rec['name'])
            m_name_raw = m_rec['name']

            # 1. Non-Latin script in candidate name
            script = detect_script(m_name_raw)
            if script and script != 'Devanagari':
                blocking_miss_reasons[f'Non-Latin Script ({script})'] += 1
                if len(detail_examples[f'script_{script}']) < 2:
                    detail_examples[f'script_{script}'].append((s1_rec['name'], m_name_raw))
                continue

            # 2. Empty/missing address in candidate
            if not m_rec['addr'].strip() or m_rec['addr'].strip() == 'null':
                blocking_miss_reasons['Empty/Null Address in Candidate'] += 1
                if len(detail_examples['empty_addr']) < 2:
                    detail_examples['empty_addr'].append((s1_rec['name'], m_name_raw, m_rec['addr']))
                continue

            # 3. Domain name as business name
            if '.com' in m_name_raw.lower() or '.in' in m_name_raw.lower() or '.org' in m_name_raw.lower():
                blocking_miss_reasons['Domain Name as Business Name'] += 1
                if len(detail_examples['domain']) < 2:
                    detail_examples['domain'].append((s1_rec['name'], m_name_raw))
                continue

            # 4. Very different name (no overlapping tokens)
            m_name_norm = normalize_business_name(m_name_raw)
            s1_tokens = set(s1_name_norm.split())
            m_tokens = set(m_name_norm.split())
            overlap = s1_tokens.intersection(m_tokens)
            if not overlap or all(len(t) < 3 for t in overlap):
                blocking_miss_reasons['No Overlapping Name Tokens (DBA/Trade Name)'] += 1
                if len(detail_examples['no_overlap']) < 2:
                    detail_examples['no_overlap'].append((s1_rec['name'], m_name_raw, s1_rec['addr'], m_rec['addr']))
                continue

            # 5. Severe typo / misspelling
            blocking_miss_reasons['Typo / Near-Miss (tokens exist but not matched)'] += 1
            if len(detail_examples['typo']) < 2:
                detail_examples['typo'].append((s1_rec['name'], m_name_raw, s1_name_norm, m_name_norm))

print(f"\n{'='*75}")
print(f"BLOCKING FAILURE BREAKDOWN")
print(f"{'='*75}")
print(f"Total true match links evaluated : {total_true}")
print(f"Total missed by blocking         : {blocking_miss_total} ({blocking_miss_total/total_true*100:.2f}%)")
print(f"Blocking Recall achieved         : {(total_true-blocking_miss_total)/total_true*100:.2f}%")
print(f"\n--- MISS CATEGORY BREAKDOWN ---")
for cat, cnt in blocking_miss_reasons.most_common():
    pct = cnt / blocking_miss_total * 100 if blocking_miss_total > 0 else 0
    print(f"  {cat:<50}: {cnt:>5} ({pct:.1f}% of misses)")

print(f"\n{'='*75}")
print(f"DETAILED FAILURE EXAMPLES")
print(f"{'='*75}")
for key, examples in detail_examples.items():
    print(f"\n--- Category: {key} ---")
    for ex in examples[:2]:
        print(f"  S1 Name: \"{ex[0]}\"")
        print(f"  Candidate: \"{ex[1]}\"")
        if len(ex) > 2:
            print(f"  Extra: {ex[2:]}")

# Script distribution in candidate pool
print(f"\n{'='*75}")
print(f"NON-LATIN SCRIPT DISTRIBUTION IN CANDIDATES")
print(f"{'='*75}")
script_counts = Counter()
for eid, rec in cand_raw.items():
    s = detect_script(rec['name'])
    if s:
        script_counts[s] += 1
for s, c in script_counts.most_common():
    print(f"  {s:<15}: {c:>6} records")
