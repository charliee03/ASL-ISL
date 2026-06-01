#!/usr/bin/env python3
"""Validate the ASL-ISL paired training data against the ISL-CSLGR metadata."""

import json
import sys
from pathlib import Path


def validate_pairs(pairs_path, metadata_path=None):
    # Load pairs
    with open(pairs_path) as f:
        pairs = json.load(f)

    errors = []
    warnings = []

    # 1. Basic structure checks
    if not isinstance(pairs, list):
        errors.append("Root element must be a JSON array")
        print_report(errors, warnings, 0)
        return False

    print(f"Total pairs: {len(pairs)}")

    # 2. Check each entry has required keys and no empty values
    required_keys = {"english", "asl", "isl"}
    for i, entry in enumerate(pairs):
        missing = required_keys - set(entry.keys())
        if missing:
            errors.append(f"Entry {i}: missing keys {missing}")
        for key in required_keys:
            if key in entry and not entry[key].strip():
                errors.append(f"Entry {i}: empty value for '{key}'")

    # 3. Check for duplicate English sentences
    english_values = [p["english"] for p in pairs]
    seen = set()
    for i, eng in enumerate(english_values):
        if eng in seen:
            warnings.append(f"Entry {i}: duplicate english sentence '{eng}'")
        seen.add(eng)

    # 4. Token format check: uppercase, space-separated
    for i, entry in enumerate(pairs):
        for field in ["asl", "isl"]:
            val = entry.get(field, "")
            
            # Allow known exceptions: parentheses in "(AGE)", periods, commas
            cleaned = val.replace("(", "").replace(")", "").replace(".", "").replace(",", "")
            if cleaned != cleaned.upper():
                warnings.append(f"Entry {i}: '{field}' value '{val}' contains lowercase characters")

    # 5. Cross-reference ISL glosses against metadata if available
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        known_isl_glosses = set(m["isl_gloss"] for m in metadata)

        for i, entry in enumerate(pairs):
            isl = entry["isl"]
            if isl not in known_isl_glosses:
                warnings.append(f"Entry {i}: ISL gloss '{isl}' not found in metadata.json")

        matched = sum(1 for p in pairs if p["isl"] in known_isl_glosses)
        print(f"ISL glosses matching metadata: {matched}/{len(pairs)}")

    # 6. Summary stats
    identical = sum(1 for p in pairs if p["asl"] == p["isl"])
    different = len(pairs) - identical
    avg_asl_tokens = sum(len(p["asl"].split()) for p in pairs) / len(pairs)
    avg_isl_tokens = sum(len(p["isl"].split()) for p in pairs) / len(pairs)

    print(f"Pairs where ASL == ISL: {identical}")
    print(f"Pairs where ASL != ISL: {different} (these are the interesting translation cases)")
    print(f"Average ASL tokens per gloss: {avg_asl_tokens:.1f}")
    print(f"Average ISL tokens per gloss: {avg_isl_tokens:.1f}")

    # Print report
    print_report(errors, warnings, len(pairs))
    return len(errors) == 0


def print_report(errors, warnings, total):
    print(f"\n{'='*50}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [ERROR] {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN]  {w}")
    if not errors and not warnings:
        print("PASSED: All validation checks passed.")
    elif not errors:
        print(f"PASSED with {len(warnings)} warning(s).")
    else:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s).")


if __name__ == "__main__":
    pairs_path = sys.argv[1] if len(sys.argv) > 1 else "data/paired/asl_isl_pairs.json"
    metadata_path = sys.argv[2] if len(sys.argv) > 2 else "data/isl/processed/metadata.json"

    print(f"Validating: {pairs_path}")
    print(f"Metadata:   {metadata_path}")
    print()

    success = validate_pairs(pairs_path, metadata_path)
    sys.exit(0 if success else 1)
