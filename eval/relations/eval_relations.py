"""Report retirement of the authored relation-language benchmark path.

Historical result artifacts remain historical measurements of removed heuristics.
This entry point performs no benchmark, downloads no data, and writes no results.
"""
import argparse
import json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({
        'status': 'retired',
        'measured': False,
        'reason': 'Automatic language relation extraction, unit guessing, and operand selection were removed.',
        'historical_artifacts': ['eval/results/relation_hotpot.json', 'eval/results/relation_hotpot_after_defect_fixes.json'],
        'replacement': 'Explicit measurement evidence and selected typed calculations; no language benchmark replacement is claimed.',
    }, indent=2))


if __name__ == '__main__':
    main()
