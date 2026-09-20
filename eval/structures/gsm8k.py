"""Report retirement of the cue-driven arithmetic word-problem evaluator.

Historical results measured removed language and operand-selection heuristics.
This entry point runs no benchmark, downloads no data, and writes no results.
Explicit measurements and selected calculations do not replace GSM8K evaluation.
"""
import argparse
import json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({
        'status': 'retired',
        'measured': False,
        'reason': 'Numeric language extraction, lexical unit guessing, and cue-selected arithmetic were removed.',
        'historical_artifacts': ['eval/results/structures_gsm8k.json'],
        'replacement': 'Explicit measurement evidence and selected typed calculations; no language benchmark replacement is claimed.',
    }, indent=2))


if __name__ == '__main__':
    main()
