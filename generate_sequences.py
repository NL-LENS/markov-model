"""Generate Markov life-event sequences and write them to a tidy CSV.

Example
-------
    python generate_sequences.py --n 1000 --seed 0 --out sequences.csv

The output has one row per event step; its attributes are collected into a
single ``context`` column holding a dictionary:

    seq_id, step, time, event, context

An event with no attributes (e.g. ``birth``) yields ``{}``; ``start_job`` yields
``{'company': 'c8', 'salary': 74.54}``. With ``--no-timestamps`` the ``time``
column is left blank.
"""

import argparse
import csv

import numpy as np

from model import MarkovEventModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML model config.")
    parser.add_argument("--n", type=int, default=1000, help="Number of sequences to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out", default="sequences.csv", help="Output CSV path.")
    parser.add_argument("--max-len", type=int, default=50, help="Max events per sequence.")
    parser.add_argument("--no-attributes", action="store_true", help="Disable attribute generation.")
    parser.add_argument("--no-timestamps", action="store_true", help="Disable timestamp generation.")
    parser.add_argument(
        "--bayes-error",
        action="store_true",
        help="Also print Monte-Carlo Bayes-error estimates.",
    )
    return parser.parse_args()


def write_sequences(model: MarkovEventModel, rng, n: int, max_len: int, out_path: str) -> int:
    """Simulate ``n`` sequences and write one row per event step. Returns the row count."""
    rows_written = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["seq_id", "step", "time", "event", "context"])
        for seq_id in range(n):
            sequence = model.simulate_sequence(rng, max_len=max_len)
            for step, (time, event, attrs) in enumerate(sequence):
                time_cell = "" if time is None else time
                context = str(dict(attrs))
                writer.writerow([seq_id, step, time_cell, event, context])
                rows_written += 1
    return rows_written


def main() -> None:
    args = parse_args()

    model = MarkovEventModel.from_yaml(
        args.config,
        generate_attributes=not args.no_attributes,
        generate_timestamps=not args.no_timestamps,
    )
    rng = np.random.default_rng(args.seed)

    rows = write_sequences(model, rng, args.n, args.max_len, args.out)
    print(f"Wrote {args.n} sequences ({rows} rows) to {args.out}")

    if args.bayes_error:
        # Use a fresh generator so the reported estimates are independent of the
        # sequences written above.
        eval_rng = np.random.default_rng(args.seed + 1)
        event_err = model.bayes_error_events(eval_rng, n_sequences=5_000, max_len=args.max_len)
        attr_err = model.bayes_error_attributes(eval_rng, n_sequences=5_000, max_len=args.max_len)
        print(f"Monte-Carlo Bayes error (next event): {event_err:.4f}")
        for name, err in attr_err.items():
            print(f"Monte-Carlo Bayes error ({name}): {err:.4f}")


if __name__ == "__main__":
    main()
