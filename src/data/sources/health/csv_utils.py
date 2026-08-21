"""Shared DATASUS CSV parsing helpers.

DATASUS's exported CSVs are semicolon-delimited but not strictly rectangular:
rows are sometimes short (trailing columns omitted) or long (an unescaped
`;` inside a text field splits it into extra columns). Both the live-scrape
path (fetch/forms.py) and the batch-download path (preprocess/preprocess.py)
need to repair this the same way before handing rows to pandas.
"""

import csv


def normalize_rows_to_header_width(header, rows):
    """Pad short rows and rejoin overflowing rows so every row matches `header`'s width."""
    expected_width = len(header)
    normalized_rows = []
    for row in rows:
        if len(row) < expected_width:
            row = row + [None] * (expected_width - len(row))
        elif len(row) > expected_width:
            row = row[: expected_width - 1] + [";".join(row[expected_width - 1 :])]
        normalized_rows.append(row)
    return normalized_rows


def parse_datasus_csv_rows(text_or_lines, *, delimiter=";", quotechar='"'):
    """Parse DATASUS CSV rows/text into (header, width-normalized body rows)."""
    rows = list(csv.reader(text_or_lines, delimiter=delimiter, quotechar=quotechar))
    if not rows:
        return None, []
    header, body = rows[0], rows[1:]
    return header, normalize_rows_to_header_width(header, body)
