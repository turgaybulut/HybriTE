from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-table", required=True)
    parser.add_argument("--mouse-table", required=True)
    parser.add_argument("--biomart", required=True)
    parser.add_argument("--output-table", default=None)
    parser.add_argument("--output-mapping", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--human-id-column", default="gene_id")
    parser.add_argument("--mouse-id-column", default="gene_id")
    parser.add_argument("--homology-type-filter", default="ortholog_one2one")
    return parser.parse_args()


def main() -> None:
    from hybrite.transfer_orthology import build_orthology_transfer

    args = parse_args()
    build_orthology_transfer(
        human_table_path=Path(args.human_table),
        mouse_table_path=Path(args.mouse_table),
        biomart_path=Path(args.biomart),
        output_table_path=Path(args.output_table)
        if args.output_table is not None
        else None,
        output_mapping_path=Path(args.output_mapping),
        output_manifest_path=Path(args.output_manifest),
        human_id_column=args.human_id_column,
        mouse_id_column=args.mouse_id_column,
        homology_type_filter=args.homology_type_filter,
    )


if __name__ == "__main__":
    main()
