from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import save_json


def _biochemical_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("biochemical_")]


def build_orthology_transfer(
    human_table_path: Path,
    mouse_table_path: Path,
    biomart_path: Path,
    output_table_path: Path | None,
    output_mapping_path: Path,
    output_manifest_path: Path,
    human_id_column: str,
    mouse_id_column: str,
    homology_type_filter: str,
) -> None:
    human_frame = pd.read_csv(human_table_path)
    mouse_frame = pd.read_csv(mouse_table_path)
    biomart_frame = pd.read_csv(biomart_path, sep=None, engine="python")
    biomart_frame = biomart_frame.rename(
        columns={
            "Gene stable ID": "human_gene_id",
            "Mouse gene stable ID": "mouse_gene_id",
            "Mouse homology type": "homology_type",
        }
    )
    biomart_frame = biomart_frame.dropna(subset=["human_gene_id", "mouse_gene_id"])
    biomart_frame = biomart_frame[
        biomart_frame["homology_type"] == homology_type_filter
    ].copy()

    biochemical_columns = _biochemical_columns(human_frame.columns.tolist())
    if not biochemical_columns:
        raise ValueError("No biochemical columns found in the human table")

    human_transfer_frame = human_frame[[human_id_column, *biochemical_columns]].copy()
    transfer_frame = biomart_frame.merge(
        human_transfer_frame,
        left_on="human_gene_id",
        right_on=human_id_column,
        how="inner",
    )
    transfer_frame = transfer_frame.drop_duplicates(subset=["mouse_gene_id"])

    mouse_base_columns = [
        column
        for column in mouse_frame.columns
        if not column.startswith("biochemical_")
    ]
    transferred_mouse_frame = mouse_frame[mouse_base_columns].merge(
        transfer_frame[["mouse_gene_id", *biochemical_columns]],
        left_on=mouse_id_column,
        right_on="mouse_gene_id",
        how="left",
    )
    transferred_mouse_frame = transferred_mouse_frame.drop(columns=["mouse_gene_id"])

    output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_frame[["human_gene_id", "mouse_gene_id", "homology_type"]].to_csv(
        output_mapping_path,
        index=False,
    )

    if output_table_path is not None:
        output_table_path.parent.mkdir(parents=True, exist_ok=True)
        transferred_mouse_frame.to_csv(output_table_path, index=False)

    transferred_rows = int(
        transferred_mouse_frame[biochemical_columns].notna().any(axis=1).sum()
    )
    save_json(
        {
            "human_table": str(human_table_path.absolute()),
            "mouse_table": str(mouse_table_path.absolute()),
            "biomart_export": str(biomart_path.absolute()),
            "output_table": (
                str(output_table_path.absolute())
                if output_table_path is not None
                else None
            ),
            "output_mapping": str(output_mapping_path.absolute()),
            "human_id_column": human_id_column,
            "mouse_id_column": mouse_id_column,
            "homology_type_filter": homology_type_filter,
            "ortholog_pairs": int(len(transfer_frame)),
            "mouse_rows_with_transferred_biochemistry": transferred_rows,
            "biochemical_column_count": int(len(biochemical_columns)),
            "mouse_biochemical_annotation_origin": "orthology_transferred_human",
            "mouse_biochemical_annotations_are_native": False,
        },
        output_manifest_path,
    )
