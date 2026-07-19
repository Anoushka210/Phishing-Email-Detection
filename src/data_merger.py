"""
Phase 1 - Steps 3-4: Load each manually-downloaded raw source, map it to
a unified schema, and merge into a single interim CSV.

Note: ethancratchley is NOT merged here - it contains no raw email text,
only precomputed metadata features (num_words, num_links, etc). It's kept
separate and used later as a standalone metadata-only dataset for a
supplementary "structural features only" model comparison.
"""
import pandas as pd

from src.config import RAW_SOURCES, INTERIM_DIR, UNIFIED_COLUMNS


def load_naserabdullahalam() -> pd.DataFrame:
    path = RAW_SOURCES["naserabdullahalam"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - place the downloaded CSV there first."
        )

    df = pd.read_csv(path)
    print(f"[naserabdullahalam] columns found: {list(df.columns)}")

    df = df.rename(columns={
        "text_combined": "email_text",
    })
    df["subject"] = None
    df["sender"] = None
    df["source_dataset"] = "naserabdullahalam"
    return df[UNIFIED_COLUMNS]


def load_ethancratchley_metadata() -> pd.DataFrame:
    """
    Loaded separately - NOT part of the text corpus merge.
    Saved as-is to data/interim/ for later use as a metadata-only
    experiment (Phase 4 supplementary model).
    """
    path = RAW_SOURCES["ethancratchley"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - place the downloaded CSV there first."
        )

    df = pd.read_csv(path)
    print(f"[ethancratchley] columns found: {list(df.columns)} "
          f"(metadata-only, kept separate from text merge)")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df


def merge_all() -> pd.DataFrame:
    # --- Text corpus merge (only sources with raw email text) ---
    text_loaders = [load_naserabdullahalam]
    frames = []

    for loader in text_loaders:
        try:
            frames.append(loader())
        except FileNotFoundError as e:
            print(f"[warning] skipping source: {e}")

    if not frames:
        raise RuntimeError("No text sources loaded - check data/raw/ folder contents.")

    merged = pd.concat(frames, ignore_index=True)
    merged["label"] = merged["label"].astype(str).str.lower().str.strip()

    out_path = INTERIM_DIR / "merged_raw.csv"
    merged.to_csv(out_path, index=False)
    print(f"[saved] text corpus: {len(merged)} rows -> {out_path}")

    # --- Metadata-only dataset, saved separately ---
    try:
        metadata_df = load_ethancratchley_metadata()
        meta_out_path = INTERIM_DIR / "ethancratchley_metadata_only.csv"
        metadata_df.to_csv(meta_out_path, index=False)
        print(f"[saved] metadata-only set: {len(metadata_df)} rows -> {meta_out_path}")
    except FileNotFoundError as e:
        print(f"[warning] skipping metadata source: {e}")

    return merged


if __name__ == "__main__":
    merge_all()