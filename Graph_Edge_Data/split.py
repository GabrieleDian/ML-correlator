import pandas as pd
import sys

def split(file1, file2, output_file):
    # Read both CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Strip potential whitespace and ensure consistent string format
    df1["EDGES"] = df1["EDGES"].str.strip()
    df2["EDGES"] = df2["EDGES"].str.strip()

    # Find rows in df2 not in df1
    df_extra = pd.merge(df2, df1, on=["COEFFICIENTS", "EDGES"], how="outer", indicator=True)
    df_extra = df_extra[df_extra["_merge"] == "left_only"].drop(columns=["_merge"])

    # Save to new file
    df_extra.to_csv(output_file, index=False)
    print(f"Saved {len(df_extra)} additional rows to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_extra_rows.py file1.csv file2.csv extra_rows.csv")
        sys.exit(1)

    file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    split(file1, file2, output_file)
