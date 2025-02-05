from pathlib import Path
import sys
from src.assistants import StructuredCleaner, DataCleaner
import pandas as pd

if len(sys.argv) < 2:
    print("Error: A file path must be provided")
    sys.exit(1)

file_path = sys.argv[1]


def demo_structured_cleaning():
    def make_kwargs(file_path: Path) -> dict:
        df = pd.read_csv(file_path)
        return {
            'first_five': df.head(5).to_string(),
            'valid_headers': df.columns.tolist()
        }

    cleaner = StructuredCleaner()
    file_paths = [make_kwargs(p) for p in Path("./data").glob("*.csv")]
    df = pd.concat(cleaner.pmap(file_paths))
    df.to_csv("cleaned_data.csv", index=False)

def demo_data_cleaning():
    cleaner = DataCleaner()
    cleaner.run(file_path=file_path)



if __name__ == "__main__":
    demo_structured_cleaning()
    demo_data_cleaning()
