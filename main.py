from pathlib import Path
import sys
from src.assistants.data_profiler import DataProfiler
from src.assistants.structured_cleaner import StructuredCleaner
import pandas as pd


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
    profiler = DataProfiler()
    profiler.run(file_path='./data/sample_data_001.csv')



if __name__ == "__main__":
    # demo_structured_cleaning()
    demo_data_cleaning()
