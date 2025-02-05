from pathlib import Path
import pandas as pd
import numpy as np

def generate_messy_data(output_dir: Path = Path("sample_data"), num_files: int = 100, rows_per_file: int = 100):
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    for file_num in range(num_files):
        # Generate dates with different formats
        dates = pd.date_range('2023-01-01', periods=rows_per_file)
        messy_dates = [
            d.strftime(np.random.choice(['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']))
            for d in dates
        ]

        # Generate numeric data with outliers and missing values
        values = np.random.normal(100, 15, rows_per_file)
        values[np.random.choice(rows_per_file, 5)] = np.random.normal(500, 50, 5)  # outliers
        values[np.random.choice(rows_per_file, 3)] = np.nan  # missing values

        # Generate categorical data with inconsistent cases
        categories = ['red', 'blue', 'green']
        cats = [np.random.choice(categories) for _ in range(rows_per_file)]
        cats = [c.upper() if np.random.random() > 0.5 else c for c in cats]

        # Create DataFrame with some duplicates
        df = pd.DataFrame({
            'date': messy_dates,
            'value': values,
            'category': cats,
            'id': range(rows_per_file)
        })

        # Add some duplicates (5% of rows)
        duplicates = df.sample(n=rows_per_file // 20)
        df = pd.concat([df, duplicates])

        # Save to CSV
        output_path = output_dir / f"sample_data_{file_num:03d}.csv"
        df.to_csv(output_path, index=False)

    print(f"Generated {num_files} messy data files in {output_dir}/")

if __name__ == "__main__":
    generate_messy_data()
