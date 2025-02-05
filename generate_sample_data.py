from pathlib import Path
import pandas as pd
import numpy as np

def generate_messy_data(output_path: Path = Path("sample_data.csv"), rows: int = 1000):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate dates with different formats
    dates = pd.date_range('2023-01-01', periods=rows)
    messy_dates = [
        d.strftime(np.random.choice(['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']))
        for d in dates
    ]

    # Generate numeric data with outliers and missing values
    values = np.random.normal(100, 15, rows)
    values[np.random.choice(rows, 50)] = np.random.normal(500, 50, 50)  # outliers
    values[np.random.choice(rows, 30)] = np.nan  # missing values

    # Generate categorical data with inconsistent cases
    categories = ['red', 'blue', 'green']
    cats = [np.random.choice(categories) for _ in range(rows)]
    cats = [c.upper() if np.random.random() > 0.5 else c for c in cats]

    # Create DataFrame with some duplicates
    df = pd.DataFrame({
        'date': messy_dates,
        'value': values,
        'category': cats,
        'id': range(rows)
    })

    # Add some duplicates
    duplicates = df.sample(n=50)
    df = pd.concat([df, duplicates])

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated messy data saved to {output_path}")

if __name__ == "__main__":
    generate_messy_data()
