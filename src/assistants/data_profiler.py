from src.agents.file_agent import FileAgent
from dataclasses import dataclass
from typing import List
from hypers import TBD

@dataclass
class DataProfiler(FileAgent):
    name: str = "Data Profiler"
    prompt: str = """Profile the data in {file_path} and generate a comprehensive analysis."""
    instructions: str = """
        You will use Python code to analyze the dataset. Here's how:

        1. First, read the data:
        ```python
        import pandas as pd
        df = pd.read_csv(file_path)
        ```

        2. Generate basic statistics:
        ```python
        # Basic stats for numeric columns
        numeric_stats = df.describe()

        # Missing value analysis
        missing_stats = df.isnull().sum()

        # Categorical column analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        value_counts = etc
        ```

        3. Create visualizations:
        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Distribution plots
        for col in df.select_dtypes(include=['number']).columns:
            plt.figure()
            sns.histplot(df[col])
            plt.title(f'Distribution of col')
            plt.savefig(f'col_distribution.png')
            plt.close()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True)
        plt.savefig('correlation_matrix.png')
        plt.close()
        ```

        4. Save the profile report:
        ```python
        profile_report = {{
            'basic_stats': numeric_stats.to_dict(),
            'missing_values': missing_stats.to_dict(),
            'categorical_summaries': value_counts,
            'generated_plots': ['correlation_matrix.png'] + [f'col_distribution.png' for col in df.select_dtypes(include=['number']).columns]
        }}

        with open('profile_report.json', 'w') as f:
            json.dump(profile_report, f, indent=2)
        ```

        Execute the code in sections, handle any errors, and ensure all outputs are saved.
        Return a summary of the findings and locations of generated files.

        Save the profile report to the current directory and make it available for download.
        """
    model: str = "gpt-4o"
    tools: List[str] = TBD(["code_interpreter"])
    timeout: int = 300
