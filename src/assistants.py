from src.agents import StructuredOutputAgent, FileAgent
from dataclasses import dataclass
from typing import List
from hypers import TBD
from src.structure import BijectiveListMixin

@dataclass
class DataCleaner(FileAgent):
    model: str = "gpt-4o"
    name: str = "Data Cleaner"
    prompt: str = "Clean the data in {file_path} following the instructions provided."
    instructions: str = """
        You are a data cleaner that reads files, applies cleaning rules, and saves a cleaned version.
        Follow these steps:
        1. Remove any duplicate rows
        2. Handle missing values appropriately
        3. Standardize date formats to YYYY-MM-DD
        4. Remove any obvious outliers (values > 3 standard deviations)
        Save the cleaned file using the interpreter tool."""
    tools: List[str] = TBD(["code_interpreter"])
    timeout: int = 300


@dataclass
class StructuredCleaner(BijectiveListMixin, StructuredOutputAgent):
    model: str = "gpt-4o"
    name: str = "Structured Cleaner"
    instructions: str = """
    Map CSV headers to standardized clean headers following these rules:
        1. Maintain one-to-one mapping between dirty and clean headers
        2. NO DUPLICATE clean headers allowed
        3. Preserve information when combining fields
            Example: 'job_company_location_street_address' -> 'company_street_address'
        4. Create new headers if no valid match exists
        5. Use CSV row context for better mapping decisions
        6. Preserve original header if already clean
        7. Keep 'zip' and 'postal_code' distinct
        8. All headers must contain a 'email' field (or 'email' and 'work_email' if two email fields are present)
        9. map urls to 'source'
        Return headers in original order."""
    prompt: str = """
    CSV preview (header may be in first row):
    {first_five}

    Map to these valid headers:
    {valid_headers}

    Return ordered lists of dirty and clean headers.
    """
    tools: List[str] = TBD(["code_interpreter"])


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
