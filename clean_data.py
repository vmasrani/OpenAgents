from pathlib import Path
import sys

from openai import BaseModel
import pandas as pd
from agents import StructuredOutputAgent, FileAgent
from dataclasses import dataclass, field
from typing import List
from hypers import TBD


@dataclass
class Cleaner(StructuredOutputAgent):
    name: str = "Data Cleaner"
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

    model: str = "gpt-4-turbo-preview"
    tools: List[str] = TBD(["code_interpreter"])

    class _OutputStructure(BaseModel):
        dirty_headers: list[str]
        clean_headers: list[str]

    @property
    def output(self) -> type[BaseModel]:
        return self._OutputStructure

    def _post_process(self, parsed):
        """Post-process the parsed response"""
        return pd.DataFrame({
            'dirty_headers': parsed.dirty_headers,
            'clean_headers': parsed.clean_headers,
        })

def main():
    if len(sys.argv) < 2:
        print("Error: A file path must be provided")
        sys.exit(1)

    file_path = sys.argv[1]

    # cleaner = DataCleaner()
    # cleaner.run(file_path=file_path)




if __name__ == "__main__":
    main()

# @dataclass
# class DataCleaner(FileAgent):
#     name: str = "Data Cleaner"
#     prompt: str = "Clean the data in {file_path} following the instructions provided."
#     instructions: str = """
#         You are a data cleaner that reads files, applies cleaning rules, and saves a cleaned version.
#         Follow these steps:
#         1. Remove any duplicate rows
#         2. Handle missing values appropriately
#         3. Standardize date formats to YYYY-MM-DD
#         4. Remove any obvious outliers (values > 3 standard deviations)
#         Return the cleaned file."""
#     model: str = "gpt-4-turbo-preview"
#     tools: List[str] = TBD(["code_interpreter"])
#     timeout: int = 300


# def main():
#     if len(sys.argv) < 2:
#         print("Error: A file path must be provided")
#         sys.exit(1)

#     file_path = sys.argv[1]

#     # cleaner = DataCleaner()
#     # cleaner.run(file_path=file_path)




# if __name__ == "__main__":
#     main()
