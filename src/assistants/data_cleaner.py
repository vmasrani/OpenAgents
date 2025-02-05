from src.agents.file_agent import FileAgent
from dataclasses import dataclass
from typing import List
from hypers import TBD

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

