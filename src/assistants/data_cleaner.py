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
        You are a data cleaner that reads json files and applies cleaning rules.
        Follow these steps:
        1. Read the input json file using the code interpreter
        2. Make sure all the values in the json are unique
        3. If there are duplicates, keep the first value, and intelligently come up with a new value for the duplicate. The new value should be consistent style-wise with the other values in the json.
        4. Save the cleaned data back to a new json file using the code interpreter
        5. Return a message confirming the file has been cleaned with a link to download it
    """
    tools: List[str] = TBD(["code_interpreter"])
    timeout: int = 300

