from src.agents import StructuredOutputAgent, FileAgent
from dataclasses import dataclass
from typing import List
from hypers import TBD
from src.structure import BijectiveListMixin

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

