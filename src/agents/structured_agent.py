from src.agents.structured_agent import StructuredOutputAgent
from dataclasses import dataclass, field
from typing import List
from src.structure import BijectiveListMixin


@dataclass()
class StructuredCleaner(BijectiveListMixin, StructuredOutputAgent):
    model: str = "gpt-4o-mini"
    name: str = "Structured Cleaner"
    instructions: str = """
    Follow this two-step process to map CSV headers:

    STEP 1 - Initial Mapping:
        1. Maintain one-to-one mapping between dirty and clean headers
        2. NO DUPLICATE clean headers allowed
        3. Preserve information when combining fields
            Example: 'job_company_location_street_address' -> 'company_street_address'
        4. Create new headers if no valid match exists
        5. Use CSV row context for better mapping decisions
        6. Preserve original header if already clean
        7. Keep 'zip' and 'postal_code' as distinct fields - do not convert between them
        8. All headers must contain a 'email' field (or 'email' and 'work_email' if two email fields are present)
        9. Preserve social media URLs in their original form:
            - Keep 'linkedin_url' as 'linkedin_url'
            - Keep 'twitter_url' as 'twitter_url'
            - Keep 'facebook_url' as 'facebook_url'
            - Keep 'github_url' as 'github_url'
            Only map generic/unknown URLs to 'source'

    STEP 2 - Validation Check:
        Review the proposed mapping and verify:
        1. No 'zip' fields were converted to 'postal_code' or vice versa
        2. All social media URLs maintain their specific platform identifiers
        3. No information loss occurred in field combinations
        4. Email fields are properly preserved
        5. No duplicate clean headers exist
        6. All mappings are one-to-one

        If any issues are found, revise the mapping before returning.

    Return the final validated headers in original order.

    """
    prompt: str = """
    CSV preview (header may be in first row):
    {first_five}

    Map to these valid headers:
    {valid_headers}

    First create a mapping, then validate it against these criteria:
    1. No 'zip'/'postal_code' conversions
    2. Social media URLs preserved
    3. No information loss
    4. Email fields preserved
    5. No duplicate clean headers
    6. One-to-one mapping maintained

    Return the final validated ordered lists of dirty and clean headers.
    """
    valid_headers: List[str] = field(default_factory=list)
