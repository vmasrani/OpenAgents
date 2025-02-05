from dataclasses import dataclass
from openai import OpenAI
from string import Formatter
from abc import ABC, abstractmethod

import pandas as pd


@dataclass
class BaseAgent(ABC):
    prompt: str
    instructions: str
    model: str

    def __post_init__(self):
        """Initialize OpenAI client and assistant"""
        self.client = OpenAI()

    def _get_required_kwargs(self) -> set:
        """Extract required template variables from prompt"""
        return {
            fname for _, fname, _, _
            in Formatter().parse(self.prompt)
            if fname is not None
        }

    def _get_prompt_and_instructions(self, **kwargs) -> tuple[str, str]:
        prompt = self.prompt.format(**kwargs)
        instructions = self.instructions.format(**kwargs)
        return prompt, instructions

    def split_success_failure(self, res: list) -> tuple[list, list]:
        successes = [r for r in res if isinstance(r, pd.DataFrame)]
        failures = [r for r in res if isinstance(r, dict)]
        if failures:
            failures = pd.DataFrame(failures)
        if successes:
            successes = pd.concat(successes)
        return successes, failures


    @abstractmethod
    def process_response(self, response, *args,**kwargs) -> None:
        """Process the response"""
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Post-process the response"""
        pass


