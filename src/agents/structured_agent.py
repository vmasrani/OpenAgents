from typing import List
from openai import OpenAI
from abc import abstractmethod
from ml_helpers import timeit
from pydantic import BaseModel
from hypers import TBD
import parallel
from agents.base_agent import BaseAgent

class StructuredOutputAgent(BaseAgent):
    temperature: float = 0.0
    response_format: BaseModel = TBD()

    @property
    @abstractmethod
    def output(self) -> type[BaseModel]:
        """Define the expected response structure"""
        pass

    @abstractmethod
    def _post_process(self, parsed):
        """Post-process the parsed response"""
        pass

    def process_response(self, response) -> None:
        message = response.choices[0].message
        if message.parsed:
            return self._post_process(message.parsed)
        else:
            print(message.refusal)


    @timeit(message="Total GPT processing time")
    def run(self, **kwargs) -> None:
        prompt, instructions = self._get_prompt_and_instructions(**kwargs)

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": instructions
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            response_format=self.output
        )
        return self.process_response(response)

    def pmap(self, list_of_dicts: List[dict], **kwargs) -> None:
        return parallel.pmap(lambda x: self.run(**x),
                             list_of_dicts,
                             prefer='threads',
                             **kwargs)

