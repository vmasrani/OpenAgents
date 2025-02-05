from dataclasses import dataclass, field
from pathlib import Path
from turtle import pd
from typing import List, Dict
from openai import OpenAI
import time
from string import Formatter
from abc import ABC, abstractmethod
from ml_helpers import timeit
from pydantic import BaseModel
from hypers import TBD
import parallel

@dataclass
class Agent(ABC):
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

    @abstractmethod
    def process_response(self, response, *args,**kwargs) -> None:
        """Process the response"""
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Post-process the response"""
        pass

    def pmap(self, **kwargs) -> None:
        """Post-process the response"""
        pass



@dataclass
class FileAgent(Agent):
    name: str
    timeout: int
    tools: List[str]
    temperature: float = 0.0

    def __post_init__(self):
        """Initialize OpenAI client and assistant"""
        self.client = OpenAI()

    def _get_tools(self) -> List[Dict[str, str]]:
        """Convert tool names to OpenAI tool format"""
        tool_mapping = {
            "code_interpreter": {"type": "code_interpreter"},
            "retrieval": {"type": "retrieval"},
            "function": {"type": "function"}
        }
        return [tool_mapping[tool] for tool in self.tools]

    def save_output_file(self, file_id: str, original_path: Path) -> None:
        print("Saving output file...")
        new_filename = original_path.parent / f"{original_path.stem}_cleaned{original_path.suffix}"
        content = self.client.files.content(file_id)
        with open(new_filename, "wb") as f:
            f.write(content.read())


    def process_response(self, response, file_path: Path) -> None:
        cleaned_file_id = response.attachments[0].file_id if response.attachments else None
        if cleaned_file_id:
            self.save_output_file(cleaned_file_id, file_path)
        else:
            print("No output file was generated.")
            print("Assistant's response:")
            for content_block in response.content:
                if content_block.type == 'text':
                    print(content_block.text.value)

    def upload_file(self, file_path: Path) -> str:
        response = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        )
        return response.id


    @timeit(message="Done.")
    def _wait_for_completion(self, thread_id: str, run_id: str) -> None:
        start_time = time.time()
        print("Waiting for assistant to complete...")
        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Assistant did not complete within {self.timeout} seconds")

            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            if run.status == 'completed':
                break
            time.sleep(2)
            print(".", end="", flush=True)

    @timeit(message="Total GPT processing time")
    def run(self, **kwargs) -> None:
        """
        Run the GPT agent on a file with required template variables.
        The file_path is automatically available as {file_path} in the prompt.

        Args:
            file_path: Path to the file to process
            **kwargs: Additional template variables required by the prompt
        """
        # Validate and prepare inputs
        assert 'file_path' in kwargs, "file_path is required"

        file_path = Path(kwargs['file_path'])

        prompt, instructions = self._get_prompt_and_instructions(**kwargs)

        # Upload file and format prompt
        file_id = self.upload_file(file_path)

        assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=instructions,
            model=self.model,
            tools=self._get_tools()
        )

        # Create thread and start run
        thread = self.client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": prompt,
                "attachments": [{
                    "file_id": file_id,
                    "tools": self._get_tools()
                }]
            }]
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            temperature=self.temperature
        )

        # Wait for completion
        self._wait_for_completion(thread.id, run.id)

        # Process results
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        self.process_response(messages.data[0], file_path)

    def pmap(self, file_paths: List[Path], **kwargs) -> None:

        """Post-process the response"""
        def make_kwargs(file_path: Path) -> dict:
            kwargs = kwargs.copy()  # Don't modify original
            kwargs['file_path'] = file_path
            return kwargs

        mapped_kwargs = [make_kwargs(fp) for fp in file_paths]

        return parallel.pmap(self.run, mapped_kwargs, prefer='threads', **kwargs)


@dataclass
class StructuredOutputAgent(Agent):
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

