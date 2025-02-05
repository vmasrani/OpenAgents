from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Dict
from openai import OpenAI
import time

import pandas as pd
from ml_helpers import timeit
import parallel
from src.agents.base_agent import BaseAgent


@dataclass
class FileAgent(BaseAgent):
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
        print(f"Saving results to {original_path}...")
        content = self.client.files.content(file_id)
        with open(original_path, "wb") as f:
            f.write(content.read())


    def process_response(self, response, file_path: Path) -> pd.DataFrame:
        import ipdb; ipdb.set_trace()
        cleaned_file_id = response.attachments[0].file_id if response.attachments else None
        response_text = ""
        if cleaned_file_id:
            self.save_output_file(cleaned_file_id, file_path)
        else:
            print(f"{file_path} failed to save")
            for content_block in response.content:
                if content_block.type == 'text':
                    response_text = content_block.text.value

        return pd.DataFrame({
            'file_path': [str(file_path)],
            'success': [cleaned_file_id is not None],
            'response': [response_text]
        })

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
        return self.process_response(messages.data[0], file_path)

    def pmap(self, file_paths: List[Path], **kwargs) -> tuple[list, list]:

        default_kwargs = kwargs.copy()

        """Post-process the response"""
        def make_kwargs(file_path: Path) -> dict:
            kwargs = default_kwargs.copy()  # Don't modify original
            kwargs['file_path'] = file_path
            return kwargs

        # to avoid rate limiting
        def jitter_run(kwargs: dict) -> dict:
            time.sleep(random.uniform(1, 5))
            return self.run(**kwargs)

        mapped_kwargs = [make_kwargs(fp) for fp in file_paths]

        res = parallel.pmap(jitter_run, mapped_kwargs, prefer='threads', safe_mode=True, **kwargs)
        successes, failures = self.split_success_failure(res)
        return successes, failures
