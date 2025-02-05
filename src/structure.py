
from openai import BaseModel
import pandas as pd

class _Output(BaseModel, extra='forbid'):
    from_list: list[str]
    to_list: list[str]

class BijectiveListMixin:
    @property
    def output(self) -> type[_Output]:
        """Implements StructuredOutputAgent's abstract output property"""
        return _Output

    def _post_process(self, parsed):
        """Post-process the parsed response"""
        return pd.DataFrame({
            'from_list': parsed.from_list,
            'to_list': parsed.to_list,
        })

