# OpenAgents
_AI generated README_

A lightweight wrapper around OpenAI's API that allows you to quickly define LLM Agents that do useful things.

## Overview

OpenAgents provides a flexible framework for creating AI agents powered by OpenAI's API. It includes base classes for both file-based and structured output agents, along with several pre-built assistants for common data tasks.

## Features

- **File-based Agents**: Process files with GPT-4 and handle file uploads/downloads
- **Structured Output Agents**: Get structured, validated responses from GPT
- **Parallel Processing**: Built-in support for parallel execution
- **Pre-built Assistants**:
  - Data Profiler: Generate comprehensive data analysis reports
  - Data Cleaner: Clean and standardize data files
  - Structured Cleaner: Map and standardize CSV headers

## Installation

```bash
pip install -r requirements.txt
```

And make sure your your OPENAI_API_KEY is set in your env variables
## Quick Start

```python
from src.assistants.data_profiler import DataProfiler
from src.assistants.structured_cleaner import StructuredCleaner

# Profile a dataset
profiler = DataProfiler()
profiler.run(file_path='./data/sample_data.csv')

# Clean data structure
cleaner = StructuredCleaner()
results = cleaner.run(
    first_five=df.head(5).to_string(),
    valid_headers=df.columns.tolist()
)
```

## Agent Types

### FileAgent
Base class for agents that process files. Handles:
- File upload/download with OpenAI
- Async processing with timeouts
- Tool management (code interpreter, retrieval, function calling)

### StructuredOutputAgent
Base class for agents that return structured data. Features:
- Pydantic model validation
- Parallel processing support
- Custom response formatting

## Pre-built Assistants

### DataProfiler
Generates comprehensive data analysis reports including:
- Basic statistics
- Missing value analysis
- Visualizations
- Correlation analysis
- Distribution plots

### DataCleaner
Automated data cleaning pipeline that:
- Removes duplicates
- Handles missing values
- Standardizes date formats
- Removes outliers

### StructuredCleaner
Intelligent CSV header standardization that:
- Maps headers to standardized names
- Maintains bijective mapping
- Preserves information content
- Uses row context for better mapping

## Development

### Project Structure
```
openagents/
├── src/
│   ├── agents/
│   │   ├── base_agent.py      # Base agent classes
│   │   ├── file_agent.py      # File processing agents
│   │   └── structured_agent.py # Structured output agents
│   ├── assistants/
│   │   ├── data_cleaner.py    # Data cleaning assistant
│   │   ├── data_profiler.py   # Data profiling assistant
│   │   └── structured_cleaner.py # Header standardization
│   └── structure.py           # Response models
└── main.py                    # Usage examples
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]


