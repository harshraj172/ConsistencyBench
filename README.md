# ConsistencyBench

## Setup

**Install the requirements**
```bash
conda create -n venv python=3.10
conda activate venv
pip install -r requirements.txt
```

**To start generation and scoring**

Set the arguments from `run_eval.py`

```bash
python run_eval.py --openai_api_key <OPENAI_API_KEY>
```

**Example Output file**: `consistencybench/result_gpt-3.5-turbo_paraphrasing.csv`

**Example Jupyter Notebook**: `example.ipynb`
