# Example Python Script

This directory (`src/`) contains your project's Python source code.

Create your modules and packages here. Example:

```python
# src/main.py
import torch
import numpy as np

def hello():
    print("Engineering AI Project")
    print(f"PyTorch version: {torch.__version__}")

if __name__ == "__main__":
    hello()
```

Run it with:
```bash
python src/main.py
```

## Problem 4.1 Script

To run the Bayesian sensor reasoning program:

```bash
python src/bayesian_sensor_reasoning.py
```

Optional arguments:

```bash
python src/bayesian_sensor_reasoning.py --prior 0.15 --tp 0.85 --fp 0.08 --tp2 0.95 --fp2 0.02
```
