This is the author's implementation of the paper 'A Novel Framework for Uncertainty-Driven Adaptive Exploration'. 
All code is implemented in python. It is suggested to use a virtual env such as conda or venv. 
Please use python=3.10. 

To download to it follow the above commands:

## Installation

Clone the repository:
```bash
git clone https://github.com/leoBakop/adaptive_exploration
cd adaptive_exploration
```

Create and activate the conda environment:
```bash
conda create -n adeu python=3.10
conda activate adeu
```

Install dependencies:
```bash
pip install -r requirements.txt
```

To test the installation pleaserun a simple MuJoCo experiment:
```python
python3 test/test_mujoco.py     
```


