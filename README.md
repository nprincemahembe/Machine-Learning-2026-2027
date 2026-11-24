# Introduction to Machine Learning Systems — Labs 01–07

Companion implementation to the supplied lab manuals and their assigned reading in Vijay Janapa Reddi, *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems*.

**Important:** This package contains reproducible implementations and written analysis, not fabricated experiment results. Run the scripts to produce hardware-dependent evidence. No Git commits, MLflow runs, or lecturer approval are claimed. The supplied manuals require independent work and your own commit history; review and adapt before submission.

## Setup
Use Python 3.11.8, create a virtual environment, then `pip install -r requirements.txt`. From the repository root run `python lab01/src/run.py`, and similarly for each lab. Lab 3 and Labs 6–7 download Fashion-MNIST to `data/` (internet needed). Lab 4 fetches Adult from OpenML. Lab 6 additionally requires MLflow. CPU runs for labs 3, 6, 7 can take many hours.

Each `report.md` is a structured report with methods, interpretive answers and clear locations for real results. Outputs are saved in each lab's `results/` folder. Hardware benchmarks are meaningful only on the machine used to execute them.
