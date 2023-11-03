# Does Difficulty even Matter? Investigating Difficulty Adjustment and Practice Behavior in an Open-Ended Learning Task

This repository contains data, code, and plots used in the paper "Does Difficulty even Matter? Investigating Difficulty Adjustment and Practice Behavior in an Open-Ended Learning Task". There are two main Python scripts for generating the results:

- `compare-conditions.py` - Generates plots comparing the different measures between the conditions.
- `clustering.py` - Reads the practice behaviors of the students, then clusters the students based on that. The script outputs plots comparing different measures between the clusters, and performs Kruskal-Wallis tests. The script requires a JSON setting file, which is at `plots/click_type_num_time/settings.json`. In this case, we run the script with `python clustering.py click_type_num_time`.

We recommend using venv, with the pip requirements provided `requirements.txt`.
