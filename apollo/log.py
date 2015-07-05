"""
Implementation of any options for the Apollo logging system
"""

import logging
import sys

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def log_to_stdout():
    # set up logging to stdout e.g. for ipython notebook
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in root.handlers:
        root.removeHandler(handler)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
