# %% md

# # Fig 5e: Genus control experiment

# Design probes for a gene in the genome of an entire genus.

# Colocalize the gene stain with the 16s rRNA genus stain.

# =============================================================================
# ## Setup
# =============================================================================

# Imports.

# %% codecell
import os
import glob
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3e/probe_design'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# =============================================================================
# ## Get dictionairy of grouped MAGs
# =============================================================================

# Load the table

# %% codecell
output_dir = '../../../../outputs/fig_5/metagenomic_analysis'
sum_tab_fn = output_dir + '/Summary_small_NN.txt'
sum_tab = pd.read_csv(sum_tab_fn, sep='\t')
sum_tab.columns
