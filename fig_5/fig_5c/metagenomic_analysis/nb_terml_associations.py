# %% md

# # Figure 5c: Phage TermL2 associations with metagenome

# Used "hiprfish" conda environment

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

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/metagenomic_analysis'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# =============================================================================
# ## Load binning summary table and look for genuses
# =============================================================================

# Load the table

# %% codecell
output_dir = '../../../../outputs/fig_5/metagenomic_analysis'
sum_tab_fn = output_dir + '/Summary_small_NN.txt'
sum_tab = pd.read_csv(sum_tab_fn, sep='\t')
sum_tab.columns

# %% md

# Search for target genuses

# %% codecell
target_taxa = ['Fretibacterium', 'Oribacterium', 'Butyrivibrio', 'Christensenella']
tax_sum_list = []
for tax in target_taxa:
    bool_k = sum_tab['ggenus'] == 'g__' + tax
    bool_g = sum_tab['kgenus'] == tax
    sub = sum_tab.loc[bool_k | bool_g, :]
    tax_sum_list.append(sub)
tax_sum_df = pd.concat(tax_sum_list)
tax_sum_df

# %% md

# Blast terml gene against contigs

# Generate command then run in shell

# %% codecell
output_dir = '../../../../outputs/fig_5/metagenomic_analysis'
gene_input_fn = '../../../../data/fig_5/fig_5c/probe_design_input/BMG_I_ph624_termL.fasta'
gene_basename = os.path.splitext(os.path.split(gene_input_fn)[1])[0]
contig_db_fn = output_dir + '/contig_blastdb/2021_04_03_contigs.fna'
gene_blast_fn = output_dir + '/' + gene_basename + '.fasta.blast.out'
out_format = ('"6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids"')

command = " ".join([
        'blastn',
        '-db', contig_db_fn,
        '-query', gene_input_fn,
        '-out', gene_blast_fn,
        '-outfmt', out_format
        ])

run_fn = 'run_{}.sh'.format('blast_termL')
with open(run_fn, 'w') as f:
    f.write(command)

# %% md

# Check blast results

# %% codecell
# get blast results
gene_blast = pd.read_csv(gene_blast_fn, sep='\t')
gene_blast.columns = [out_format.split()[1:]]
gene_blast_ids = gene_blast['sseqid'].values.squeeze()
gene_blast_ids

# %% md

# Search for blast result contigs in binning

# %% codecell
# Get dict for contig to bin assignment
sample_key_fn = '../../../../data/fig_5/metagenomic_sequencing/sample_key.csv'
sample_key = pd.read_csv(sample_key_fn)
dict_assm_pacb = {a:p for p,a in sample_key.values}
dict_contig_bin = {}
bin_dir = output_dir + '/bins'
bin_fns = glob.glob(bin_dir + '/*.fna')
for fn in bin_fns:
    bin_id = os.path.splitext(os.path.split(fn)[1])[0]
    assm_id = re.search(r'^S\dC', record.id).group(0)
    pacb_id = dict_assm_pacb[assm_id]
    cont_id = re.sub(assm_id + 'NODE', pacb_id, record.id)  # Convert assembly id to pacbio id, which is used in the blast db
    for record in SeqIO.parse(fn, 'fasta'):
        dict_contig_bin[cont_id] = bin_id

# %% codecell
# Get dict for bin to family/genus/species assignment
sample_key_fn = '../../../../data/fig_5/metagenomic_sequencing/sample_key.csv'
sample_key = pd.read_csv(sample_key_fn)
tax_cols = ['bin_ID', 'kfamily', 'kgenus', 'kspecies', 'gfamily', 'ggenus', 'gspecies']
dict_bin_tax = {}
for r in sum_tab[tax_cols].values:
    dict_bin_tax[r[0]] = r[1:].tolist()


# %% codecell
# convert contig alignment to familygenus species
tax_list = []
for contig in gene_blast_ids:
    try:
        bin = dict_contig_bin[contig[0]]
        tax_list.append(dict_bin_tax[bin])
    except:
        pass

tax_list
for k in dict_contig_bin.keys():
    if '653' in k:
        print(k)

# %% codecell
from pylab import *
cmap = cm.get_cmap('tab20', 20)    # PiYG
hex = []
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    hex.append(matplotlib.colors.rgb2hex(rgba)[1:])
",".join(hex)












# %% md

# Blast terml gene against contigs

# Generate command then run in shell

# %% codecell
output_dir = '../../../../outputs/fig_5/metagenomic_analysis'
gene_input_fn = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/plaque/metagenomic_analysis/2021_11_09_albert_phage/lytic_terminase_large.fa'
gene_basename = os.path.splitext(os.path.split(gene_input_fn)[1])[0]
contig_db_fn = output_dir + '/contig_blastdb/2021_04_03_contigs.fna'
gene_blast_fn = output_dir + '/' + gene_basename + '.fasta.blast.out'
out_format = ('"6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids"')

command = " ".join([
        'blastn',
        '-db', contig_db_fn,
        '-query', gene_input_fn,
        '-out', gene_blast_fn,
        '-outfmt', out_format
        ])

run_fn = 'run_{}.sh'.format('blast_terml')
with open(run_fn, 'w') as f:
    f.write(command)

# %% md

# Check blast results

# %% codecell
# get blast results
gene_blast = pd.read_csv(gene_blast_fn, sep='\t')
gene_blast.columns = [out_format.split()[1:]]
gene_blast_ids = gene_blast[['qseqid','sseqid']].values
gene_blast_ids

# %% md

# Search for blast result contigs in binning

# %% codecell
# Get dict for contig to bin assignment
sample_key_fn = '../../../../data/fig_5/metagenomic_sequencing/sample_key.csv'
sample_key = pd.read_csv(sample_key_fn)
dict_assm_pacb = {a:p for p,a in sample_key.values}
dict_contig_bin = {}
bin_dir = output_dir + '/bins'
bin_fns = glob.glob(bin_dir + '/*.fna')
for fn in bin_fns:
    bin_id = os.path.splitext(os.path.split(fn)[1])[0]
    for record in SeqIO.parse(fn, 'fasta'):
        assm_id = re.search(r'^S\dC', record.id).group(0)
        pacb_id = dict_assm_pacb[assm_id]
        cont_id = re.sub(assm_id + 'NODE', pacb_id, record.id)  # Convert assembly id to pacbio id, which is used in the blast db
        dict_contig_bin[cont_id] = bin_id

# %% codecell
# Get dict for bin to family/genus/species assignment
sample_key_fn = '../../../../data/fig_5/metagenomic_sequencing/sample_key.csv'
sample_key = pd.read_csv(sample_key_fn)
tax_cols = ['bin_ID', 'kgenus', 'ggenus']
dict_bin_tax = {}
for r in sum_tab[tax_cols].values:
    dict_bin_tax[r[0]] = r[1:].tolist()


# %% codecell
# convert contig alignment to familygenus species
dict_phage_tax = defaultdict(list)
for phage, contig in gene_blast_ids:
    try:
        bin_ = dict_contig_bin[contig]
        dict_phage_tax[phage].append(dict_bin_tax[bin_])
    except:
        pass

len(dict_phage_tax)

# %% codecell
dict_phage_tax_unq = {}
for k, v in dict_phage_tax.items():
    cov = float(re.search('(?<=cov_)\d+.\d+', k).group(0))
    dict_phage_tax_unq[cov] = [k, np.unique(v, return_counts=True)]

dict(sorted(dict_phage_tax_unq.items()))


# %% codecell
['S4CNODE_624_length_20248_cov_118.281880_4', 'Veillonella']

['S3CNODE_869_length_18620_cov_133.907353_14', 'Actinomyces']

['S5CNODE_889_length_19238_cov_56.809050_10', 'Capnocytophaga']


target_phage = ['S3CNODE_869_length_18620_cov_133.907353_14', 'S5CNODE_889_length_19238_cov_56.809050_10']
out_file = '../../../fig_3/fig_3e/probe_design/inputs/phage_terminase_large.fasta'
with open(out_file, 'w') as f:
    for record in SeqIO.parse(gene_input_fn, 'fasta'):
        if record.id in target_phage:
            SeqIO.write(record, f, 'fasta')



# %% codecell
from pylab import *
cmap = cm.get_cmap('tab20', 20)    # PiYG
hex = []
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    hex.append(matplotlib.colors.rgb2hex(rgba)[1:])
",".join(hex)
























# %% md

# Blast CAPSB gene against contigs

# Generate command then run in shell

# %% codecell
output_dir = '../../../../outputs/fig_5/metagenomic_analysis'
gene_input_fn = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/plaque/probe_design/2021_11_15_phage/inputs/active_prophage.fasta'
gene_basename = os.path.splitext(os.path.split(gene_input_fn)[1])[0]
contig_db_fn = output_dir + '/contig_blastdb/2021_04_03_contigs.fna'
gene_blast_fn = output_dir + '/' + gene_basename + '.fasta.blast.out'
out_format = ('"6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids"')

command = " ".join([
        'blastn',
        '-db', contig_db_fn,
        '-query', gene_input_fn,
        '-out', gene_blast_fn,
        '-outfmt', out_format
        ])

run_fn = 'run_{}.sh'.format('blast_capsb')
with open(run_fn, 'w') as f:
    f.write(command)

# %% md

# Check blast results

# %% codecell
# get blast results
gene_blast = pd.read_csv(gene_blast_fn, sep='\t')
gene_blast.columns = [out_format.split()[1:]]
gene_blast_ids = gene_blast[['qseqid','sseqid']].values
gene_blast.columns
alns = gene_blast[['bitscore', 'evalue', 'gapopen', 'length', 'mismatch', 'pident', 'qcovhsp']].values
gene_blast_ids, alns

# %% md

# Search for blast result contigs in binning

# %% codecell
# Get dict for contig to bin assignment
sample_key_fn = '../../../../data/fig_5/metagenomic_sequencing/sample_key.csv'
sample_key = pd.read_csv(sample_key_fn)
dict_assm_pacb = {a:p for p,a in sample_key.values}
dict_contig_bin = {}
bin_dir = output_dir + '/bins'
bin_fns = glob.glob(bin_dir + '/*.fna')
for fn in bin_fns:
    bin_id = os.path.splitext(os.path.split(fn)[1])[0]
    for record in SeqIO.parse(fn, 'fasta'):
        assm_id = re.search(r'^S\dC', record.id).group(0)
        pacb_id = dict_assm_pacb[assm_id]
        cont_id = re.sub(assm_id + 'NODE', pacb_id, record.id)  # Convert assembly id to pacbio id, which is used in the blast db
        dict_contig_bin[cont_id] = bin_id

# %% codecell
# Get dict for bin to family/genus/species assignment
sample_key_fn = '../../../../data/fig_5/metagenomic_sequencing/sample_key.csv'
sample_key = pd.read_csv(sample_key_fn)
tax_cols = ['bin_ID', 'kgenus', 'ggenus']
dict_bin_tax = {}
for r in sum_tab[tax_cols].values:
    dict_bin_tax[r[0]] = r[1:].tolist()


# %% codecell
# convert contig alignment to familygenus species
dict_phage_tax = defaultdict(list)
for phage, contig in gene_blast_ids:
    try:
        bin_ = dict_contig_bin[contig]
        dict_phage_tax[phage].append(dict_bin_tax[bin_])
    except:
        pass

len(dict_phage_tax)

# %% codecell
dict_phage_tax_unq = {}
for k, v in dict_phage_tax.items():
    cov = float(re.search('(?<=cov_)\d+.\d+', k).group(0))
    dict_phage_tax_unq[cov] = [k, np.unique(v, return_counts=True)]

dict(sorted(dict_phage_tax_unq.items()))
