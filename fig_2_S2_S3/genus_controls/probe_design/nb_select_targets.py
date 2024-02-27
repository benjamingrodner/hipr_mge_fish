# %% md

# # Figure 5c: Phage TermL2 associations with metagenome

# Used "hiprfish" conda environment

# Go through the metagenomic data and select target genes that are genus specific

# I want to probe the genus with hiprfish probes as a ground truth

# Then we can compare with the MeGA-FISH signal.

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
# ## Get streptococcus assembly
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
target_taxa = ['Streptococcus']
tax_sum_list = []
for tax in target_taxa:
    bool_k = sum_tab['ggenus'] == 'g__' + tax
    bool_g = sum_tab['kgenus'] == tax
    sub = sum_tab.loc[bool_k | bool_g, :]
    tax_sum_list.append(sub)
tax_sum_df = pd.concat(tax_sum_list)
tax_sum_df.shape

# %% md

# sort table

# %% codecell
tax_sum_df_sort = tax_sum_df.sort_values(by=['checkm_%abundance_community','checkm_%complete'], ascending=[False, False])
tax_sum_df_sort

# %% md

# Annotate genes in strep

# %% codecell
# Extract one of the bins
index = 484
bin_id, tax = tax_sum_df_sort.loc[index,['bin_ID','kgenus']]
bin_id

# %% codecell
# Rename the contigs since they are too long for prokka
bin_dir = '../../../../outputs/fig_5/metagenomic_analysis/bins'
bin_fasta_fn = bin_dir + '/' + bin_id + '.fna'
output_dir = '../../../../outputs/fig_3/fig_3e/probe_design'
bin_fasta_rename_fn = output_dir + '/' + bin_id + '_rename.fna'
dict_prok_assm = {}
record_new = []
for i, record in enumerate(SeqIO.parse(bin_fasta_fn, 'fasta')):
    new_name = bin_id + '_contig_' + str(i+1)
    dict_prok_assm[new_name] = record.id
    record_new.append(SeqRecord(record.seq, id=new_name))
SeqIO.write(record_new, bin_fasta_rename_fn, 'fasta')

# %% codecell
# write Prokka
prokka_dir = output_dir + '/prokka/' + bin_id
command = " ".join([
        'prokka',
        '-outdir', prokka_dir,
        '-prefix', bin_id,
        bin_fasta_rename_fn
        ])

run_fn = 'run_{}.sh'.format('prokka_strep')
with open(run_fn, 'w') as f:
    f.write(command)

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate prokka
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3e/probe_design
# $ sh run_prokka_strep.sh
# ```

# %% md

# Look through genes and pick a few

# %% codecell
prok_tsv_fn = prokka_dir + '/' + bin_id + '.tsv'
prok_tsv = pd.read_csv(prok_tsv_fn, sep='\t')

prok_tsv_filt = prok_tsv[prok_tsv['product'] != 'hypothetical protein']

potential = []
for i, r in prok_tsv_filt.iterrows():
    p = str(r['product'])
    if 'neur' in p:
        potential.append(r)

pot_df = pd.DataFrame(potential)
pot_df

# %% codecell
chosen_gene_index = 728

gene, desc = pot_df.loc[chosen_gene_index,['gene', 'product']]
locus_tag = pot_df.loc[chosen_gene_index,'locus_tag']
with open('/fs/cbsuvlaminck2//workdir/bmg224/manuscripts/mgefish/outputs/fig_3/fig_3e/probe_design/prokka/S5C507/S5C507.gff','r') as f:
    for line in f:
        if locus_tag in line:
            chosen_line = line.split('\t')

chosen_line

# %% codecell
chosen_contig_name = chosen_line[0]
for record in SeqIO.parse(bin_fasta_fn, 'fasta'):
    id = dict_prok_assm[chosen_contig_name]
    if record.id == id:
        chosen_contig = record

print(chosen_contig)
# %% codecell
st, en, strand = int(chosen_line[3]), int(chosen_line[4]), chosen_line[6]
chosen_gene_seg = chosen_contig.seq[st:en+1]
chosen_gene_seg = chosen_gene_seg.reverse_complement() if strand == '-' else chosen_gene_seg
len(chosen_gene_seg)

# %% codecell
out_file = 'inputs/' + gene + '_' + tax + '.fasta'
with open(out_file, 'w') as f:
    SeqIO.write(SeqRecord(chosen_gene_seg, id=gene, description=desc + ';' + tax + ';bin_id=' + bin_id), f, 'fasta')





for i in range(5):
    if i > 2:
        break
print(i)










# %% md

# Search for target genuses

# %% codecell
target_taxa = ['Corynebacterium']
tax_sum_list = []
for tax in target_taxa:
    bool_k = sum_tab['ggenus'] == 'g__' + tax
    bool_g = sum_tab['kgenus'] == tax
    sub = sum_tab.loc[bool_k | bool_g, :]
    tax_sum_list.append(sub)
tax_sum_df = pd.concat(tax_sum_list)
tax_sum_df.shape

# %% md

# sort table

# %% codecell
tax_sum_df_sort = tax_sum_df.sort_values(by=['checkm_%abundance_community','checkm_%complete'], ascending=[False, False])
tax_sum_df_sort

# %% md

# Annotate genes in strep

# %% codecell
# Extract one of the bins
index = 494
bin_id, tax = tax_sum_df_sort.loc[index,['bin_ID','kgenus']]
bin_id

# %% codecell
# Rename the contigs since they are too long for prokka
bin_dir = '../../../../outputs/fig_5/metagenomic_analysis/bins'
bin_fasta_fn = bin_dir + '/' + bin_id + '.fna'
output_dir = '../../../../outputs/fig_3/fig_3e/probe_design'
bin_fasta_rename_fn = output_dir + '/' + bin_id + '_rename.fna'
dict_prok_assm = {}
record_new = []
for i, record in enumerate(SeqIO.parse(bin_fasta_fn, 'fasta')):
    new_name = bin_id + '_contig_' + str(i+1)
    dict_prok_assm[new_name] = record.id
    record_new.append(SeqRecord(record.seq, id=new_name))
SeqIO.write(record_new, bin_fasta_rename_fn, 'fasta')

# %% codecell
# write Prokka
prokka_dir = output_dir + '/prokka/' + bin_id
command = " ".join([
        'prokka',
        '-outdir', prokka_dir,
        '-prefix', bin_id,
        bin_fasta_rename_fn
        ])

run_fn = 'run_{}.sh'.format('prokka_strep')
with open(run_fn, 'w') as f:
    f.write(command)

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate prokka
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3e/probe_design
# $ sh run_prokka_strep.sh
# ```

# %% md

# Look through genes and pick a few

# %% codecell
prok_tsv_fn = prokka_dir + '/' + bin_id + '.tsv'
prok_tsv = pd.read_csv(prok_tsv_fn, sep='\t')

prok_tsv_filt = prok_tsv[prok_tsv['product'] != 'hypothetical protein']

potential = []
for i, r in prok_tsv_filt.iterrows():
    p = str(r['product'])
    if 'lysine' in p:
        potential.append(r)

pot_df = pd.DataFrame(potential)
pot_df

# %% codecell
chosen_gene_index = 1659

gene, desc = pot_df.loc[chosen_gene_index,['gene', 'product']]
locus_tag = pot_df.loc[chosen_gene_index,'locus_tag']
gff_file = prokka_dir + '/' + bin_id + '.gff'
with open(gff_file,'r') as f:
    for line in f:
        if locus_tag in line:
            chosen_line = line.split('\t')

chosen_line

# %% codecell
chosen_contig_name = chosen_line[0]
for record in SeqIO.parse(bin_fasta_fn, 'fasta'):
    id = dict_prok_assm[chosen_contig_name]
    if record.id == id:
        chosen_contig = record

print(chosen_contig)
# %% codecell
st, en, strand = int(chosen_line[3]), int(chosen_line[4]), chosen_line[6]
chosen_gene_seg = chosen_contig.seq[st:en+1]
chosen_gene_seg = chosen_gene_seg.reverse_complement() if strand == '-' else chosen_gene_seg
len(chosen_gene_seg)

# %% codecell
out_file = 'inputs/' + gene + '_' + tax + '.fasta'
with open(out_file, 'w') as f:
    SeqIO.write(SeqRecord(chosen_gene_seg, id=gene, description=desc + ';' + tax + ';bin_id=' + bin_id), f, 'fasta')















# %% md

# Design probes for target genes

# =============================================================================
# ## Blast welch et al 2016 probes for rRNA probes against pacbio database
# =============================================================================

# Run blast

# %% codecell
# Load blast functions
sys.path.append('/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/probe_design/split_probe_design/scripts')
import blast_functions as bf

# %% codecell
# Get the database
db_fn = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_3/fig_3e/probe_design/pacbio_database/PBG-3104.plaque_mk_m_20210301_.CCS2.0000006459_plus_pstrip_filt_unique.fasta'
# Get the probes
welch_probes_fn = 'inputs/welch_etal_2016_probes.fasta'
# blast the probes
out_dir_blast = output_dir + '/blast'
if not os.path.exists(out_dir_blast): os.makedirs(out_dir_blast)
bf.blastProbes(welch_probes_fn, db_fn, out_dir_blast)

# %% md

# Get the ID of the alignments genus

# %% codecell
# Hash the usearch output table
map_fn = output_dir + '/usearch/PBG-3104.plaque_mk_m_20210301_.CCS2.0000006459_plus_pstrip_filt_unique_map.txt'
dict_pac_otu = {}
with open(map_fn) as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        l = line.split('\t')
        mol = l[0]
        dict_pac_otu[l[0]] = l[1][:-1]
        if i == 0: print(l[1][:-1])
        i+=1
        # t2otu = re.search('otu',l[1])
        # t2chim = re.search('chimera',l[1])
        # if t2otu:
        #     otu = [re.sub('o','O',l[1])]  # Capitalize to match naming in otu fasta
        # elif t2chim:
        #     otu = re.findall(r'Otu\d+', l[2])  # If its a chimera, get the parents
        # else:
        #     t3otu = re.search(r'Otu\d+', l[2])
        #     if t3otu:
        #         otu = [t3otu.group(0)]
        #     else:
        #         otu = [re.search(r'Chimera\d+', l[2])]
        # dict_pac_otu[mol] = otu

len(dict_pac_otu)

# %% codecell
# Hash the usearch output table
sintax_fn = output_dir + '/usearch/PBG-3104.plaque_mk_m_20210301_.CCS2.0000006459_plus_pstrip_filt_unique_otu.sintax'
dict_otu_genus = {}
with open(sintax_fn) as f:
    lines = f.readlines()
    for line in lines:
        l = line.split('\t')
        otu = l[0]
        t2genus = re.search(r'(?<=,g:)\w+',l[1])
        if t2genus:
            genus = t2genus.group(0)
        else:
            genus = 'Unassigned'
        dict_otu_genus[otu] = genus

len(dict_otu_genus)

# %% md
# Get taxon for each blast result

# Group blast results by genus

# Get the max homology for each genus and list

# %% codecell
bn = os.path.split(welch_probes_fn)[1]
blast_out_fn = out_dir_blast + '/' + bn + '.blast.out'
dict_probe_genus = defaultdict(lambda: defaultdict(dict))
with open(blast_out_fn) as f:
    lines = f.readlines()
    # i = 0
    for line in lines:
        l = line.split('\t')
        probe = l[0]
        evalue = int(l[4]) - int(l[5]) - int(l[6])
        pac = l[1]
        size = int(re.search('(?<=size=)\d+', pac).group(0))
        try:
            otu = dict_pac_otu[pac]
            genus = dict_otu_genus[otu]
        except KeyError:
            otu = 'Unassigned'
            genus = 'Unassigned'
        try:
            ev_curr = dict_probe_genus[probe][genus]['length']
            dict_probe_genus[probe][genus]['count'] += size
            if evalue > ev_curr:
                dict_probe_genus[probe][genus]['length'] = evalue
        except KeyError:
            dict_probe_genus[probe][genus]['length'] = evalue
            dict_probe_genus[probe][genus]['count'] = size

        # if i < 5: print(probe, pac, otu, genus)
        # i += 1

dict_probe_genus

# %% md
# =============================================================================
# Design split probes for select genuses in the pacbio library
# =============================================================================

# Get representative sequences for target genuses

# %% codecell
# Convert otu to pacbio list
dict_otu_pac = defaultdict(list)
with open(map_fn) as f:
    lines = f.readlines()
    for line in lines:
        l = line.split('\t')
        dict_otu_pac[l[1][:-1]].append(l[0])


# %% codecell
# Convert genus to otu
dict_genus_otu = defaultdict(list)
with open(sintax_fn) as f:
    lines = f.readlines()
    for line in lines:
        l = line.split('\t')
        otu = l[0]
        t2genus = re.search(r'(?<=,g:)\w+',l[1])
        if t2genus:
            genus = t2genus.group(0)
        else:
            genus = 'Unassigned'
        dict_genus_otu[genus].append(otu)

len(dict_genus_otu)

# %% codecell
# Get pacbio names for each genus
dict_genus_pac = {}
for genus, otus in dict_genus_otu.items():
    pac_list = []
    for otu in otus:
        pac_list += dict_otu_pac[otu]
    dict_genus_pac[genus] = pac_list

dict_genus_pac

# %% codecell
# Get pacbio name with the most repeated sequences for each genus
dict_genus_pacmax = {}
for genus, pacs in dict_genus_pac.items():
    size = 0
    for pac_ in pacs:
        size_ = int(re.search(r'(?<=size=)\d+', pac_).group(0))
        if size_ > size:
            size = size_
            pac = pac_
    dict_genus_pacmax[genus] = pac

dict_genus_pacmax

# %% md

# Select target genuses

# %% codecell
genus_list = ['Streptococcus','Veillonella', 'Corynebacterium']
pac_list = [dict_genus_pacmax[g] for g in genus_list]
records = []
for record in SeqIO.parse(db_fn, 'fasta'):
    if record.id in pac_list:
        records.append(record)

genus_fn = 'inputs/genus_rRNA_bmg_m.fasta'
SeqIO.write(records, genus_fn, 'fasta')

# %% md

# Set up the config file externally and Run probe design

# %% codecell
pipeline_dir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/probe_design/split_probe_design/Snakefile'
config_fn = 'config.yaml'
n_cores = 40
command = " ".join([
        'snakemake',
        '-s', pipeline_dir,
        '--configfile', config_fn,
        '-j', str(n_cores)
        ])

run_title = 'probe_design_rRNA'

run_fn = 'run_{}.sh'.format(run_title)
with open(run_fn, 'w') as f:
    f.write(command)

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3e/probe_design
# $ sh run_probe_design_rRNA.sh
# ```


# %% codecell
a = [2,3,1]
a.sort()
a

al_sstart, pr_send, al_send, pr_sstart = [1,0,1,0]
if not ((al_sstart >= pr_send) or (al_send <= pr_sstart)):
    ot = 1  # Blast is on target return 1


x = '44 128 373'.split()
x = [int(i) for i in x]

blast = [2+14/60, 4+15/60,13+19/60]

measure = [2+26/60, 5+25/60, 14+49/60]

import matplotlib.pyplot as plt
plt.plot(x, blast)
plt.plot(x, measure)

(blast[2] - blast[1]) / (x[2] - x[1]) * 60
370/60

a = defaultdict(lambda: defaultdict(list))

a['b']['c'] += [1]
a
l = ['a','b\n']
l[-1] = l[-1][:-1]  # Remove newline character
l


a['c']['d'] = 0
a
a['c']['d'] += 1
a
' '.join(['/programs/primer3-2.3.5/src/primer3_core', '-p3_settings_file', '../../../../outputs/fig_3/fig_3e/probe_design/design_probes/test_03/primer3_settings.txt', '-output', '../../../../outputs/fig_3/fig_3e/probe_design/design_probes/test_03/primer3_output.txt', '-format_output', '../../../../outputs/fig_3/fig_3e/probe_design/design_probes/test_03/primer3_input.txt'])

GGTAAGGTTC T TCGCG
GGTAAGGTTC C TCGCG

16/18

2.5e6 / (20*5000)


# %% codecell
def off_target_02(sseqid, pr_sstart, pr_send, pr_sstrand, dict_aln_target):
    ot = 1  # If the blast is off target return 0
    try:  # See if the the blast is in the target alignment dict
        al_sstart, al_send, al_sstrand = dict_aln_target[sseqid]
        if al_sstrand == pr_sstrand:  # Check the strand of the probe blast
            al_sstart, al_send = sorted([al_sstart, al_send])
            pr_sstart, pr_send = sorted([pr_sstart, pr_send])
            # Check if the probe is aligned where the target aligns
            if not ((al_sstart >= pr_send) or (al_send <= pr_sstart)):
                ot = 0  # Blast is on target return 1
    except KeyError:
        pass
    return ot


sseqid, pr_sstart, pr_send, pr_sstrand = 'a', 20, 40, 'minus'
dict_aln_target = {}
dict_aln_target['a'] = [1,1500, 'minus']
off_target_02(sseqid, pr_sstart, pr_send, pr_sstrand, dict_aln_target)


(214572764-144974235) / 214572764
# %% codecell
pid = '1107'
blasts = []
with open('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_3/fig_3e/probe_design/genus_rRNA_bmg_m/pident_70_qcovhsp_70/mch_15_gc_8/filter_blasts/bmg_m.18;size=1253;_probe_blast_filtered.csv','r') as f:
    for line in f:
        l = line.split('\t')
        l[-1] = l[-1][:-1]  # Remove newline character
        if l[0] == pid:
            blasts.append(l)

blasts_df = pd.DataFrame(blasts)
blasts_df
# %% codecell
import yaml
with open('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_3/fig_3e/probe_design/genus_rRNA_bmg_m/pident_70_qcovhsp_70/filter_target_alns/bmg_m.18;size=1253;_dict_aln_target.yaml','r') as f:
    dict_aln_target = yaml.safe_load(f)

for aln in blasts_df.iloc[:,1].values:
    if aln in dict_aln_target:
        print(aln)

dict_aln_target


# %% md
# =============================================================================
# Take probes from primer blast from ncbi and convert them to idt format
# =============================================================================

# reverse complement half the sequences

# %% codecell
rc_key = '.R'
flanking = {'L': ' tat ATCATCCAgTAAACCgCC', 'R':'CCTCgTAAATCCTCATCA tat '}
probe_fn = '/fs/cbsuvlaminck2//workdir/bmg224/manuscripts/mgefish/outputs/fig_3/fig_3e/probe_design/MGEFISH/e_coli/primerblast_Ec_codA_Lp_Ppase.fasta'
ids = []
seqs = []
for r in SeqIO.parse(probe_fn, 'fasta'):
    ids.append(r.id)
    if rc_key in r.id:
        s = flanking['R'] + str(r.seq.reverse_complement())
    else:
        s = str(r.seq) + flanking['L']
    seqs.append(s)

print(seqs)

# %% codecell
idt_df = pd.DataFrame()
idt_df['Name'] = ids
idt_df['Sequence'] = [str(s) for s in seqs]
idt_df['Scale'] = '25nm'
idt_df['Purification'] = 'STD'
idt_df

# %% codecell
out_fn = re.sub('.fasta','.csv',probe_fn)
idt_df.to_csv(out_fn, index=False)
