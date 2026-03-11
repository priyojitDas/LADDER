LADDER is a multimodal deep learning tool that predicts chromatin lamina association. It uses the DNA sequence with different genomic properties, including gene density, LINE1 density, and SINE density as input. The predictions made by LADDER are not cell-type specific in nature. LADDER also has the ability to predict the effect of different genomic structural rearrangements, such as deletions, on the lamina tethering of the surrounding region by performing _in silico_ genetic perturbations**.**

**Dependencies and Installation**

**Create environment**

Create a new conda environment for LADDER using the given _ladder_environment.yml_ file. But before that, please make sure you have already installed anaconda or mini-conda.

_conda env create -f ladder_environment.yml_

_conda activate laddermodel_

**Training model**

**Genomic features**

Bigwig files for the corresponding gene density, LINE1 and SINE densities are needed for LADDER training. To create the density files, you need to divide the corresponding reference genome into non-overlapping sliding windows of 10-kb size and then, for each window, count the number of overlapping genes or LINE1 elements or SINE elements using tileGenome and countOverlaps functions from GenomicRanges R package. These genome-wide count values at 10-kb resolution serve as densities.

**Lamina associated domain data**

Experimental cell-type non-specific lamina association data is needed for training. We recommend transforming the lamina association data to binary level (1: lamina associated and 0: non-lamina associated) and further converting to BigWig file.

**Data directory**

LADDER expects the input data to be structured in a specific fashion. For example, for human hg19 genome, the directory structure would be:

root_human/

\`-- hg19

|-- centrotelo.bed

|-- dna_sequence

| |-- chr1.fa.gz

| |-- chr10.fa.gz

| |-- chr11.fa.gz

| |-- chr12.fa.gz

| |-- chr13.fa.gz

| |-- chr14.fa.gz

| |-- chr15.fa.gz

| |-- chr16.fa.gz

| |-- chr17.fa.gz

| |-- chr18.fa.gz

| |-- chr19.fa.gz

| |-- chr2.fa.gz

| |-- chr20.fa.gz

| |-- chr21.fa.gz

| |-- chr22.fa.gz

| |-- chr3.fa.gz

| |-- chr4.fa.gz

| |-- chr5.fa.gz

| |-- chr6.fa.gz

| |-- chr7.fa.gz

| |-- chr8.fa.gz

| |-- chr9.fa.gz

| |-- chrX.fa.gz

|-- genomic_features

| |-- genedensity.bw

| |-- linedensity.bw

| \`-- sinedensity.bw

\`-- lad_features

\`-- lad.bw

Please make sure the data directory is structured as above with the assembly name 🡪 centrotelo.bed (this is a bed file of any regions you wish to exclude ex. telomeres and centromeres), dna_sequence, genomic_features and lad_features directories. The genomic_features contains the gene density, LINE1 density and SINE density bigwigs (files should have these exact same names!) specific to the genome. And finally, the lad_features directory has the cell-type non-specific lamina associated domain file for the genome.

**Training**

Use _LADDER/main.py_ to train a model. For example, to train the model for human hg19 genome, the command will be:

_python LADDER/main.py --output ./model_human --input ./root_human --assembly hg19 --gpu 4 --epochs 50_

\# --output: path of the model parameter output

\# --input: path of the model input data

\# --assembly: genome assembly version

\# --gpu: # of GPUs to use to train the model

\# --epochs: # of epochs

After training, use the _metrics.csv_ file located inside the _./model_human/csv/lightning_logs/version_0/_ directory to pick the best model.

The human hg19 reference genome training data: DNA sequence, gene density, LINE1 density, SINE density and lamina associated domain data is available under the _./root_human_ directory. The pretrained model weight (_epoch=112-step=67235.ckpt_) for hg19 genome can be downloaded from <https://drive.google.com/file/d/1l6AbO3sDmymWbwX7BL7QhNx6EvciVvlM/view?usp=sharing>.

**Deploy model**

**Generate chromosome-wide prediction**

1) First, use _LADDER/predict.py_ to predict base-level probabilities for lamina association for a specific chromosome broken into overlapping 512-kb windows with a step size of 200-kb. The command is as follow:

    _python LADDER/predict.py --chr chr11 --chrbins 676 --species human --assembly hg19 --model ./model_human/models/epoch\\=4-step\\=85.ckpt --genedensity ./root_human/hg19/genomic_features/genedensity.bw --linedensity ./root_human/hg19/genomic_features/linedensity.bw --sinedensity ./root_human/hg19/genomic_features/sinedensity.bw --seq ./root_human/hg19/dna_sequence --out ./prediction_human_

    \# --chr: chromosome to predict

    \# --chrbins: # of overlapping chromosomal windows (e.g., ⌈chromosome length / 200000⌉)

    \# --species: species

    \# --assembly: genome assembly version

    \# --model: path of the saved model

    \# --genedensity: path of the gene density file

    \# --linedensity: path of the LINE1 density file

    \# --sinedensity: path of the SINE density file

    \# --seq: path of the sequence fasta file

    \# --out: path of the model generated prediction files

2) Then, use _patchWindows.py_ to merge 512-kb sized predictions at the corresponding chromosomal locations to obtain chromosome-wide base-level probabilities.

    _python patchWindows.py prediction_output species assembly chr_

    \# prediction_output: base path of the directory where model generated prediction files are present (e.g., _./prediction_human_)

    \# species: species (e.g., human)

    \# assembly: genome assembly version (e.g., hg19)

    \# chr: chromosome to merge (e.g., chr11)

    The output file from step 2) can be found inside the prediction_output directory (for this case, _./prediction_human/npy/human/hg19_)

3) Finally, use _writeBED.py_ to convert chromosome-wide probabilities to binary lamina associated domain signal.

    _python writeBED.py prediction_output species assembly chr_

    \# prediction_output: base path of the directory where chromosome-wide base-level probabilities file is present (e.g., _./prediction_human_)

    \# species: species (e.g., human)

    \# assembly: genome assembly version (e.g., hg19)

    \# chr: chromosome to merge (e.g., chr11)

    The output files from step 3) can be found inside the _./bedfiles_ directory.

**_In-silico_ deletion**

1) Use _LADDER/predictdel.py_ to perform _in silico_ genetic deletion for a specific locus using LADDER model. The command is as follow:

    _python LADDER/predictdel.py --chr chr1 --species human --assembly hg19 --model ./model_human/models/epoch\\=4-step\\=85.ckpt --genedensity ./root_human/hg19/genomic_features/genedensity.bw --linedensity ./root_human/hg19/genomic_features/linedensity.bw --sinedensity ./root_human/hg19/genomic_features/sinedensity.bw --seq ./root_human/hg19/dna_sequence --location hg19.CPDEL.pos.n10.bed --out ./prediction_human_del_

    \# --chr: chromosome to predict

    \# --species: species

    \# --assembly: genome assembly version

    \# --model: path of the saved model

    \# --genedensity: path of the gene density file

    \# --linedensity: path of the LINE1 density file

    \# --sinedensity: path of the SINE density file

    \# --seq: path of the sequence fasta file

    \# --location: path to the bed file containing the deletion sites

    \# --out: path of the model generated prediction files

2) Then, to check its effect in upstream and downstream region, use _patchWindows_del.py_, which produces base-level lamina association probabilities around the deletion locus.

    _python patchWindows_del.py prediction_output species assembly chr location_

    \# prediction_output: base path of the directory where model generated prediction files are present (e.g., _./prediction_human_del_)

    \# species: species (e.g., human)

    \# assembly: genome assembly version (e.g., hg19)

    \# chr: chromosome (e.g., chr11)

    \# location: path to the bed file containing the deletion sites

     The output file from step 2) can be found inside the _./delfiles_ directory.

3) Finally, use _writedelBED.py_ to convert base-level lamina association probabilities to binary lamina associated domain signal.

    _python writedelBED.py prediction_output species assembly chr_

    \# prediction_output: base location of the directory where chromosome-wide base-level probabilities file is present (e.g., _./prediction_human_del_)

    \# species: species (e.g., human)

    \# assembly: genome assembly version (e.g., hg19)

    \# chr: chromosome (e.g., chr11)

    The output file from step 3) can be found inside the _./delfiles_ directory.
