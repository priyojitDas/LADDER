import pandas as pd
import numpy as np
import random
import gzip
import pyBigWig as pbw
from torch.utils.data import Dataset

class GenomicDataset(Dataset):
    def __init__(self, datatype_root, 
                       genome_assembly,
                       feat_dicts, 
                       mode = 'train', 
                       include_sequence = True,
                       include_features = True,
                       use_aug = True):
        self.data_root = datatype_root
        self.include_sequence = include_sequence
        self.include_features = include_features
        self.use_aug = use_aug

        if mode != 'train': self.use_aug = False

        if genome_assembly.startswith('hg'):
            self.chromosomes = ['chr11','chr15','chr18']#['chr'+str(chrm) for chrm in range(1,23)]
        elif genome_assembly.startswith('mm'):
            self.chromosomes = ['chr'+str(chrm) for chrm in range(1,20)]
        if mode == 'train':
            self.chromosomes.remove('chr11')
            self.chromosomes.remove('chr15')
        elif mode == 'val':
            self.chromosomes = ['chr11']
        elif mode == 'test':
            self.chromosomes = ['chr15']
        else:
            pass

        print(self.chromosomes)
        
        self.features = [GenomicFeature('%s/genomic_features/%s' % (datatype_root, feat['file_name']), feat['norm']) for feat in list(feat_dicts.values())]
        self.ctgaps_dict = self.locate_ctgaps('%s/centrotelo.bed' % (datatype_root))
        self.chromosome_dict = [ChromosomeDataset(self.data_root, chrm, self.ctgaps_dict[chrm], self.features, self.use_aug) for chrm in self.chromosomes]
        self.chromosome_dict = dict(zip(self.chromosomes,self.chromosome_dict))
        self.lengths = [len(self.chromosome_dict[chrm]) for chrm in self.chromosomes]
        self.ranges = []
        current = 0
        for length in self.lengths:
            self.ranges.append([current, current + length - 1])
            current += length
        
    def __getitem__(self, idx):
        chrm, chrm_id = self.get_chr_idx(idx)
        seq, features, lad, start, end = self.chromosome_dict[chrm][chrm_id]
        outputs = seq, features, lad, start, end, chrm, chrm_id
        return outputs

    def __len__(self):
        return sum(self.lengths)      

    def get_chr_idx(self, idx):
        for i, chrm_range in enumerate(self.ranges):
            start, end = chrm_range
            if start <= idx <= end:
                return self.chromosomes[i], idx - start

    def locate_ctgaps(self, bed):
        df = pd.read_csv(bed , sep = '\t', header=None)
        ctgaps_dict = {}
        for chrm in self.chromosomes:
            tdf = df[df[0] == chrm]
            regions = tdf.iloc[:,[1,2]].values
            ctgaps_dict[chrm] = regions
        return ctgaps_dict

class ChromosomeDataset(Dataset):
    
    def __init__(self, datatype_root, chrm, omit_regions, feature_list, use_aug = True):
        self.use_aug = use_aug
        self.resolution = 10000
        self.wlen = 524288
        self.sbins = 500
        self.stride = 50
        self.chrm = chrm

        print('Chromosome DataLoader %s...' % (chrm))

        self.seq = SequenceFeature(path = '%s/dna_sequence/%s.fa.gz' % (datatype_root, self.chrm),t_chr=self.chrm)
        self.features = feature_list
        self.lad = LADFeature(path = '%s/lad_features/lad.bw' % (datatype_root))

        self.omit_regions = omit_regions
        self.check_length()
        
        genomic_pos = np.arange(((len(self.seq) / self.resolution) - self.sbins) / self.stride).reshape(-1,1) * self.stride
        self.all_intervals = np.append(genomic_pos, genomic_pos + self.sbins, axis=1)
        self.all_intervals = np.array(self.all_intervals * self.resolution, dtype='int')

        self.intervals = self.filter(self.all_intervals, omit_regions)

    def __getitem__(self, idx):
        s_, e_ = self.intervals[idx]
        t_size = self.wlen

        if self.use_aug: 
            extra = random.choice(range(e_ - s_ - t_size))
            s_, e_ = s_+extra, s_+extra+t_size
        else:
            s_, e_ = s_, s_+t_size
        
        seq = self.seq.get(s_, e_)
        features = [item.get(self.chrm, s_, e_) for item in self.features]
        lad = self.lad.get(self.chrm, s_)

        if self.use_aug:
            seq = seq + np.random.randn(seq.shape[0],seq.shape[1]) * 0.1
            features = [item + np.random.randn(item.shape[0]) * 0.1 for item in features]
            
            if np.random.rand(1) < 0.5:
                seq = np.flip(seq, 0).copy()
                features = [np.flip(item, 0).copy() for item in features]
                lad = np.flip(lad,0).copy()
                
                if np.random.rand(1) < 0.5:
                    seq = np.concatenate([seq[:, 1:2],
                                               seq[:, 0:1],
                                               seq[:, 3:4],
                                               seq[:, 2:3],
                                               seq[:, 4:5]], axis = 1)
        return seq, features, lad, s_, e_

    def __len__(self):
        return len(self.intervals)

    def filter(self, intervals, omit_regions):
        valid_intervals = []
        for start, end in intervals: 
            start_cond = start <= omit_regions[:, 1]
            end_cond = omit_regions[:, 0] <= end
            if sum(start_cond * end_cond) == 0:
                valid_intervals.append([start, end])
        return valid_intervals
    
    def check_length(self):
        assert len(self.seq) == self.features[0].length(self.chrm), f'Sequence {len(self.seq)} and First feature {self.features[0].length(self.chr_name)} have different length.' 
        assert len(self.seq) == self.lad.length(self.chrm), f'Sequence {len(self.seq)} and LAD {self.lad.length(self.chr_name)} have different length.' 

class LADFeature():
    
    def __init__(self, path):
        self.path = path

    def get(self, chrm, s_, window = 524288, res = 10000):
        e_ = s_ + window
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
        signals = np.array(np.array(signals) >= 0.75,dtype='float')
        return signals

    def load_lad(self, path):
        print('Reading LAD Data: %s' % (path))
        bw_file = pbw.open(path)
        return bw_file

    def length(self, chrm):
        with pbw.open(self.path) as bw_file:
            length = bw_file.chroms(chrm)
        return length

class GenomicFeature():

    def __init__(self, path, norm):
        self.path = path
        self.norm = norm
        self.line = {'a':3.20, 'g':3.02, 'c': 3.02, 't':3.20}
        self.sine = {'a':6.02, 'g':6.49, 'c': 6.49, 't':6.02}
        self.gene = {'a':0.55, 'g':0.59, 'c': 0.59, 't':0.55}
        print('Feature path: %s \n Normalization status: %s' % (path, norm))

    def get(self, chrm, s_, e_):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        return feature

    def getdel(self, chrm, s_, e_, pos_):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        feature = np.concatenate([feature[:ds_],feature[de_:]])
        return feature

    def getdelup(self, chrm, s_, e_, pos_):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        feature = feature[de_:]
        return feature

    def getdeldown(self, chrm, s_, e_, pos_):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        feature = feature[:ds_]
        return feature

    def getdelins(self, chrm, s_, e_, pos_, ftype):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if ftype == 'line':
            r_val = [*map(self.line.get, pos_[3].lower())]
        elif ftype == 'sine':
            r_val = [*map(self.sine.get, pos_[3].lower())]
        elif ftype == 'gene':
            r_val = [*map(self.gene.get, pos_[3].lower())]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        signals = np.concatenate((signals[:ds_],r_val,signals[de_:]))
        signals = signals[:524288]
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        return feature

    def getdelinsup(self, chrm, s_, e_, pos_, ftype):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if ftype == 'line':
            r_val = [*map(self.line.get, pos_[3].lower())]
        elif ftype == 'sine':
            r_val = [*map(self.sine.get, pos_[3].lower())]
        elif ftype == 'gene':
            r_val = [*map(self.gene.get, pos_[3].lower())]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        signals = np.concatenate((r_val,signals[len(pos_[2]):]))
        signals = signals[:524288]
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        return feature

    def getdelinsdown(self, chrm, s_, e_, pos_, ftype):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chrm, int(s_), int(e_))
            signals = np.nan_to_num(signals, 0)
        if ftype == 'line':
            r_val = [*map(self.line.get, pos_[3].lower())]
        elif ftype == 'sine':
            r_val = [*map(self.sine.get, pos_[3].lower())]
        elif ftype == 'gene':
            r_val = [*map(self.gene.get, pos_[3].lower())]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        signals = np.concatenate((signals[len(pos_[3]):524288],r_val))
        if self.norm == 'log':
            feature = np.log(np.array(signals) + 1)
        else:
            feature = signals
        return feature

    def length(self, chrm):
        with pbw.open(self.path) as bw_file:
            length = bw_file.chroms(chrm)
        return length

class SequenceFeature():

    def __init__(self, path = None, t_chr = None):
        self.path = path
        self.chrm = t_chr
        self.load()

    def load(self):
        self.seq = self.read_seq(self.path)

    def get(self, s_, e_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def getdel(self, s_, e_, pos_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        seq = seq[:ds_] + seq[de_:]
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def getdelup(self, s_, e_, pos_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        seq = seq[de_:]
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def getdeldown(self, s_, e_, pos_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        seq = seq[:ds_]
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def getdelins(self, s_, e_, pos_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        seq = seq[:ds_] + pos_[3].lower() + seq[de_:]
        seq = seq[:524288]
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def getdelinsup(self, s_, e_, pos_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        seq = pos_[3].lower() + seq[len(pos_[2]):]
        seq = seq[:524288]
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def getdelinsdown(self, s_, e_, pos_):
        base_encode = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        seq = self.seq[s_ : e_]
        ds_ = pos_[0] - s_
        de_ = ds_ + pos_[1] - pos_[0] + 1
        seq = seq[len(pos_[3]):524288] + pos_[3].lower()
        encoded_seq = np.array([base_encode[ch] for ch in seq], dtype='int')
        seq_5 = np.zeros((encoded_seq.shape[0],5))
        seq_5[np.arange(seq_5.shape[0]),encoded_seq] = 1
        return seq_5

    def __len__(self):
        chrLen = {'chrX': 155270560, 'chr13': 115169878, 'chr12': 133851895, 'chr11': 135006516, 'chr10': 135534747, 'chr17': 81195210, 'chr16': 90354753, 'chr15': 102531392, 'chr14': 107349540, 'chr19': 59128983, 'chr18': 78077248, 'chr22': 51304566, 'chr20': 63025520, 'chr21': 48129895, 'chr7': 159138663, 'chr6': 171115067, 'chr5': 180915260, 'chr4': 191154276, 'chr3': 198022430, 'chr2': 243199373, 'chr1': 249250621, 'chr9': 141213431, 'chr8': 146364022}
        return chrLen[self.chrm]

    def read_seq(self, dna_dir):
        print('Reading sequence: %s' % (dna_dir))
        with gzip.open(dna_dir, 'r') as f:
            seq = f.read().decode("utf-8")
        seq = seq[seq.find('\n'):]
        seq = seq.replace('\n', '').lower()
        return seq

if __name__ == '__main__':
    main()
