import numpy as np
import pandas as pd
import os
import statsmodels.api as sm

def save_gene_state(df_burst_common_final, \
                    mRNA_decay, copy_nums, \
                    burst_freq, file_path):
    ngenes_ = df_burst_common_final.shape[0]
    mRNA_decay = np.ones(df_burst_common_final.shape[0])
    df_gene_state_file.to_csv(file_path, header=None, index=None)

def save_burst_size(df_burst_common_final, \
                    mRNA_decay, copy_nums, \
                    burst_freq, file_path):
    ngenes_ = df_burst_common_final.shape[0]
    mRNA_decay = np.ones(df_burst_common_final.shape[0])
    df_gene_state_file = \
        pd.DataFrame(data={"gene" : df_burst_common_final.index.values, \
                        "gene_type" : ["two-state reduced mRNA"] * ngenes_, \
                        "rxn_1" : ["transcription"] * ngenes_, 
                        "rxn_rate_1" : burst_freq * copy_nums * mRNA_decay, \
                        "rxn_2" : ["mRNA decay"] * ngenes_, \
                        "rxn_rate_2" : mRNA_decay})
    df_gene_state_file = df_gene_state_file.drop_duplicates(subset=["gene"])
    df_gene_state_file.to_csv(file_path, header=None, index=None)

def ttest_slope(x, y, slope_0=1, num_bootstraps=10000):
    # Perform bootstrap resampling
    # num_bootstraps = 1000
    bootstrap_slopes = np.zeros(num_bootstraps)

    for i in range(num_bootstraps):
        indices = np.random.choice(len(x), len(x), replace=True)
        bootstrap_x = x[indices]
        bootstrap_y = y[indices]
        model = sm.OLS(bootstrap_y, bootstrap_x)
        result = model.fit()
        bootstrap_slope, _, _, _, _ = linregress(bootstrap_x, bootstrap_y)
        bootstrap_slopes[i] = bootstrap_slope


def get_burst_kinetics_mESC(cur_dir):
    dir_path = os.path.abspath(os.path.join(cur_dir, "data", \
                                            "larsson_et_al_2019"))

    # cast allele
    file_path = os.path.abspath(os.path.join(dir_path, \
                                            f'41586_2018_836_MOESM5_ESM.xlsx'))
    df_burst = pd.read_excel(file_path, sheet_name="CAST", index_col=0)

    # get mRNA half-life
    dir_path = cur_dir
    file_path = os.path.join(dir_path, "data", \
                        "scRNA_seq", "sharova_etal_2009", \
                        "halfLife_hr.csv")
    half_life = pd.read_csv(file_path, header=None, index_col=1)
    half_life.columns = ['half_life']
    genes_common = \
        list(set(list(half_life.index.values)).intersection(set(list(df_burst.index.values))))
    df_half_common = half_life.copy()
    df_half_common = df_half_common.loc[genes_common, :]

    df_burst_common = df_burst.copy()
    df_burst_common = df_burst_common.loc[genes_common, :]

    return df_burst_common, df_half_common