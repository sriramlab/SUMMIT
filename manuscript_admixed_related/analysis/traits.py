"""The 40 UK Biobank traits of the SUMMIT paper, age and sex covariates, and origin covariates, for given samples.

Input: a UK Biobank table with the column f.eid and columns f.<field>.<instance>.<array> for the fields below (and 31, 34;
for --origin also 54, 1647, 3659, 20115, 21000). Continuous traits use the first non-missing measurement (instance, then
array, in increasing order) after removing negative special codes, and are inverse rank-normal transformed within each
sample. Grip strength is the mean of the first left and right readings; waist-to-hip ratio removes |z| > 6 within sex;
educational attainment maps qualifications (6138) to years and takes the maximum. Binary traits (ever smoked; first
occurrence of I25, E78, I10, E11) are coded 1.25 for cases and 0.25 for controls, so that PLINK 2 fits a linear model;
their sample prevalence is written for the liability scale.
Origin covariates: birth country unknown; for participants born abroad, age at immigration in bins (<12, 12-17, 18-29,
>=30, unknown) and birth countries with at least min(50, 1% of the sample) participants; self-reported ethnic background
categories with at least 20 participants; assessment centres with at least 100 participants (the rest pooled when at least
20). Columns with fewer than 20 ones or zeros in either sample, or adding no rank to the base covariates, are dropped.
Writes OUT/pheno_<sample>.tsv, OUT/agesex_<sample>.tsv, OUT/origin_<sample>.tsv (with --origin) and OUT/prevalence.tsv.
usage: python traits.py --ukb TABLE --sample NAME=FAM [--sample NAME=FAM ...] --out DIR [--origin]
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import irnt

CONTINUOUS = {   # trait: (field, extra missing codes)
    'albumin': (30600, ()), 'alka_phos': (30610, ()), 'alanine_aminotransferase': (30620, ()), 'asp_at': (30650, ()),
    'creatinine': (30700, ()), 'cystatin_c': (30720, ()), 'ggt': (30730, ()), 'glucose': (30740, ()), 'hba1c': (30750, ()),
    'hdl': (30760, ()), 'igf_1': (30770, ()), 'ldl': (30780, ()), 'testosterone': (30850, ()), 'triglycerides': (30870, ()),
    'urate': (30880, ()), 'vitamin_d': (30890, ()), 'blood_rbc_count': (30010, ()), 'white_blood_cell_count': (30000, ()),
    'blood_platelet': (30080, ()), 'blood_mscv': (30270, ()), 'haemoglobin_concentration': (30020, ()), 'height': (50, ()),
    'bmi': (21001, ()), 'bp_systolic': (4080, ()), 'diastolic_blood_pressure': (4079, ()),
    'heel_bone_mineral_density': (3148, ()), 'calcium': (30680, ()), 'urea': (30670, ()), 'apo_b': (30640, ()),
    'fluid_intelligence_score': (20016, (-1,)), 'neuroticism_score': (20127, ()), 'sleep_duration': (1160, (-1, -3)),
}
FIRST_OCCURRENCE = {'chronic_ischaemic_heart_disease_i25': 131306, 'dyslipidaemia_e78': 130814, 'hypertension_i10': 131286,
                    'type2_diabetes_e11': 130708}
BINARY = ['ever_smoked'] + list(FIRST_OCCURRENCE)
TRAITS = list(CONTINUOUS) + ['educational_qualification', 'hand_grip_strength', 'whr'] + BINARY
EDU_YEARS = {1: 20, 2: 13, 3: 12, 4: 12, 5: 19, 6: 15, -7: 6}            # 6138 qualification -> years of education
PLACEHOLDER_DATES = {'1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03', '1909-09-09', '2037-07-07'}


def columns(df, field):
    cols = [c for c in df.columns if c.startswith(f'f.{field}.')]
    return sorted(cols, key=lambda c: tuple(int(x) for x in c.split('.')[2:]))


def first_value(df, field, missing=()):
    x = df[columns(df, field)].apply(pd.to_numeric, errors='coerce')
    x = x.mask(x.isin([-9, *missing]))
    return x.bfill(axis=1).iloc[:, 0]


def raw_traits(df):
    out = {t: first_value(df, f, miss) for t, (f, miss) in CONTINUOUS.items()}
    q = df[columns(df, 6138)].apply(pd.to_numeric, errors='coerce')
    out['educational_qualification'] = q.where(q.isin(list(EDU_YEARS))).replace(EDU_YEARS).max(axis=1)
    left, right = (pd.to_numeric(df[columns(df, f)[0]], errors='coerce') for f in (46, 47))
    out['hand_grip_strength'] = pd.concat([left, right], axis=1).mean(axis=1)
    pos = lambda f: df[columns(df, f)].apply(pd.to_numeric, errors='coerce').where(lambda x: x > 0).bfill(axis=1).iloc[:, 0]
    out['whr'] = pos(48) / pos(49)
    smk = df[columns(df, 20160)].apply(pd.to_numeric, errors='coerce')
    out['ever_smoked'] = pd.Series(np.where((smk == 1).any(axis=1), 1.0, np.where((smk == 0).any(axis=1), 0.0, np.nan)),
                                   index=df.index)
    for t, f in FIRST_OCCURRENCE.items():
        d = df[columns(df, f)].astype('string').apply(lambda c: c.str.strip())
        placeholder = d.isin(PLACEHOLDER_DATES).any(axis=1)
        real = d.mask(d.isin(PLACEHOLDER_DATES)).apply(pd.to_datetime, errors='coerce').notna().any(axis=1)
        out[t] = pd.Series(np.where(real, 1.0, np.where(placeholder | d.notna().any(axis=1), np.nan, 0.0)), index=df.index)
    return pd.DataFrame(out)


def whr_qc(whr, sex):
    whr = whr.copy()
    for s in (0, 1):
        m = (sex == s) & whr.notna()
        z = (whr - whr[m].mean()) / whr[m].std(ddof=0)
        whr[m & (z.abs() > 6)] = np.nan
    return whr


def origin(df, base, samples):
    """Origin indicators (see the module docstring), filtered on every sample."""
    f = {k: df[columns(df, k)[0]] if columns(df, k) else pd.Series(np.nan, index=df.index) for k in (34, 54, 1647, 3659, 20115, 21000)}
    abroad = f[1647].astype(str) == '6'
    X = {'cob_unknown': ~f[1647].astype(str).isin([str(k) for k in range(1, 7)])}
    age_imm = pd.to_numeric(f[3659], errors='coerce').where(lambda v: v > 0) - pd.to_numeric(f[34], errors='coerce')
    for lab, lo, hi in (('imm_lt12', -np.inf, 12), ('imm_12_17', 12, 18), ('imm_18_29', 18, 30), ('imm_ge30', 30, np.inf)):
        X[lab] = abroad & (age_imm >= lo) & (age_imm < hi)
    X['imm_unknown'] = abroad & age_imm.isna()
    n = len(df)
    valid = lambda v: v.notna() & ~v.astype(str).str.startswith('-')        # missing and negative special codes
    cob = f[20115].astype(str)
    for code, cnt in cob[abroad & valid(f[20115])].value_counts().items():
        if cnt >= min(50, 0.01 * n):
            X[f'cob_{code}'] = abroad & (cob == code)
    eth = f[21000].astype(str)
    for code, cnt in eth[valid(f[21000])].value_counts().iloc[1:].items():
        if cnt >= 20:
            X[f'eth_{code}'] = eth == code
    centre = f[54].astype(str)
    counts = centre[valid(f[54])].value_counts()
    big = [c for c, k in counts.iloc[1:].items() if k >= 100]
    for code in big:
        X[f'centre_{code}'] = centre == code
    rest = ~centre.isin(big + [counts.index[0]])
    if rest.sum() >= 20:
        X['centre_other'] = rest
    X = pd.DataFrame(X).astype(float)
    keep = [c for c in X if all(20 <= X.loc[s, c].sum() <= len(s) - 20 for s in samples.values())]
    X = X[keep]
    ids = list(samples.values())[0]
    B = np.column_stack([np.ones(len(ids)), base.loc[ids].to_numpy(float)])
    kept = []
    for c in X:                                         # keep columns that add rank to the base covariates
        Z = np.column_stack([B] + [X.loc[ids, k].to_numpy() for k in kept + [c]])
        if np.linalg.matrix_rank(Z) == Z.shape[1]:
            kept.append(c)
    return X[kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ukb', required=True)
    ap.add_argument('--sample', action='append', required=True, help='NAME=FAM (repeatable); traits are transformed within each')
    ap.add_argument('--out', required=True)
    ap.add_argument('--origin', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.ukb, sep='\t', dtype=str, keep_default_na=False, na_values=['', 'NA']).set_index('f.eid')
    samples = {}
    for spec in a.sample:
        name, fam = spec.split('=', 1)
        samples[name] = pd.read_csv(fam, sep=r'\s+', header=None, usecols=[1], dtype=str)[1].tolist()
    raw = raw_traits(df)
    sex = pd.to_numeric(df['f.31.0.0'], errors='coerce')
    age = 2023.0 - pd.to_numeric(df['f.34.0.0'], errors='coerce')
    prev = []
    for name, ids in samples.items():
        P = pd.DataFrame({'FID': ids, 'IID': ids})
        for t in TRAITS:
            x = (whr_qc(raw[t].reindex(ids), sex.reindex(ids)) if t == 'whr' else raw[t].reindex(ids)).to_numpy(float)
            if t in BINARY:
                prev.append(dict(sample=name, trait=t, n=int(np.isfinite(x).sum()), prevalence=float(np.nanmean(x))))
                P[t] = x + 0.25
            else:
                P[t] = irnt(x)
        P.to_csv(f'{a.out}/pheno_{name}.tsv', sep='\t', index=False, na_rep='NA', float_format='%.8g')
        A = pd.DataFrame({'FID': ids, 'IID': ids, 'age': age.reindex(ids).to_numpy() - age.reindex(ids).mean(),
                          'sex': sex.reindex(ids).to_numpy() + 1})
        A['age_sex'], A['age2'] = A.age * A.sex, A.age ** 2
        A['age2_sex'] = A.age2 * A.sex
        assert A.notna().all().all(), f'missing age or sex in {name}'
        A.to_csv(f'{a.out}/agesex_{name}.tsv', sep='\t', index=False, float_format='%.10g')
    pd.DataFrame(prev).to_csv(f'{a.out}/prevalence.tsv', sep='\t', index=False, float_format='%.6g')
    if a.origin:
        base = pd.read_csv(f'{a.out}/agesex_{list(samples)[0]}.tsv', sep='\t', dtype={'IID': str}).set_index('IID').drop(columns='FID')
        X = origin(df, base, samples)
        for name, ids in samples.items():
            O = X.loc[ids].reset_index(drop=True)
            O.insert(0, 'IID', ids)
            O.insert(0, 'FID', ids)
            O.to_csv(f'{a.out}/origin_{name}.tsv', sep='\t', index=False, float_format='%.0f')
        print(f'{X.shape[1]} origin columns kept')


if __name__ == '__main__':
    main()
