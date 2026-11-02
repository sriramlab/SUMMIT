# All of Us cells for heritability due to direct genetic effects in siblings: run in order in one notebook of the Researcher
# Workbench after relatives_cells.py (stage 3 for the groups to analyse). D1 finds sibships and extended families, D2 computes
# the siblings' class GRMs, D3 checks the estimator on simulated phenotypes, D4 fits the real traits, and D5 exports aggregate
# statistics (counts of 1-20 masked; breakdowns whose totals would reveal masked counts are left out).
# ======================================================================================================================================
# F. Links. Run first in a new folder that holds direct_helpers.py and sib_direct.py: FOUNDATION as in relatives_cells.py, and
# RELATIVES the folder where relatives_cells.py ran (its rel_work/ and rel_results/ are read, never changed).
import os
FOUNDATION = os.path.expanduser('~/foundation')                 # set to the foundation folder
RELATIVES = os.path.expanduser('~/summit_relatives')            # set to the folder of relatives_cells.py
KIT = os.getcwd()   # this kit's folder: every later cell returns here first
for f, src in (('summit_work', FOUNDATION), ('v1', RELATIVES)):
    if os.path.islink(f) and os.path.realpath(f) != os.path.realpath(src):
        os.remove(f)
    if not os.path.lexists(f):
        os.symlink(src, f)
assert os.path.isdir('v1/rel_work'), 'RELATIVES has no rel_work/: run relatives_cells.py first'
# ======================================================================================================================================
# R1. Setup: parameters, helpers, and the relatives cells' environment (~/summit_rel_env, made by R1 of relatives_cells.py; nothing is installed).
# CHECK: "environment OK" with the package versions, and the CPU and memory line.
import os, json, glob, shutil, subprocess, time, sys, filecmp, collections, numpy as np, pandas as pd
os.chdir(KIT) if 'KIT' in dir() else None
assert os.path.islink('summit_work') and os.path.islink('v1'), f'{os.getcwd()} has no summit_work/v1 links: run cell F first'
KIT = os.getcwd()   # this kit's folder: every later cell returns here first
sys.path.insert(0, KIT)
import direct_helpers as AD
W, V1, R, OUT = 'summit_work', 'v1', 'dir_work', 'dir_results'   # foundation (linked, read only), v1 kit (linked, read only), this kit
SCR = os.path.expanduser('~/summit_direct_scratch')   # sibling genotypes and kernels: large, rebuilt by D1-D2
for d in (R, OUT, SCR): os.makedirs(d, exist_ok=True)
GROUPS = ['eur', 'afr', 'amr']
THREADS = os.cpu_count(); MEM_GIB = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30)
SIB_KIN = (0.177, 0.354)      # KING kinship of first-degree pairs (pruned SNPs, as in the relatives cells' R3)
IBS0_MIN = 0.0012             # IBS0 above which a first-degree pair is a full-sibling pair (KING; parent-offspring pairs have ~0)
REL_MIN = 0.0442              # GRM relatedness above which a pair is related (the relatives cells' families): extended families
MIN_SIB_MEMBERS = 500         # groups with fewer siblings (with every covariate) are skipped
MIN_TRAIT_MEMBERS = 200       # traits measured in fewer siblings are skipped
NBLOCKS = 200                 # SNP-block jackknife, as SUMMIT's default
NREP_SIM = 50                 # simulated phenotypes per setting (six settings)
SEED = 20261028
N_PCS = 10
COVS = ['age', 'sex', 'age_sex', 'age2', 'age2_sex', 'byear'] + [f'wPC{k}' for k in range(1, N_PCS + 1)]   # the relatives cells' covariates
AS_COLS = 'age,sex,age_sex,age2,age2_sex,byear'   # removed in the whole group before centering within sibships (the PCs are
                                                  # then shared by siblings and removed by the centering, as in UK Biobank)
SIM_TARGET = {'base': 0.4, 'sibenv': 0.4, 'extfam': 0.4, 'nurture': 0.4, 'ancestry': 0.4,
              'sibeffect': (0.4 ** 0.5 - 0.02 ** 0.5) ** 2}   # sibling effects: the direct estimator targets var(direct - sibling)
def sh(c, quiet=False, check=True):
    """Run a shell command and stream its output (quiet: print only the last lines)."""
    print('$', c[:600], flush=True); t0 = time.time()
    p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not quiet: print(line, end='', flush=True)
    p.wait()
    if quiet: print(''.join(lines[-12:]), end='')
    print(f'[{time.time() - t0:.0f} s, exit {p.returncode}]', flush=True)
    if check and p.returncode: raise RuntimeError(f'command failed: {c[:300]}')
    return ''.join(lines)
done = lambda *f: all(os.path.exists(x) and os.path.getsize(x) > 0 for x in f)
def fresh(out, *deps):
    """out exists and is at least as new as every input it was made from (otherwise it is rebuilt)."""
    return done(out) and all(os.path.getmtime(out) >= os.path.getmtime(d) for d in deps if d and os.path.exists(d))
def write_if_changed(df, path, **kw):
    """Write df, but keep the old file (and its time stamp) when the content is the same, so that later steps are not redone."""
    tmp = path + '.tmp'; df.to_csv(tmp, **kw)
    if os.path.exists(path) and filecmp.cmp(tmp, path, shallow=False): os.remove(tmp)
    else: os.replace(tmp, path)
M = AD.mask_count
ENV = os.path.expanduser('~/summit_rel_env'); PY = f'{ENV}/bin/python'; P2 = f'{ENV}/bin/plink2'
assert os.path.exists(PY) and os.path.exists(P2), '~/summit_rel_env is missing: run cell R1 of relatives_cells.py first'
sh(f'{PY} -c "import numpy, pandas, scipy, bed_reader; print(\'environment OK: numpy\', numpy.__version__, \'pandas\', pandas.__version__, \'scipy\', scipy.__version__)"')
BLAS = f'OMP_NUM_THREADS={THREADS} OPENBLAS_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS}'
SD = f'{BLAS} {PY} sib_direct.py'
print(THREADS, 'CPUs', MEM_GIB, 'GiB memory;', round(shutil.disk_usage(SCR).free / 2 ** 30), 'GiB free disk for', SCR)

# ======================================================================================================================================
# D1. Sibships and extended families per group, from the relatives cells' KING table (rel_work/{g}/king.kin0) and related pairs
# (rel_work/{g}/pairs.tsv); siblings must have every covariate. Then, for groups with enough siblings, the inputs of the kernel
# steps in ~/summit_direct_scratch/{g}: the siblings' genotypes at the MAF/LD-annotated SNPs, the annotation in their order, the
# covariates (every member of the group: the phenotypes are adjusted in the whole group), PC15 for one simulation setting, and the
# related pairs among the siblings. Minutes.
# CHECK: first-degree pairs split into sibling and parent-offspring pairs, with an IBS0
# gap between them (ibs0_first_degree: few pairs in the bins between 0.0003 and 0.005); sibling GRM median near 0.5;
# sibling_edges_grm_le_rel_min and sibships_with_non_sibling_pairs near 0; memory and disk within the machine.
os.chdir(KIT)   # every cell runs in this kit's folder
RUN, CHUNK, D1 = [], {}, []
WPC = pd.read_csv(f'{W}/within_group_pcs.tsv', sep='\t', dtype={'person_id': str})
HID = 'wPC15' if 'wPC15' in WPC.columns else None   # an ancestry PC that is not a covariate, for the 'ancestry' simulation setting
for grp in GROUPS:
    v1w, rp, sp = f'{V1}/rel_work/{grp}', f'{R}/{grp}', f'{SCR}/{grp}'
    for d in (rp, sp): os.makedirs(d, exist_ok=True)
    for f in ('king.kin0', 'pairs.tsv', 'cov.tsv'):
        assert done(f'{v1w}/{f}'), f'{grp}: {f} of the relatives cells is missing: run relatives_cells.py through R3 first'
    fam = AD.read_fam(f'{W}/{grp}.qc'); ids = fam.IID.to_numpy()
    cov = pd.read_csv(f'{v1w}/cov.tsv', sep='\t', dtype={'FID': str, 'IID': str})
    assert (cov.IID.to_numpy() == ids).all(), f'{grp}: the covariate file of the relatives cells is not in the order of {grp}.qc.fam'
    comp = cov.IID.to_numpy()[cov[COVS].notna().all(axis=1).to_numpy()]
    king = AD.read_king(f'{v1w}/king.kin0')
    pairs = pd.read_csv(f'{v1w}/pairs.tsv', sep='\t', dtype={'IID1': str, 'IID2': str})
    tab, rep = AD.sibships(king, pairs, ids, comp, SIB_KIN, IBS0_MIN, REL_MIN)
    write_if_changed(tab, f'{rp}/sibships.tsv', sep='\t', index=False)
    N, S = len(tab), rep['S_pairs']
    row = dict(group=grp, **AD.mask_report(rep))
    if N < MIN_SIB_MEMBERS:
        row['status'] = f'skipped (fewer than {MIN_SIB_MEMBERS} siblings)'
    else:
        annot = pd.read_csv(f'{W}/{grp}.mafld8.annot.gz', sep='\t', dtype={'SNP': str, 'CHR': str})
        classes = list(annot.columns[4:])
        assert len(classes) == 8 and (annot[classes].sum(axis=1) == 1).all() and annot.SNP.is_unique, f'{grp}: unexpected MAF/LD annotation'
        keep = fam[fam.IID.isin(set(tab.IID))]
        write_if_changed(keep, f'{sp}/sibs.keep', sep='\t', index=False, header=False)
        write_if_changed(annot[['SNP']], f'{sp}/annot.snps', index=False, header=False)
        if not fresh(f'{sp}/sibs.bed', f'{sp}/sibs.keep', f'{sp}/annot.snps', f'{W}/{grp}.qc.bed'):
            sh(f'{P2} --bfile {W}/{grp}.qc --keep {sp}/sibs.keep --extract {sp}/annot.snps --make-bed --out {sp}/sibs --threads {THREADS}', quiet=True)
        sfam = AD.read_fam(f'{sp}/sibs')
        assert set(sfam.IID) == set(tab.IID) and len(sfam) == N, f'{grp}: the sibling genotype file does not hold exactly the siblings'
        bim = pd.read_csv(f'{sp}/sibs.bim', sep=r'\s+', header=None, usecols=[1], names=['SNP'], dtype=str)
        A = annot.set_index('SNP').reindex(bim.SNP)
        assert A[classes].notna().all().all(), f'{grp}: SNPs of the genotype file without annotation'
        A = A.reset_index()[['CHR', 'BP', 'SNP', 'CM'] + classes]
        A[classes] = A[classes].astype(int)
        assert (A[classes].sum() > 0).all(), f'{grp}: a MAF/LD class has no SNP among the siblings'
        write_if_changed(A, f'{sp}/sibs.annot.tsv.gz', sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
        cp = cov[cov.IID.isin(set(comp))][['FID', 'IID'] + COVS].rename(columns={f'wPC{k}': f'PC{k}' for k in range(1, N_PCS + 1)})
        write_if_changed(cp, f'{sp}/covpc.tsv', sep='\t', index=False, float_format='%.10g')
        wp = WPC[WPC.group == grp].drop_duplicates('person_id').set_index('person_id').reindex(tab.IID)
        if HID:
            hid = wp[HID].to_numpy(float)
            assert np.isfinite(hid).all(), f'{grp}: {HID} missing for some siblings'
        else:   # no 15th PC: the 'ancestry' setting then uses a random family-level variable (it is reported, not checked)
            hid = np.random.default_rng(SEED).standard_normal(N)
        write_if_changed(pd.DataFrame({'IID': tab.IID.to_numpy(), 'PC15_AVG': hid}), f'{sp}/pcs20.tsv', sep='\t', index=False, float_format='%.10g')
        sset = set(tab.IID)
        sp_ = pairs[(pairs.GRM > REL_MIN) & pairs.IID1.isin(sset) & pairs.IID2.isin(sset)]
        write_if_changed(pd.DataFrame({'IID1': sp_.IID1, 'IID2': sp_.IID2, 'Kinship': sp_.GRM / 2}), f'{sp}/simpairs.tsv', sep='\t',
                         index=False, float_format='%.8g')
        res = AD.resources(N, S, len(bim))
        free = shutil.disk_usage(SCR).free / 2 ** 30
        row.update(snps=len(bim), ram_gib=res['ram_gib'], disk_gib=res['disk_gib'], free_disk_gib=round(free))
        if res['ram_gib'] > 0.85 * MEM_GIB or res['disk_gib'] > 0.9 * free:
            row['status'] = 'TOO LARGE for this machine'
            print(json.dumps(row)); raise RuntimeError(f"{grp}: the kernel steps need about {res['ram_gib']} GiB memory and {res['disk_gib']} GiB "
                                                       f"disk ({MEM_GIB} GiB memory, {free:.0f} GiB free disk here)")
        row['status'] = 'run'
        RUN.append(grp); CHUNK[grp] = res['blocks_chunk']
    D1.append(row)
    print(json.dumps(row))
AD.write_json(D1, f'{OUT}/sibships.json')
print(pd.DataFrame([{k: r.get(k) for k in ('group', 'sib_members', 'sibships', 'nu', 'sibling_pairs', 'parent_offspring_pairs', 'snps',
                                           'ram_gib', 'disk_gib', 'status')} for r in D1]).to_string(index=False))

# ======================================================================================================================================
# D2. Kernels: for each group, the eight MAF/LD-class GRMs of the siblings (sib_direct.py grm), then the phenotype-free parts of the
# SNP-block jackknife (sib_direct.py blocks). Both are in ~/summit_direct_scratch/{g}/grm (rebuilt after a
# new app). Time grows with the square of the number of siblings: about 10-30 minutes for 10,000 siblings and 1 million SNPs on
# 32 CPUs, four times that for 20,000.
# CHECK: "GRMs done" and "blocks done" for every group that D1 runs; the SNP count of each class.
os.chdir(KIT)   # every cell runs in this kit's folder
for grp in RUN:
    rp, sp = f'{R}/{grp}', f'{SCR}/{grp}'
    meta, inputs = f'{sp}/grm/meta.json', (f'{sp}/sibs.bed', f'{rp}/sibships.tsv', f'{sp}/sibs.annot.tsv.gz')
    if os.path.exists(meta) and not fresh(meta, *inputs):
        shutil.rmtree(f'{sp}/grm')                          # kernels of earlier inputs (an interrupted run of the same inputs resumes)
    if not (done(meta) and len(json.load(open(meta))['M']) == 8):
        sh(f'{SD} grm --bfile {sp}/sibs --sibs {rp}/sibships.tsv --annot {sp}/sibs.annot.tsv.gz --out {sp}/grm --threads {THREADS}')
    print(grp, 'SNPs per class:', json.load(open(meta))['M'])
    if not fresh(f'{sp}/grm/blocks.npz', meta):
        sh(f'{SD} blocks --grm {sp}/grm --bfile {sp}/sibs --annot {sp}/sibs.annot.tsv.gz --nblocks {NBLOCKS} --chunk {CHUNK[grp]}', quiet=True)

# ======================================================================================================================================
# D3. Simulation check on the siblings' genotypes (sib_direct.py sim and simjk): direct heritability 0.4 and six settings with
# NREP_SIM replicates each. base: no family effects; sibenv: an environment shared by siblings (0.2); extfam: environments and
# untagged genetic effects shared in extended families; nurture: an effect of the sibship's mean genetic value of a second score
# (parental effects); ancestry: an environment that follows the sibship's mean of PC15 (not a covariate); sibeffect: each
# sibling's genetic value affects the others (the direct estimator then targets var(direct - sibling) = 0.241).
# dir_ext is the direct estimator (SNP-block jackknife s.e.); pop_ext the population estimator on the same siblings with the
# off-diagonal pairs of each extended family left out (s.e. from the jackknife over groups of families). Minutes to an hour.
# CHECK: dir_ext |bias| below about 2 mcse in base, sibenv and nurture (also extfam and ancestry), s.e./SD
# near 1; pop_ext biased upward in nurture.
os.chdir(KIT)   # every cell runs in this kit's folder
DS = []
for grp in RUN:
    rp, sp = f'{R}/{grp}', f'{SCR}/{grp}'
    os.makedirs(f'{rp}/sim', exist_ok=True)
    if not fresh(f'{sp}/sim/sim_results_dir_snpjk.tsv', f'{sp}/grm/blocks.npz', f'{sp}/simpairs.tsv', f'{sp}/pcs20.tsv', f'{sp}/covpc.tsv'):
        shutil.rmtree(f'{sp}/sim', ignore_errors=True)    # sim reuses its genetic values when present: start clean
        sh(f'{SD} sim --grm {sp}/grm --bfile {sp}/sibs --pairs {sp}/simpairs.tsv --pcs20 {sp}/pcs20.tsv --covpc {sp}/covpc.tsv '
           f'--out {sp}/sim --nrep {NREP_SIM} --seed {SEED}', quiet=True)
        sh(f'{SD} simjk --grm {sp}/grm --bfile {sp}/sibs --sim {sp}/sim', quiet=True)
    for f in ('sim_results.tsv', 'sim_results_dir_snpjk.tsv'):        # replicate estimates (aggregate statistics), kept with the kit
        same_ = os.path.exists(f'{rp}/sim/{f}') and filecmp.cmp(f'{sp}/sim/{f}', f'{rp}/sim/{f}', shallow=False)
        if not same_: shutil.copy2(f'{sp}/sim/{f}', f'{rp}/sim/{f}')
    res = pd.read_csv(f'{rp}/sim/sim_results.tsv', sep='\t')
    snp = pd.read_csv(f'{rp}/sim/sim_results_dir_snpjk.tsv', sep='\t')
    for setting in SIM_TARGET:
        x = snp[snp.setting == setting].assign(target=SIM_TARGET[setting])
        DS.append(dict(group=grp, setting=setting, estimator='dir_ext', se_from='SNP blocks', **AD.summarize(x, se='se_snpjk').to_dict()))
        y = res[(res.setting == setting) & (res.estimator == 'pop_ext')].assign(target=0.4)
        DS.append(dict(group=grp, setting=setting, estimator='pop_ext', se_from='families', **AD.summarize(y).to_dict()))
DS = pd.DataFrame(DS); DS.to_csv(f'{OUT}/dsim_summary.csv', index=False)
print(DS.round(4).to_string(index=False))

# ======================================================================================================================================
# D4. Real traits: direct-effect heritability in the siblings (sib_direct.py fit and jkfit; single-class and 8-class models,
# SNP-block jackknife s.e.), for the relatives cells' stage 3 traits, beside the population estimate of the whole group with related pairs
# excluded (rel_results/stage3_h2.csv). Phenotypes are those of the relatives cells (inverse-normal transformed in the whole group), adjusted for age, sex,
# their products and enrollment year in the whole group, then centered within sibships; h2 is on the scale of the residual variance
# after all covariates of the relatives cells in the whole group. The z of the difference treats the two estimates as independent, which overstates
# its s.e. (the siblings are part of the group). Minutes to an hour.
# CHECK: every trait of stage 3 of the relatives cells is there; direct s.e. of height and education.
os.chdir(KIT)   # every cell runs in this kit's folder
assert done(f'{V1}/rel_results/stage3_h2.csv'), 'stage 3 of the summit_relatives_v1 kit has not finished (rel_results/stage3_h2.csv): run it first'
S3 = pd.read_csv(f'{V1}/rel_results/stage3_h2.csv')
S3 = S3[S3.arm == 'related pairs excluded']
DR = []
for grp in RUN:
    v1w, rp, sp = f'{V1}/rel_work/{grp}', f'{R}/{grp}', f'{SCR}/{grp}'
    ph_file = f'{v1w}/real_all.pheno.tsv'
    assert done(ph_file), f'{grp}: {ph_file} missing: run stage 3 of relatives_cells.py first'
    ph = pd.read_csv(ph_file, sep='\t', dtype={'FID': str, 'IID': str})
    sib = pd.read_csv(f'{rp}/sibships.tsv', sep='\t', dtype={'IID': str})
    obs = ph.set_index('IID').reindex(sib.IID)
    traits, skipped = [], []
    for t in sorted(set(S3.trait[S3.group == grp])):
        o = obs[t].notna().to_numpy()
        n_obs = pd.Series(o).groupby(sib.sib.to_numpy()).transform('sum').to_numpy()
        (traits if int((o & (n_obs >= 2)).sum()) >= MIN_TRAIT_MEMBERS else skipped).append(t)
    if skipped: print(grp, 'skipped (fewer than', MIN_TRAIT_MEMBERS, 'siblings measured):', skipped)
    fd = f'{rp}/fit'; os.makedirs(fd, exist_ok=True)
    deps = (f'{sp}/grm/blocks.npz', ph_file, f'{sp}/covpc.tsv', f'{rp}/sibships.tsv')
    ok = [t for t in traits if fresh(f'{fd}/{t}.json', *deps) and 'dir_8bin_se_snpjk' in json.load(open(f'{fd}/{t}.json'))]
    if len(ok) < len(traits):
        for f in glob.glob(f'{fd}/*.json'): os.remove(f)  # refit every trait, so that jkfit sees one consistent set
        sh(f"{SD} fit --grm {sp}/grm --pheno {ph_file} --covpc {sp}/covpc.tsv --traits {','.join(traits)} --out {fd} --as-cols {AS_COLS}")
        sh(f'{SD} jkfit --grm {sp}/grm --bfile {sp}/sibs --pheno {ph_file} --covpc {sp}/covpc.tsv --fit {fd} --as-cols {AS_COLS}')
    for t in traits:
        r = json.load(open(f'{fd}/{t}.json'))
        row = dict(group=grp, trait=t, n_members=M(r['n_members']), n_sibships=M(r['n_sibships']),
                   direct_single=r['dir_single_jk'], direct_single_se=r['dir_single_se_snpjk'],
                   direct_8bin=r['dir_8bin_jk'], direct_8bin_se=r['dir_8bin_se_snpjk'],
                   pop_siblings_8bin=r['pop_8bin'], pop_siblings_8bin_se_families=r['pop_8bin_se'], block_check=r['block_check'])
        for m_ in ('8bin', 'single'):
            v = S3[(S3.group == grp) & (S3.trait == t) & (S3.model == m_)]
            row[f'pop_v1_{m_}'] = float(v.h2.iloc[0]) if len(v) else np.nan
            row[f'pop_v1_{m_}_se'] = float(v.se.iloc[0]) if len(v) else np.nan
            if m_ == '8bin': row['n_v1'] = v.n.iloc[0] if len(v) else np.nan      # already masked
            d_, s_ = row[f'pop_v1_{m_}'] - row[f'direct_{m_}'], np.hypot(row[f'pop_v1_{m_}_se'], row[f'direct_{m_}_se'])
            row[f'diff_{m_}'], row[f'z_{m_}'] = d_, d_ / s_
            row[f'ratio_{m_}'] = row[f'direct_{m_}'] / row[f'pop_v1_{m_}']
        DR.append(row)
DR = pd.DataFrame(DR); DR.to_csv(f'{OUT}/dreal_h2.csv', index=False)
show = ['group', 'trait', 'n_members', 'direct_8bin', 'direct_8bin_se', 'pop_v1_8bin', 'pop_v1_8bin_se', 'z_8bin', 'ratio_8bin',
        'direct_single', 'direct_single_se', 'pop_v1_single', 'pop_siblings_8bin']
print(DR[show].round(4).to_string(index=False))

# ======================================================================================================================================
# D5. Export: one JSON with every summary (aggregate statistics only; counts of 1-20 are shown as "<=20"). Download
# dir_results/aou_summit_direct_v1_output.json.
os.chdir(KIT)   # every cell runs in this kit's folder
out = {'kit': 'summit_direct_v1', 'groups': GROUPS, 'run_groups': RUN,
       'parameters': dict(SIB_KIN=SIB_KIN, IBS0_MIN=IBS0_MIN, REL_MIN=REL_MIN, MIN_SIB_MEMBERS=MIN_SIB_MEMBERS, MIN_TRAIT_MEMBERS=MIN_TRAIT_MEMBERS,
                          NBLOCKS=NBLOCKS, NREP_SIM=NREP_SIM, SEED=SEED, AS_COLS=AS_COLS, HIDDEN_PC=HID),
       'sibships': None}
if done(f'{OUT}/sibships.json'):   # breakdowns with masked cells are left out: their totals would reveal the masked counts
    PARTIAL = ('sibship_sizes', 'ibs0_first_degree', 'sibships_with_non_sibling_pairs', 'non_sibling_pairs_in_sibships')
    out['sibships'] = [{k: v for k, v in s.items() if k not in PARTIAL} for s in json.load(open(f'{OUT}/sibships.json'))]
for k, f in (('simulation', 'dsim_summary.csv'), ('real', 'dreal_h2.csv')):
    out[k] = pd.read_csv(f'{OUT}/{f}').round(6).to_dict(orient='records') if done(f'{OUT}/{f}') else None   # counts already masked
AD.write_json(out, f'{OUT}/aou_summit_direct_v1_output.json'); print('written', f'{OUT}/aou_summit_direct_v1_output.json')

