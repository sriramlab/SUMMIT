# All of Us cells for "SUMMIT with related individuals": run in order in one notebook of the Researcher Workbench, in a folder
# that holds relatives_helpers.py. Stage 1 checks SUMMIT on the unrelated panel with simulated phenotypes; stage 3 fits real
# traits for the unrelated panel, everyone with related pairs kept and everyone with related pairs excluded; ARMS adds the
# unrelated panel with its own listed pairs excluded and paired jackknife SEs of the differences; R7 exports aggregate statistics
# (counts of 1-20 masked; breakdowns whose totals would reveal masked counts are left out). Every step skips work whose files are
# up to date, so after a kernel restart run F and R1-R3 again (minutes) and continue.
# ======================================================================================================================================
# F. Foundation files. Set FOUNDATION to the folder that holds, for each group g in GROUPS: g.qc.{bed,bim,fam} (array genotypes
# of every quality-controlled member of the group, relatives included), g.panel.{bed,bim,fam} (the unrelated panel, plink2
# --king-cutoff 0.0442), g.prune.prune.in (LD-pruned SNPs), g.mafld8.annot.gz (8 MAF-LD bins of the group's SNPs), and
# within_group_pcs.tsv (person_id, group, wPC1..) and cohort_D.pkl (person_id, age, female, pm_date, bmi, sbp, dbp, edu: the
# participant table of the foundation). Run this cell first in a new folder; it links the folder as summit_work.
import os
FOUNDATION = os.path.expanduser('~/foundation')   # set to the foundation folder
KIT = os.getcwd()   # this kit's folder: every later cell returns here first
if os.path.islink('summit_work') and os.path.realpath('summit_work') != os.path.realpath(FOUNDATION):
    os.remove('summit_work')
if not os.path.lexists('summit_work'):
    os.symlink(FOUNDATION, 'summit_work')
assert os.path.exists('summit_work/within_group_pcs.tsv'), 'FOUNDATION does not hold the foundation files'
# ======================================================================================================================================
# R1. Setup: parameters, helpers, and SUMMIT f9858ae with plink2 2.0.0a.7.8 in their own environment (~/summit_rel_env). About 10-20 minutes the first time in an app.
# CHECK: the commit prints f9858ae, "installed SUMMIT matches commit f9858ae: True", "SUMMIT OK", and plink2 v2.0.0-a.7.8.
import os, json, glob, shutil, subprocess, time, sys, filecmp, collections, concurrent.futures as cf, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import relatives_helpers as AR
W, R, OUT = 'summit_work', 'rel_work', 'rel_results'   # foundation files (linked by F; only read), this kit's files, outputs
assert os.path.islink(W) and os.path.exists(f'{W}/within_group_pcs.tsv'), \
    f'{os.getcwd()} has no summit_work link: run cell F first'
KIT = os.getcwd()   # this kit's folder: every later cell returns here first
for d in (R, OUT): os.makedirs(d, exist_ok=True)
GROUPS = ['eur', 'afr', 'amr']
THREADS = os.cpu_count(); MEM_GIB = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30)
LD_MEM_GIB = max(8, int(MEM_GIB * 0.5))   # the related-pair part of the in-sample LD scores needs memory on top of this
SUMMIT_COMMIT = 'f9858ae'     # SUMMIT main: related individuals, one REL_M value per trait (github.com/sriramlab/SUMMIT)
N_PCS = 10                    # within-group PCs in every model
KING_FILTER = 0.0125          # KING kinship above which a pair is a candidate; its relatedness then comes from the PC-adjusted GRM
REL_MIN = 0.0442              # GRM relatedness (twice kinship) above which a pair is related: fourth-degree relatives and closer
FIRST_DEGREE = 0.354          # relatedness above which a pair is first degree (households in the simulations)
NVECS = 1024                  # random vectors for the LD scores
SEED = 20261025
MODELS = ['8bin']             # MAF/LD model of the fits; add 'single' to also compute single-component LD scores (about 1.5x the LD time)
FIT_THREADS = 4               # threads per SUMMIT fit; fits run side by side
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
def run_many(cmds, workers=None):
    """Run shell commands side by side; returns the commands that failed."""
    cmds = [c for c in cmds if c]
    if not cmds: return []
    workers = workers or max(1, THREADS // FIT_THREADS)
    print(f'running {len(cmds)} commands, up to {workers} at a time', flush=True); t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).returncode, cmds))
    print(f'[{time.time() - t0:.0f} s]', flush=True)
    return [c for c, r in zip(cmds, res) if r != 0]
def summarize(df, keys):
    """AR.summarize_sims per group of rows (a loop, so that it works with any pandas version)."""
    return pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **AR.summarize_sims(d).to_dict(), 'n': AR.mask_count(d.n.max())}
                         for k, d in df.groupby(keys, sort=True)])
M = AR.mask_count
ENV = os.path.expanduser('~/summit_rel_env'); SUMMIT = f'{ENV}/bin/summit'; PY = f'{ENV}/bin/python'; P2 = f'{ENV}/bin/plink2'
SRC = os.path.expanduser('~/SUMMIT_rel')    # source clone
MM = shutil.which('micromamba') or shutil.which('mamba') or next((p for p in [os.path.expanduser('~/mm/bin/micromamba')] if os.path.exists(p)), None)
if MM is None:   # a static micromamba solves this environment in about a minute; the download is retried if it breaks off
    for attempt in range(1, 4):
        try:
            sh('mkdir -p ~/mm && curl -Ls --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/mm bin/micromamba'); break
        except RuntimeError:
            if attempt == 3: raise
            time.sleep(15)
    MM = os.path.expanduser('~/mm/bin/micromamba')
os.environ.setdefault('MAMBA_ROOT_PREFIX', os.path.expanduser('~/mmroot'))
PKGS = ("python=3.12.15 numpy=1.26.4 pandas=3.0.6 scipy=1.17.1 tqdm=4.70.1 bed-reader=1.0.0 psutil=7.2.2 threadpoolctl=3.7.0 pip "   
        "cmake=4.4.4 ninja=1.13.2 nanobind=3.1.0 scikit-build-core=1.1.1 pkg-config=0.29.2 openblas=0.3.34 'libblas=3.11.0=*openblas' "
        "'liblapack=3.11.0=*openblas' cxx-compiler=2.0.0 make git")
if not os.path.exists(f'{SRC}/.git'): sh(f'git clone -q https://github.com/sriramlab/SUMMIT.git {SRC}')
sh(f'git -C {SRC} checkout -q {SUMMIT_COMMIT} 2>/dev/null || (git -C {SRC} fetch -q origin && git -C {SRC} checkout -q {SUMMIT_COMMIT})')
sh(f'git -C {SRC} rev-parse --short HEAD')                                                                  # CHECK: f9858ae
FILES = ('relatives/moments.py', 'inference/sumrhe.py', 'ldscore/gw_ldscore.py', 'sumstats/moments.py', 'cli.py')
def installed_is_source():
    """The environment's SUMMIT files equal the source checkout's (False if the environment is missing or broken)."""
    if not (os.path.exists(SUMMIT) and os.path.exists(P2)): return False
    r = subprocess.run([PY, '-c', 'import summit, os; print(os.path.dirname(summit.__file__))'], capture_output=True, text=True)
    inst = r.stdout.strip()
    return r.returncode == 0 and all(os.path.exists(f'{inst}/{f}') and filecmp.cmp(f'{SRC}/src/summit/{f}', f'{inst}/{f}', shallow=False) for f in FILES)
if os.path.exists(ENV) and not installed_is_source():   # an interrupted install, or the SUMMIT commit of an earlier kit version
    print('~/summit_rel_env is incomplete or holds another SUMMIT commit: rebuilding it (10-20 minutes)', flush=True); shutil.rmtree(ENV)
if not os.path.exists(ENV):
    sh(f'{MM} create -y -q -p {ENV} -c conda-forge --override-channels {PKGS}', quiet=True)
    sh(f'{MM} run -p {ENV} pip install -q pgenlib==0.94.1')
    sh(f'cd {SRC} && CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH={ENV} -DCMAKE_LIBRARY_PATH={ENV}/lib -DCMAKE_INCLUDE_PATH={ENV}/include" '
       f'{MM} run -p {ENV} pip install --no-build-isolation -v . > ~/summit_rel_build.log 2>&1 || (tail -40 ~/summit_rel_build.log; exit 1)')
    sh(f'{MM} install -y -q -p {ENV} -c bioconda -c conda-forge plink2=2.0.0a.7.8', quiet=True)
same = installed_is_source()
print(f'installed SUMMIT matches commit {SUMMIT_COMMIT}:', same)
assert same, f'the installed SUMMIT is not commit {SUMMIT_COMMIT} after installing it: check the installation log'
sh(f'{PY} -c "import summit, summit.gwldcore, bed_reader; print(\'SUMMIT OK\')"')
sh(f'{SUMMIT} relatives moments --help', quiet=True)
sh(f'{P2} --version')
LDFLAGS = f'--nvecs {NVECS} --block-size 10000 --dtype float64 --memory-gib {LD_MEM_GIB} --num-threads {THREADS} --seed {SEED}'
print(THREADS, 'CPUs', MEM_GIB, 'GiB; LD memory', LD_MEM_GIB, 'GiB')

# ======================================================================================================================================
# R2. Inputs from the foundation: every QC'd member of each group (summit_work/{g}.qc, relatives included), the unrelated panel
# (summit_work/{g}.panel, KING kinship 0.0442), the within-group PCs, and the GWAS covariates (age, sex and
# their products, enrollment year, PC1-10). The PCs are the projections for everyone, so that every analysis uses the same definitions.
# One covariate file per group serves the GWAS, the related-pair parts and the LD scores.
# CHECK: complete (people with every covariate) and in_panel (the unrelated panel among them) for each group.
os.chdir(KIT)   # every cell runs in this kit's folder
D = pd.read_pickle(f'{W}/cohort_D.pkl'); D['person_id'] = D.person_id.astype(str); D = D.drop_duplicates('person_id').set_index('person_id')
WPC = pd.read_csv(f'{W}/within_group_pcs.tsv', sep='\t', dtype={'person_id': str})
PCS = [f'wPC{k}' for k in range(1, N_PCS + 1)]
COVS = ['age', 'sex', 'age_sex', 'age2', 'age2_sex', 'byear'] + PCS
FIDS, COMP, PANEL, CN = {}, {}, {}, []
for grp in GROUPS:
    g, rp = f'{W}/{grp}', f'{R}/{grp}'; os.makedirs(rp, exist_ok=True)
    fam = AR.read_fam(f'{g}.qc'); assert fam.IID.is_unique
    assert fam.IID.isin(D.index).all(), f'{grp}: QC members missing from cohort D'
    FIDS[grp] = dict(zip(fam.IID, fam.FID))
    pc = WPC[WPC.group == grp].drop_duplicates('person_id').set_index('person_id')[PCS].reindex(fam.IID.values)
    assert pc.notna().all().all(), f'{grp}: within-group PCs missing for some QC members '
    Q = D.reindex(fam.IID.values)
    a = (Q.age.values - 55) / 10
    X = fam.copy()
    X['age'] = a; X['sex'] = Q.female.values; X['age_sex'] = a * Q.female.values; X['age2'] = a ** 2; X['age2_sex'] = a ** 2 * Q.female.values
    X['byear'] = Q.pm_date.dt.year.values - 2020
    X[PCS] = pc.values
    write_if_changed(X[['FID', 'IID'] + COVS], f'{rp}/cov.tsv', sep='\t', index=False, na_rep='NA', float_format='%.10g')
    write_if_changed(X[['IID'] + PCS], f'{rp}/pcs.tsv', sep='\t', index=False, float_format='%.10g')
    ok = X[COVS].notna().all(axis=1).values
    COMP[grp] = list(X.IID.values[ok])
    panel = set(AR.read_fam(f'{g}.panel').IID)
    PANEL[grp] = [x for x in COMP[grp] if x in panel]
    CN.append(dict(group=grp, complete=M(len(COMP[grp])), in_panel=M(len(PANEL[grp])), missing_covariates=M(int((~ok).sum()))))
print(pd.DataFrame(CN).to_string(index=False))
AR.write_json(CN, f'{OUT}/samples.json')

# ======================================================================================================================================
# R3. Related pairs per group. KING on the pruned SNPs lists candidate pairs (kinship > 0.0125); the relatedness of each candidate is
# then taken from the GRM of the pruned SNPs after adjustment for PC1-10 (the definition in the manuscript). Related pairs: relatedness
# > 0.0442 (fourth degree and closer); the listed pairs are excluded (--no-closure).
# EUR: KING over all pairs takes a few hours ; the GRM step about 30-60 minutes.
# CHECK: the degree counts; within_family_pairs (what SUMMIT excludes); panel_pairs (related pairs left inside the unrelated
# panel, expected to be mostly fourth degree); and in the KING x GRM table, few pairs in the first row (KING <= 0.0221) with GRM above
# 0.0442 (if there are many, the KING filter is too high).
os.chdir(KIT)   # every cell runs in this kit's folder
PAIRS = {}
for grp in GROUPS:
    g, rp = f'{W}/{grp}', f'{R}/{grp}'
    if not fresh(f'{rp}/king.kin0', f'{g}.qc.bed', f'{g}.prune.prune.in'):
        sh(f'{P2} --bfile {g}.qc --extract {g}.prune.prune.in --make-king-table --king-table-filter {KING_FILTER} '
           f'--out {rp}/king --threads {THREADS}', quiet=True)
    if not fresh(f'{rp}/pairs.tsv', f'{rp}/king.kin0', f'{rp}/pcs.tsv'):
        sh(f'{PY} relatives_helpers.py grm --bfile {g}.qc --snps {g}.prune.prune.in --pcs {rp}/pcs.tsv --kin0 {rp}/king.kin0 --out {rp}/pairs.tsv')
    P_ = pd.read_csv(f'{rp}/pairs.tsv', sep='\t', dtype={'IID1': str, 'IID2': str}); PAIRS[grp] = P_
    comp, pan = set(COMP[grp]), set(PANEL[grp])
    fmap, _ = AR.families(P_, REL_MIN)                               # over every QC member, as SUMMIT forms families
    sizes = [n for n in collections.Counter(f for x, f in fmap.items() if x in comp).values() if n >= 2]
    Pc = P_[P_.IID1.isin(comp) & P_.IID2.isin(comp)]
    rel = Pc[Pc.GRM > REL_MIN]
    total, deg = AR.mask_parts(len(rel), AR.degree_counts(rel.GRM))
    tab = AR.king_grm_table(Pc)
    row = dict(group=grp, candidate_pairs=M(len(Pc)), related_pairs=total, **deg,
               people_in_families=M(sum(sizes)), families=M(len(sizes)), largest_family=M(max(sizes) if sizes else 0),
               within_family_pairs=M(sum(n * (n - 1) // 2 for n in sizes)),
               panel_pairs=M(int((rel.IID1.isin(pan) & rel.IID2.isin(pan)).sum())),
               king_x_grm={str(k): {str(c): M(v) for c, v in r_.items()} for k, r_ in tab.to_dict(orient='index').items()})
    print(json.dumps({k: v for k, v in row.items() if k != 'king_x_grm'}))
    print(tab.apply(lambda col: col.map(M)).to_string())
    AR.write_json(row, f'{OUT}/{grp}_pairs_summary.json')

# ======================================================================================================================================
# R4. Stage 1: SUMMIT on the unrelated panel, with simulated phenotypes. Genetic values have an effect at every SNP (sum of
# squared effects one, genotypes standardized), so the simulated SNP heritability is h2 exactly; residuals are independent.
# Replicates 1-10 have h2 = 0.25 and 11-20 h2 = 0.5 (replicates 21-30, with effects shared by relatives, are simulated but not
# fitted here). Fits use in-sample LD scores of the panel.
# EUR: GWAS about 30-60 minutes, LD scores about 1.5-3 hours, fits about 30 minutes.
# CHECK: |bias| below about 2 mcse and coverage near 0.95 in every row.
os.chdir(KIT)   # every cell runs in this kit's folder
SIM = ([dict(rep=k, scenario=1, h2=0.25, s2u=0.0, s2c=0.0) for k in range(1, 11)] + [dict(rep=k, scenario=1, h2=0.5, s2u=0.0, s2c=0.0) for k in range(11, 21)]
       + [dict(rep=k, scenario=3, h2=0.25, s2u=0.25, s2c=0.10) for k in range(21, 31)])
NREP = len(SIM)
def gwas(grp, bfile, pheno, names, ssdir):
    """plink2 linear GWAS of phenotype columns `names` (people of bfile with the phenotype and every covariate) to SUMMIT summary
    statistics ssdir/<name>.tsv with COV_RANK; returns {name: largest per-SNP sample size}."""
    cov = f'{R}/{grp}/cov.tsv'; os.makedirs(ssdir, exist_ok=True)
    todo = [n for n in names if not fresh(f'{ssdir}/{n}.tsv', pheno, cov, bfile + '.bed')]
    if todo:
        tmp = ssdir.rstrip('/') + '_glm'; os.makedirs(tmp, exist_ok=True)
        sh(f"{P2} --bfile {bfile} --pheno {pheno} --pheno-name {','.join(todo)} --covar {cov} --covar-name {','.join(COVS)} "
           f"--covar-variance-standardize --no-input-missing-phenotype --glm hide-covar no-firth cols=chrom,pos,ref,alt,omitted,nobs,beta,se "
           f"--out {tmp}/g --threads {THREADS}", quiet=True)
        for n in todo:
            AR.glm_to_sumstats(f'{tmp}/g.{n}.glm.linear', f'{ssdir}/{n}.tsv.tmp', len(COVS)); os.replace(f'{ssdir}/{n}.tsv.tmp', f'{ssdir}/{n}.tsv')
        shutil.rmtree(tmp)
    return {n: int(pd.read_csv(f'{ssdir}/{n}.tsv', sep='\t', usecols=['N']).N.max()) for n in names}
def same_sample(grp, bfile, ids, ss, n_rel):
    """The REL_M sample (n_rel people) is the GWAS sample `ids`: the sizes agree, and the GWAS N of every SNP equals the number of
    these people with a genotype call (plink2 --missing), so the GWAS used exactly these people."""
    tmp = f'{R}/{grp}/_check'; os.makedirs(tmp, exist_ok=True)
    pd.DataFrame({'FID': [FIDS[grp][x] for x in ids], 'IID': ids}).to_csv(f'{tmp}/s.keep', sep='\t', index=False, header=False)
    sh(f'{P2} --bfile {bfile} --keep {tmp}/s.keep --missing variant-only --out {tmp}/s --threads {THREADS}', quiet=True)
    vm = pd.read_csv(f'{tmp}/s.vmiss', sep='\t'); vm.columns = [c.lstrip('#') for c in vm.columns]
    n = pd.read_csv(ss, sep='\t', usecols=['SNP', 'N']).set_index('SNP').N
    called = (vm.OBS_CT - vm.MISSING_CT).set_axis(vm.ID)                  # OBS_CT counts every sample of the keep list
    ok = len(ids) == n_rel and bool((called.reindex(n.index) == n).all())
    shutil.rmtree(tmp)
    return ok
def ldfile(pre, model): return f"{pre}.{'ld8' if model == '8bin' else 'ld1'}.gw.ldscore.gz"
def ld_scores(grp, bfile, pre, pairs=None):
    """In-sample LD scores (V-weighted) of the people of bfile with every covariate; with pairs, also their related-pair parts."""
    cov, annot = f'{R}/{grp}/cov.tsv', f'{W}/{grp}.mafld8.annot.gz'
    for model in MODELS:
        out = ldfile(pre, model)
        if fresh(out, bfile + '.bed', cov, annot, pairs): continue
        rel = f'--relative-pairs {pairs} --kinship-col GRM --kinship-min {REL_MIN} --no-closure' if pairs else ''
        sh(f"{SUMMIT} --geno {bfile}.bed --covar {cov} {'--annot ' + annot if model == '8bin' else ''} {rel} "
           f"--out {out.replace('.gw.ldscore.gz', '')} {LDFLAGS}", quiet=True)
def fit(grp, ssdir, ldpre, out, model):
    """One SUMMIT h2 fit of every sumstats file in ssdir (200-block jackknife); returns the command, or None if current."""
    ld, files = ldfile(ldpre, model), sorted(glob.glob(f'{ssdir}/*.tsv'))
    flags = f'--annot {W}/{grp}.mafld8.annot.gz --ldscores {ld}' if model == '8bin' else f'--ldscores {ld}'
    res = f'{out}.results.tsv'
    if fresh(res, ld, *files) and len(pd.read_csv(res, sep='\t')) == len(files): return None
    return f'{SUMMIT} --h2 {ssdir} {flags} --njack 200 --max-chisq auto --num-threads {FIT_THREADS} --out {out} > {out}.stdout 2>&1'
S1 = []
for grp in GROUPS:
    g, rp = f'{W}/{grp}', f'{R}/{grp}'; os.makedirs(f'{rp}/fit', exist_ok=True)
    assert done(f'{g}.mafld8.annot.gz'), f'{g}.mafld8.annot.gz missing: the foundation must provide it'
    if not done(f'{rp}/w.tsv'): AR.write_score_file(f'{g}.qc', NREP, f'{rp}/w.tsv', SEED + GROUPS.index(grp))
    wss = pd.read_csv(f'{rp}/w.tsv', sep='\t', usecols=[f'W{k}' for k in range(1, NREP + 1)]).pow(2).sum().to_numpy()
    if not fresh(f'{rp}/gv.sscore', f'{rp}/w.tsv', f'{g}.qc.bed'):
        sh(f'{P2} --bfile {g}.qc --score {rp}/w.tsv 1 2 header-read variance-standardize cols=scoresums --score-col-nums 3-{2 + NREP} '
           f'--out {rp}/gv --threads {THREADS}', quiet=True)
    G = AR.read_scores(f'{rp}/gv.sscore', NREP) / np.sqrt(wss)[None, :]   # genetic values with sum of squared effects one
    comp = COMP[grp]; Gc = G.loc[comp]
    rng = np.random.default_rng(SEED + 100 + GROUPS.index(grp))
    Y = pd.DataFrame({'FID': [FIDS[grp][x] for x in comp], 'IID': comp})
    for s in SIM:
        u, dd = (AR.family_effects(comp, PAIRS[grp], s['s2u'], s['s2c'], REL_MIN, FIRST_DEGREE, rng) if s['s2u'] + s['s2c'] > 0 else (0.0, 0.0))
        Y[f"r{s['rep']}"] = (np.sqrt(s['h2']) * Gc[f"g{s['rep']}"].values + u + dd
                             + np.sqrt(1 - s['h2'] - s['s2u'] - s['s2c']) * rng.standard_normal(len(comp)))
    write_if_changed(Y, f'{rp}/sim.pheno.tsv', sep='\t', index=False, float_format='%.8g')
    names = [f'r{k}' for k in range(1, 21)]
    n = gwas(grp, f'{g}.panel', f'{rp}/sim.pheno.tsv', names, f'{rp}/ss_u1')
    ld_scores(grp, f'{g}.panel', f'{rp}/panel')
    old = [m for m in ('single', '8bin') if done(ldfile(g, m))]
    bad = run_many([fit(grp, f'{rp}/ss_u1', f'{rp}/panel', f'{rp}/fit/u1.{m}', m) for m in MODELS]
                   + [fit(grp, f'{rp}/ss_u1', g, f'{rp}/fit/u1B.{m}', m) for m in old])
    assert not bad, f'failed fits, see their .stdout files: {bad[:2]}'
    for tag, models, ld in (('u1', MODELS, 'new (with V weights)'), ('u1B', old, 'foundation')):
        for m in models:
            for _, x in AR.read_results(f'{rp}/fit/{tag}.{m}.results.tsv').iterrows():
                s = SIM[int(x.phen[1:]) - 1]
                S1.append(dict(group=grp, ld=ld, model=m, setting=f"h2={s['h2']}", truth=s['h2'], h2=x.h2, se=x.se, n=n[x.phen]))
S1 = pd.DataFrame(S1); S1.to_csv(f'{OUT}/stage1_replicates.csv', index=False)
S1s = summarize(S1, ['group', 'ld', 'model', 'setting'])
S1s.to_csv(f'{OUT}/stage1_summary.csv', index=False); print(S1s.round(4).to_string(index=False))

# ======================================================================================================================================
# R6. Stage 3: real traits, the same three analyses. Program physical measurements of the first visit (height, BMI, SBP, DBP, waist-to-
# hip ratio) and education from the survey (years), each inverse-normal transformed within the analysed sample. Height,
# waist and hip come from BigQuery (in-person measurements, remote self-reports excluded, the date of the program blood pressure).
# About 5 minutes of BigQuery, then GWAS, REL_M (one genotype pass for all traits) and fits: EUR about 2-4 hours.
# CHECK: the concept names printed first (height, waist and hip); the trait counts.
os.chdir(KIT)   # every cell runs in this kit's folder
if not done(f'{R}/pm_extra.pkl'):
    from google.cloud import bigquery
    CDR = os.environ['WORKSPACE_CDR']
    client = bigquery.Client(project=os.environ.get('GOOGLE_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT'))
    def q(sql):
        job = client.query(sql)
        try: return job.to_dataframe(create_bqstorage_client=True)
        except Exception: return job.to_dataframe(create_bqstorage_client=False)
    CONC = {903133: 'height', 903135: 'waist', 903136: 'hip'}
    nm = q(f'SELECT concept_id, concept_name FROM `{CDR}.concept` WHERE concept_id IN (903133, 903135, 903136)')
    print(nm.to_string(index=False))
    use = [c for c, word in CONC.items() if word in ' '.join(nm.concept_name[nm.concept_id == c]).lower()]
    assert 903133 in use, 'concept 903133 is not height in this CDR'
    x = q(f'''SELECT person_id, measurement_date AS d, measurement_source_concept_id AS src, value_as_number AS v FROM `{CDR}.measurement`
              WHERE measurement_source_concept_id IN ({', '.join(map(str, use))}) AND measurement_type_concept_id != 32865
              AND value_as_number IS NOT NULL''')
    x['person_id'] = x.person_id.astype(str); x['d'] = pd.to_datetime(x.d)
    x = x[x.person_id.isin(D.index)]
    x = x[x.d.values == D.pm_date.reindex(x.person_id).values]                 # the visit of the program blood pressure
    x = x.groupby(['person_id', 'src']).v.mean().unstack().rename(columns=CONC)
    x.to_pickle(f'{R}/pm_extra.pkl')
X3 = pd.read_pickle(f'{R}/pm_extra.pkl').reindex(D.index)
T = pd.DataFrame(index=D.index)
T['height'] = X3.height.where(X3.height.between(120, 220))
T['bmi'] = D.bmi.where(D.bmi.between(12, 75))
T['sbp'] = D.sbp; T['dbp'] = D.dbp
if {'waist', 'hip'} <= set(X3.columns):
    whr = X3.waist / X3.hip
    T['whr'] = whr.where(X3.waist.between(40, 200) & X3.hip.between(50, 220) & whr.between(0.5, 1.5))
T['edu'] = D.edu.map({0: 10, 1: 12, 2: 14, 3: 16, 4: 19})
TRAITS3 = list(T.columns)
print(pd.DataFrame({t: dict(n=M(int(T[t].notna().sum())), mean=round(T[t].mean(), 2), sd=round(T[t].std(), 2)) for t in TRAITS3}).T.to_string())
S3 = []
for grp in GROUPS:
    g, rp = f'{W}/{grp}', f'{R}/{grp}'
    for samp, ids in (('all', COMP[grp]), ('unrel', PANEL[grp])):
        Y = pd.DataFrame({'FID': [FIDS[grp][x] for x in ids], 'IID': ids})
        for t in TRAITS3: Y[t] = AR.irnt(T[t].reindex(ids).values).values     # within the analysed sample
        write_if_changed(Y, f'{rp}/real_{samp}.pheno.tsv', sep='\t', index=False, na_rep='NA', float_format='%.8g')
    tr = [t for t in TRAITS3 if int(pd.read_csv(f'{rp}/real_unrel.pheno.tsv', sep='\t', usecols=[t])[t].notna().sum()) >= 1000]
    n_all = gwas(grp, f'{g}.qc', f'{rp}/real_all.pheno.tsv', tr, f'{rp}/ss_rall')
    n_u = gwas(grp, f'{g}.panel', f'{rp}/real_unrel.pheno.tsv', tr, f'{rp}/ss_runrel')
    if not fresh(f'{rp}/realrel.rel_m.tsv', f'{rp}/real_all.pheno.tsv', f'{rp}/cov.tsv', f'{rp}/pairs.tsv', f'{g}.qc.bed', f'{W}/{grp}.mafld8.annot.gz'):
        sh(f"{SUMMIT} relatives moments --geno {g}.qc --pheno {rp}/real_all.pheno.tsv --pheno-cols {','.join(tr)} --covar {rp}/cov.tsv "
           f"--pairs {rp}/pairs.tsv --kinship-col GRM --kinship-min {REL_MIN} --no-closure --annot {W}/{grp}.mafld8.annot.gz --out {rp}/realrel", quiet=True)
    js = json.load(open(f'{rp}/realrel.json'))['phenotypes']
    ph = pd.read_csv(f'{rp}/real_all.pheno.tsv', sep='\t', dtype={'FID': str, 'IID': str})
    for t in tr:   # the related-pair parts must come from the GWAS sample and covariates of each trait
        assert js[t]['n_covariates'] == len(COVS), f'{grp} {t}: REL_M covariates differ from the GWAS'
        assert same_sample(grp, f'{g}.qc', list(ph.IID[ph[t].notna()]), f'{rp}/ss_rall/{t}.tsv', js[t]['n']), f'{grp} {t}: the REL_M sample is not the GWAS sample'
    os.makedirs(f'{rp}/ss_rfrel', exist_ok=True)
    for t in tr:
        if not fresh(f'{rp}/ss_rfrel/{t}.tsv', f'{rp}/ss_rall/{t}.tsv', f'{rp}/realrel.rel_m.tsv'):
            sh(f'{SUMMIT} relatives append --sumstats {rp}/ss_rall/{t}.tsv --rel {rp}/realrel.rel_m.tsv --pheno {t} '
               f'--annot {W}/{grp}.mafld8.annot.gz --out {rp}/ss_rfrel/{t}.tsv.tmp', quiet=True)
            os.replace(f'{rp}/ss_rfrel/{t}.tsv.tmp', f'{rp}/ss_rfrel/{t}.tsv')
    bad = run_many(sum([[fit(grp, f'{rp}/ss_runrel', f'{rp}/panel', f'{rp}/fit/runrel.{m}', m), fit(grp, f'{rp}/ss_rall', f'{rp}/all', f'{rp}/fit/rall.{m}', m),
                         fit(grp, f'{rp}/ss_rfrel', f'{rp}/all', f'{rp}/fit/rfrel.{m}', m)] for m in MODELS], []))
    assert not bad, f'failed fits, see their .stdout files: {bad[:2]}'
    for m in MODELS:
        for arm, tag, nn in (('unrelated panel', 'runrel', n_u), ('related pairs kept', 'rall', n_all), ('related pairs excluded', 'rfrel', n_all)):
            for _, x in AR.read_results(f'{rp}/fit/{tag}.{m}.results.tsv').iterrows():
                S3.append(dict(group=grp, trait=x.phen, model=m, arm=arm, n=M(nn[x.phen]), h2=x.h2, se=x.se,
                               pairs_excluded=M(js[x.phen]['pairs_used']) if tag == 'rfrel' else 0))
S3 = pd.DataFrame(S3); S3.to_csv(f'{OUT}/stage3_h2.csv', index=False)
wide = S3.pivot_table(index=['group', 'trait', 'model'], columns='arm', values=['h2', 'se'], aggfunc='first')
wide[('se_ratio', 'excluded/panel')] = wide[('se', 'related pairs excluded')] / wide[('se', 'unrelated panel')]
print(wide.round(4).to_string())

# ======================================================================================================================================
# ARMS. Two additions to stage 3 for every group in GROUPS that has stage-3 results; run it after R6.
# (a) A fourth arm: the unrelated panel with its own related pairs left out. The panel comes from KING at kinship 0.0442, so it still
#     holds pairs whose GRM relatedness is above REL_MIN; this arm leaves them out exactly as the related-pairs-excluded arm does for
#     everyone (listed pairs, --no-closure). One LD pass over the panel with related-pair parts (AFR and AMR about 1-2 hours each, EUR
#     about 2-3 hours), one REL_M pass over the panel, then a fit.
# (b) Every arm refitted with --write-jack in a new folder (fitjk; R6's fits are not touched), so that the differences between arms get
#     standard errors from the same 200 jackknife blocks. The cell checks that the refits reproduce R6's estimates and SUMMIT's SEs.
# The table holds aggregate statistics only. Running the cell again redoes nothing that is current.
os.chdir(KIT)   # every cell runs in this kit's folder
ARMS = {'U': ('ss_runrel', 'panel'), 'UX': ('ss_rufrel', 'panelrel'), 'K': ('ss_rall', 'all'), 'X': ('ss_rfrel', 'all')}
S3NAME = {'U': 'unrelated panel', 'K': 'related pairs kept', 'X': 'related pairs excluded'}
DIFFS = (('K', 'X'), ('X', 'U'), ('X', 'UX'), ('U', 'UX'))


def jack_reps(out, phen):
    """Delete-one-block replicates of the total h2 written by --write-jack (the 'full' row dropped)."""
    f = f'{out}.{phen}.jack'
    assert os.path.exists(f), f'{f} is missing (and list the fitjk folder)'
    j = pd.read_csv(f, sep='\t', dtype={'# replicate': str})
    return j.loc[j['# replicate'] != 'full', 'total_h2'].to_numpy(float)


def jk_se(x):
    x = np.asarray(x, float)
    return float(np.sqrt((len(x) - 1) / len(x) * ((x - x.mean()) ** 2).sum()))


S3 = pd.read_csv(f'{OUT}/stage3_h2.csv')
TODO = [g for g in GROUPS if (S3.group == g).any()]
print('groups:', TODO, '| skipped (no stage-3 results yet):', [g for g in GROUPS if g not in TODO])
A4 = []
for grp in TODO:
    g, rp = f'{W}/{grp}', f'{R}/{grp}'
    tr = sorted(S3.trait[(S3.group == grp)].unique())
    os.makedirs(f'{rp}/fitjk', exist_ok=True); os.makedirs(f'{rp}/ss_rufrel', exist_ok=True)
    # (a) the panel with its related pairs left out
    ld_scores(grp, f'{g}.panel', f'{rp}/panelrel', pairs=f'{rp}/pairs.tsv')
    if not fresh(f'{rp}/realrel_unrel.rel_m.tsv', f'{rp}/real_unrel.pheno.tsv', f'{rp}/cov.tsv', f'{rp}/pairs.tsv', f'{g}.panel.bed',
                 f'{W}/{grp}.mafld8.annot.gz'):
        sh(f"{SUMMIT} relatives moments --geno {g}.panel --pheno {rp}/real_unrel.pheno.tsv --pheno-cols {','.join(tr)} --covar {rp}/cov.tsv "
           f"--pairs {rp}/pairs.tsv --kinship-col GRM --kinship-min {REL_MIN} --no-closure --annot {W}/{grp}.mafld8.annot.gz "
           f"--out {rp}/realrel_unrel", quiet=True)
    js = json.load(open(f'{rp}/realrel_unrel.json'))['phenotypes']
    ph = pd.read_csv(f'{rp}/real_unrel.pheno.tsv', sep='\t', dtype={'FID': str, 'IID': str})
    for t in tr:   # the related-pair parts must come from the GWAS sample and covariates of each trait
        assert js[t]['n_covariates'] == len(COVS), f'{grp} {t}: REL_M covariates differ from the GWAS'
        assert same_sample(grp, f'{g}.panel', list(ph.IID[ph[t].notna()]), f'{rp}/ss_runrel/{t}.tsv', js[t]['n']), \
            f'{grp} {t}: the panel REL_M sample is not the panel GWAS sample'
        if not fresh(f'{rp}/ss_rufrel/{t}.tsv', f'{rp}/ss_runrel/{t}.tsv', f'{rp}/realrel_unrel.rel_m.tsv'):
            sh(f'{SUMMIT} relatives append --sumstats {rp}/ss_runrel/{t}.tsv --rel {rp}/realrel_unrel.rel_m.tsv --pheno {t} '
               f'--annot {W}/{grp}.mafld8.annot.gz --out {rp}/ss_rufrel/{t}.tsv.tmp', quiet=True)
            os.replace(f'{rp}/ss_rufrel/{t}.tsv.tmp', f'{rp}/ss_rufrel/{t}.tsv')
    # (b) every arm with jackknife replicates
    cmds = []
    for m in MODELS:
        for a, (ss, ld) in ARMS.items():
            c = fit(grp, f'{rp}/{ss}', f'{rp}/{ld}', f'{rp}/fitjk/{a}.{m}', m)
            cmds.append(c.replace(' --out ', ' --write-jack --out ', 1) if c else None)
    bad = run_many(cmds)
    assert not bad, f'failed fits, see their .stdout files: {bad[:2]}'
    for m in MODELS:
        res = {a: AR.read_results(f'{rp}/fitjk/{a}.{m}.results.tsv').set_index('phen') for a in ARMS}
        for t in tr:
            reps = {a: jack_reps(f'{rp}/fitjk/{a}.{m}', t) for a in ARMS}
            row = dict(group=grp, trait=t, model=m)
            for a in ARMS:
                h, s = float(res[a].h2[t]), float(res[a].se[t])
                assert abs(jk_se(reps[a]) / s - 1) < 1e-3, f'{grp} {t} {a}: SE from the replicates {jk_se(reps[a]):.6g} != SUMMIT {s:.6g}'
                if a in S3NAME:   # the refit reproduces R6
                    r6 = S3[(S3.group == grp) & (S3.trait == t) & (S3.model == m) & (S3.arm == S3NAME[a])].iloc[0]
                    assert abs(h - r6.h2) < 1e-6 and abs(s - r6.se) < 1e-6, f'{grp} {t} {a}: refit differs from stage3_h2.csv'
                    row[f'n_{a}'] = r6.n
                row[f'h2_{a}'], row[f'se_{a}'] = h, s
            row['n_UX'] = row['n_U']; row['pairs_excluded_UX'] = M(js[t]['pairs_used'])
            for a, b in DIFFS:   # K-X share sumstats and LD (same blocks); U, UX and X differ in sample, so their blocks may differ slightly
                row[f'd_{a}_{b}'] = row[f'h2_{a}'] - row[f'h2_{b}']
                row[f'se_d_{a}_{b}'] = jk_se(reps[a] - reps[b])
            A4.append(row)
A4 = pd.DataFrame(A4)
A4.to_csv(f'{OUT}/stage3_arms.csv', index=False)
show = ['group', 'trait', 'h2_U', 'h2_UX', 'h2_X', 'h2_K', 'd_X_U', 'se_d_X_U', 'd_X_UX', 'se_d_X_UX', 'd_K_X', 'se_d_K_X', 'pairs_excluded_UX']
print(A4[show].round(4).to_string(index=False))
print(f'wrote {OUT}/stage3_arms.csv')

# ======================================================================================================================================
# R7. Export: one JSON with every summary (aggregate statistics only; counts of 1-20 are shown as "<=20"). Download
# rel_results/aou_summit_relatives_v1_output.json 
os.chdir(KIT)   # every cell runs in this kit's folder
out = {'kit': 'summit_relatives_v1', 'summit_commit': SUMMIT_COMMIT, 'groups': GROUPS, 'models': MODELS,
       'parameters': dict(N_PCS=N_PCS, KING_FILTER=KING_FILTER, REL_MIN=REL_MIN, FIRST_DEGREE=FIRST_DEGREE, NVECS=NVECS, SEED=SEED),
       'samples': json.load(open(f'{OUT}/samples.json')) if done(f'{OUT}/samples.json') else None, 'pairs': {}}
for grp in GROUPS:
    f = f'{OUT}/{grp}_pairs_summary.json'
    if done(f): out['pairs'][grp] = {k: v for k, v in json.load(open(f)).items() if k != 'king_x_grm'}   # its totals would reveal masked cells
for stage, f in (('stage1', 'stage1_summary.csv'), ('stage2', 'stage2_summary.csv'), ('stage3', 'stage3_h2.csv')):
    out[stage] = pd.read_csv(f'{OUT}/{f}').round(6).to_dict(orient='records') if done(f'{OUT}/{f}') else None   # counts already masked
AR.write_json(out, f'{OUT}/aou_summit_relatives_v1_output.json'); print('written', f'{OUT}/aou_summit_relatives_v1_output.json')
