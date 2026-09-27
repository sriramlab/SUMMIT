"""Exercise launcher argument construction without invoking the cluster runtime."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize('chromosome,override,expected', [
    ('1', None, '39600'), ('22', None, '18000'), ('22', '7200', '7200'),
])
@pytest.mark.parametrize('extra_input', [False, True])
def test_study_launcher_preserves_optional_paths_and_runtime(tmp_path, chromosome, override, expected, extra_input):
    script=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_followup_h2.sh'
    fake=tmp_path/'capture.py'
    fake.write_text(f'#!{sys.executable}\nimport json,sys\n'
                    'if sys.argv[1] == "-c": print("a\\nb")\n'
                    'else: print(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o700)
    text=script.read_text().replace(
        'python_exe=/u/home/b/bronsonj/.conda/envs/summit/bin/python',
        'python_exe='+shlex.quote(str(fake)))
    # The native executable is replaced; its cluster-only loader paths are
    # irrelevant to constructing arguments and unavailable on CI hosts.
    text='\n'.join(line for line in text.splitlines()
                   if not line.startswith(('export LD_PRELOAD=', 'export LD_LIBRARY_PATH=')))
    env={key:value for key,value in os.environ.items()
         if key not in ('CROSS_TRAIT_EXTRA_INPUT_ROOT','CROSS_TRAIT_MAX_RUN_SECONDS')}
    env.update(NSLOTS='8', JOB_ID='1')
    if override is not None: env['CROSS_TRAIT_MAX_RUN_SECONDS']=override
    extra=str(tmp_path/'extra inputs')
    if extra_input: env['CROSS_TRAIT_EXTRA_INPUT_ROOT']=extra
    output=tmp_path/'output with spaces'
    completed=subprocess.run(['bash','-c',text,str(script),'study',str(output),chromosome],
                             env=env,capture_output=True,text=True,check=True)
    argv=json.loads(completed.stdout)
    assert argv[argv.index('--max-run-seconds')+1] == expected
    assert argv[argv.index('--output')+1] == str(output/'study'/f'chr{chromosome}')
    if extra_input: assert argv[argv.index('--extra-input-root')+1] == extra
    else: assert '--extra-input-root' not in argv
