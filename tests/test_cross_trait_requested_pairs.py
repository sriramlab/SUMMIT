import importlib.util
import json
from pathlib import Path

import pytest


def test_requested_pairs_preserve_names_orientation_and_reject_duplicates(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/generalized_gxe/cross_trait_study.py'
    spec=importlib.util.spec_from_file_location('requested_study',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'pairs.json';path.write_text(json.dumps([['c','a'],['a','b']]))
    names,pairs=module.select_trait_pairs(['a','b','c'],['c','b','a'],path)
    assert names==('c','b','a') and pairs==[(0,2),(2,1)]
    for value in ([],[['a','a']],[['a','b'],['b','a']],[['a','missing']]):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):module.select_trait_pairs(names,names,path)
    with pytest.raises(ValueError):module.select_trait_pairs(names,['a','a'])
