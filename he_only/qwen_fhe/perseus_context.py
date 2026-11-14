"""Public parameters and bootstrap programs for the sequential SM86 backend.

All logical vectors repeat across the native slots with period dividing 2048.
The plans contain no plaintext user data or private key material.
"""
import json
from pathlib import Path


def configuration(level_budget=(3, 3)):
    from perseus import _pc as pc
    cfg = pc.Config.preset('n32_gpt2_decode')
    fields = {'bts.stc_first': 0, 'bts.sparse_auto': 2,
        'bts.sparse_bts_slots': ['2048'], 'bts.sparse_b': False,
        'bts.sparse_level_budget': [str(x) for x in level_budget], 'bts.real': True,
        'bts.bts_iterations': 1, 'bts.k_max': 1,
        'bts.auto_bts_level': 48, 'exec.use_graphs': False}
    for name, value in fields.items():
        cfg[name] = value
    return cfg


def write_context(directory, cfg):
    directory = Path(directory)
    graph = directory/'graph/block_0'
    graph.mkdir(parents=True, exist_ok=False)
    budgets = [int(x) for x in cfg['bts.sparse_level_budget']]
    if budgets not in [[3, 3], [4, 4]]:
        raise ValueError('Unsupported public Classic bootstrap budget')
    landing = 2*(sum(budgets)+10)
    raw_landing = landing-2
    tag = {'slots': 32768, 'period': 2048, 'kind': 2, 'offset': 0,
           'stride': 1, 'count': 0, 'width': 1, 'field': 1}
    node = {'op_type': 'deliberate_bootstrap', 'inputs': ['input'],
        'input_levels': [48], 'output': 'v_1', 'output_level': raw_landing,
        'output_noise_level': 2, 'tag': tag,
        'shadow': {'level': raw_landing, 'deg': 2, 'nf': 0.0},
        'args': {'route': {'slots': 2048, 'stc_first': False,
                           'real': True, 'sparse_b': False}}}
    (graph/'graph.json').write_text(json.dumps({'version': 2, 'nodes': [node]}))
    for folder in [graph, graph.parent]:
        (folder/'capture_config.json').write_text(cfg.to_json())
    (directory/'config.json').write_text(cfg.to_json())
    plan = {'version': 2,
        'rules': {'bootstrap_level': landing, 'max_level': 54, 'source_level': 48,
                  'sparse_bts_slots': [2048]},
        'placements': [], 'hint_fire': [], 'rescale_after': [],
        'config': json.loads(cfg.load_bearing().to_json()),
        'sparse_slots': {'v_1': 2048},
        'summary': {'exit_var': 'v_1', 'exit_level': raw_landing, 'exit_deg': 2}}
    (directory/'plan').mkdir()
    (directory/'plan/block_0_placement.json').write_text(json.dumps(plan))


def load_programs(directory, cfg):
    from perseus import _pc as pc
    directory = Path(directory)
    programs = {}
    for level in [46, 48]:
        program = pc.Program.compile(str(directory/'graph'), str(directory/'plan'),
            config=cfg, no_batch=True, routes_built=[2048],
            input_metas={'block_0': {'input': (level, 1)}})['block_0']
        program.relabel('bootstrap_%d' % level)
        if program.n_bootstraps != 1:
            raise ValueError('Expected exactly one genuine planned bootstrap')
        programs[level] = program
    return programs


def rotation_steps(width):
    result, shift = [], 1
    while shift < width:
        result.extend([shift, -shift])
        shift *= 2
    return sorted(result)
