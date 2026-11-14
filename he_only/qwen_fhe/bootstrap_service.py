"""Single-GPU genuine refresh service. Loads only public keys and ciphertexts."""
import argparse
import json
import os
from pathlib import Path
import socket
from .cipher_rpc import receive_frame, send_frame, receive_metadata, send_metadata
from .perseus_context import load_programs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    from perseus import _pc as pc
    bundle = Path(args.bundle)
    spec = json.loads((bundle/'parameters.json').read_text())
    if spec.get('evaluation_layout') != 'separate_arithmetic_and_bootstrap_gpus':
        raise ValueError('Expected a split public bundle')
    cfg = pc.Config.from_json((bundle/'config.json').read_text())
    programs = load_programs(bundle, cfg)
    keys = pc.KeySource.generate([], layout='packed')
    session = pc.Session.from_bundle(cfg, keys, str(bundle/'public.pckb'),
        programs=list(programs.values()), regs=1,
        reg_levels=[int(spec['bootstrap_raw_landing'])],
        pool=int(spec['bootstrap_pool']), pt_slots=1)
    for program in programs.values():
        program.bind(session)
    session.complete_setup()
    if session.has_secret or session.fhe.has_secret_key or session.fhe.debug_refresh:
        raise RuntimeError('Refresh service must have no secret or debug refresh')
    status = {'server_has_secret': False, 'debug_refresh': False,
              'genuine_planned_bootstraps': [p.n_bootstraps for p in programs.values()],
              'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'transport': 'ciphertext_only'}
    print(json.dumps({'stage': 'bootstrap_service_ready', **status,
                      'memory': str(session.mem_report)}), flush=True)
    address = Path(args.socket)
    listener = socket.socket(socket.AF_UNIX)
    completed = 0
    try:
        listener.bind(str(address))
        os.chmod(address, 0o600)
        listener.listen(1)
        connection, _ = listener.accept()
        with connection:
            while True:
                try:
                    request = receive_metadata(connection)
                except EOFError:
                    break
                if request == {'op': 'status'}:
                    send_metadata(connection, status)
                    continue
                if request == {'op': 'shutdown'}:
                    send_metadata(connection, {'completed': completed})
                    break
                if set(request) != {'op', 'level'} or request['op'] != 'bootstrap':
                    raise ValueError('Unsupported ciphertext request')
                level = request['level']
                if type(level) is not int or level not in programs:
                    raise ValueError('Unsupported public input level')
                value = pc.import_ct(session, receive_frame(connection))
                if value.level != level or value.noise_deg != 1:
                    raise ValueError('Ciphertext does not match public input metadata')
                value.tag = pc.Tag(slots=32768, period=2048, field=pc.tag.F_REAL)
                metrics = dict(programs[level].run([value], outputs=[(0, 0)]))
                if metrics['meta_mismatches']:
                    raise RuntimeError('Planned refresh metadata mismatch')
                result = pc.Program.reg_ct(session, 0)
                send_frame(connection, pc.export_ct(session, result))
                completed += 1
                del value, result
    finally:
        listener.close()
        address.unlink(missing_ok=True)
        print(json.dumps({'stage': 'bootstrap_service_closed', 'completed': completed}), flush=True)
        session.close()


if __name__ == '__main__':
    main()
