#!/usr/bin/env python3
"""Stop the running queue at a completed model result and finalize partial results."""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--folder', required=True)
    p.add_argument('--runner-pid', required=True, type=int)
    args = p.parse_args()
    base = args.output.resolve()
    target = base / args.folder / 'result.json'
    command_line = Path('/proc/%d/cmdline' % args.runner_pid).read_bytes()
    if b'benchmark_ac.py' not in command_line or str(base).encode() not in command_line:
        raise RuntimeError('Runner PID does not match this experiment')
    # Kernel notifications avoid polling or consuming CPU during formal timing.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    wd = libc.inotify_add_watch(fd, os.fsencode(target.parent), 0x8 | 0x80)
    if wd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_add_watch')
    print('Waiting for', target, flush=True)
    while not target.exists():
        os.read(fd, 4096)
    os.close(fd)
    # Pause the parent before it can dispatch another model. Its current GPU
    # child has already exited before the aggregate result is written.
    os.kill(args.runner_pid, signal.SIGSTOP)
    result = json.loads(target.read_text())
    children_file = Path('/proc/%d/task/%d/children' % (args.runner_pid, args.runner_pid))
    if children_file.exists():
        for value in children_file.read_text().split():
            child = int(value)
            child_command = Path('/proc/%d/cmdline' % child)
            if child_command.exists():
                cmd = child_command.read_bytes()
                if b'benchmark_export.py' in cmd or b'trtexec' in cmd or b'benchmark_trt.py' in cmd:
                    os.kill(child, signal.SIGTERM)
    os.kill(args.runner_pid, signal.SIGTERM)
    os.kill(args.runner_pid, signal.SIGCONT)
    for _ in range(100):
        proc_stat = Path('/proc/%d/stat' % args.runner_pid)
        if not proc_stat.exists() or proc_stat.read_text().split()[2] == 'Z':
            break
        time.sleep(0.05)
    from benchmark_ac import MODELS, report
    from benchmark_trt import clock_state
    completed, pending = [], []
    for folder, name, _, _ in MODELS:
        path = base / folder / 'result.json'
        if path.exists() and json.loads(path.read_text())['status'] == 'ok':
            completed.append(name)
        else:
            pending.append(name)
    state = {'status': 'paused', 'paused_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'stopped_after': args.folder, 'current_model_status': result['status'],
             'completed_models': completed, 'pending_models': pending,
             'reason': 'User requested stopping after the current model and continuing tomorrow',
             'resume_command': ['python3', 'benchmark_ac.py', '--output', str(base), '--models', *pending]}
    (base / 'pause_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    (base / 'clock_state_at_pause.json').write_text(json.dumps(clock_state(), indent=2) + '\n')
    report(base)
    subprocess.run([sys.executable, str(Path(__file__).with_name('benchmark_audit.py')),
                    '--output', str(base), '--models', *completed], check=True)
    print('Queue paused. Completed:', ', '.join(completed), flush=True)
    print('Pending:', ', '.join(pending), flush=True)


if __name__ == '__main__':
    main()
