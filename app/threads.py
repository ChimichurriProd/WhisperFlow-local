"""Worker threads that run native ML code get a REAL stack.

macOS gives secondary pthreads 512KB of stack (the main thread gets 8MB), and
that ceiling produced an app-killing crash observed in the wild (2026-07-30,
DiagnosticReports/Python-*-105341.ips): SIGBUS / KERN_PROTECTION_FAILURE on
the mlx-stt worker inside mlx::core::detail::compile_dfs — MLX's graph
compiler recurses with graph depth, hit the guard page, and took the whole
process down with no Python traceback. 16MB of stack is VIRTUAL address space
(pages are only committed when touched), so on 64-bit this costs nothing and
removes the ceiling for every native library we run on worker threads (MLX,
onnxruntime, mlx-audio).
"""

import concurrent.futures
import threading

STACK_BYTES = 16 * 1024 * 1024


def _with_stack(fn):
    """Run *fn* (which spawns threads) while the big stack size is set,
    restoring the previous global afterwards (threading.stack_size is a
    process-wide setting read at thread creation)."""
    try:
        old = threading.stack_size(STACK_BYTES)
    except (ValueError, RuntimeError):
        return fn()  # platform refused: run with defaults rather than die
    try:
        return fn()
    finally:
        threading.stack_size(old)


def start_thread(target, name, daemon=True):
    """threading.Thread(...).start() with a 16MB stack."""
    def _spawn():
        t = threading.Thread(target=target, name=name, daemon=daemon)
        t.start()
        return t

    return _with_stack(_spawn)


def single_worker_pool(prefix):
    """A 1-thread ThreadPoolExecutor whose worker has a 16MB stack.

    The executor spawns its thread lazily on first submit, so a no-op job is
    pushed through synchronously HERE — otherwise the worker would be created
    later, under whatever stack size happens to be set then.
    """
    def _build():
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=prefix
        )
        pool.submit(lambda: None).result()
        return pool

    return _with_stack(_build)
