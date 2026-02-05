import subprocess
import os
import json
import logging

from utils.extract_task_code import file_to_string


def set_freest_gpu(force: bool = False):
    """
    Set CUDA_VISIBLE_DEVICES to the "freest" GPU on the machine.

    This is best-effort: if GPU query tools (e.g. gpustat) are unavailable,
    it will fall back gracefully instead of crashing the entire job.
    """
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing and not force:
        logging.info(
            "CUDA_VISIBLE_DEVICES is already set (%s); not overriding. "
            "Pass force=True to override.",
            existing,
        )
        return existing

    freest_gpu = get_freest_gpu()
    if freest_gpu is None:
        logging.warning("Could not determine freest GPU; leaving CUDA_VISIBLE_DEVICES unchanged.")
        return None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(freest_gpu)
    return freest_gpu


def get_freest_gpu():
    """
    Return the index of the GPU with the most free memory (best-effort).

    Prefers `gpustat --json` if available; otherwise falls back to `nvidia-smi`.
    Returns None if GPU information can't be determined.
    """
    # 1) Try gpustat (fast and simple), but it's an optional dependency.
    try:
        sp = subprocess.Popen(["gpustat", "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str, err_str = sp.communicate()
        if sp.returncode != 0:
            raise RuntimeError(err_str.decode("utf-8", errors="replace").strip())

        gpustats = json.loads(out_str.decode("utf-8"))
        gpus = gpustats.get("gpus", [])
        if not gpus:
            return None

        # Choose GPU with most free memory = min used (gpustat reports per-GPU totals too,
        # but memory.used is the most consistent field across versions).
        freest_gpu = min(gpus, key=lambda x: x.get("memory.used", float("inf")))
        return freest_gpu.get("index")
    except FileNotFoundError:
        logging.warning("`gpustat` not found; falling back to `nvidia-smi` for GPU selection.")
    except Exception as e:
        logging.warning("Failed to query GPUs via gpustat; falling back to nvidia-smi. Error: %s", e)

    # 2) Fallback: nvidia-smi (commonly present on GPU servers).
    try:
        # Example output lines (noheader,nounits):
        # "0, 123, 24576"
        # "1, 10, 24576"
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or "").strip())

        candidates = []
        for raw_line in (cp.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx_s, used_s, total_s = parts[0], parts[1], parts[2]
            try:
                idx = int(idx_s)
                used = float(used_s)
                total = float(total_s)
            except ValueError:
                continue
            free = max(total - used, 0.0)
            candidates.append((free, -used, idx))

        if not candidates:
            return None

        # Maximize free memory; tie-break with lower used; then lower index.
        _, _, best_idx = max(candidates)
        return best_idx
    except FileNotFoundError:
        logging.warning("`nvidia-smi` not found; cannot auto-select GPU.")
        return None
    except Exception as e:
        logging.warning("Failed to query GPUs via nvidia-smi; cannot auto-select GPU. Error: %s", e)
        return None


def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break


if __name__ == "__main__":
    print(get_freest_gpu())