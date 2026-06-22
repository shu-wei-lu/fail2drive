import os
import subprocess
import time
import json
import argparse
from tqdm import tqdm
import random
import shutil
import shlex

FAIL2DRIVE_DIR = os.path.dirname(os.path.abspath(__file__))
HPC_ROOT = os.path.dirname(FAIL2DRIVE_DIR)
CARLA_IMAGE = os.path.join(HPC_ROOT, "carla-wrapper-native.sif")
CARLA_BIND_SOURCE = FAIL2DRIVE_DIR
LEADERBOARD_SCRIPT = os.path.join(
    FAIL2DRIVE_DIR, "leaderboard", "leaderboard", "leaderboard_evaluator_local.py"
)
DEFAULT_AGENT_FILE = os.path.join(FAIL2DRIVE_DIR, "team_code", "sensor_agent.py")
DEFAULT_AGENT_CONFIG = os.path.join(FAIL2DRIVE_DIR, "checkpoints", "tfpp")

FAIL2DRIVE_JOB_PREFIX = "Fail2Drive_"
MAX_JOBS_FILE = os.path.join(FAIL2DRIVE_DIR, "eval_num_jobs.txt")
RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}

# You will likely have to customize this function a bit to work with your cluster partition names etc.
# NOTE: Make sure to run this python script with the correct conda env which automatically sets the env vars. 
# If you have issues with the variables, such as WORK_DIR not being set, you can export them in the following bash file.
def bash_file(job, cfg, carla_world_port_start, carla_streaming_port_start, carla_tm_port_start):
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]
    carla_command = (
        f"apptainer exec --nv "
        f"--bind {shlex.quote(CARLA_BIND_SOURCE)}:/workspace "
        f"{shlex.quote(CARLA_IMAGE)} "
        f"/workspace/f2d_carla/CarlaUE4.sh "
        f"-carla-rpc-port=${{FREE_WORLD_PORT}} "
        f"-nosound -RenderOffScreen -graphicsadapter=0 "
        f"-carla-primary-port=0 "
        f"-carla-streaming-port=${{FREE_STREAMING_PORT}}"
    )
    with open(job_file, 'w', encoding='utf-8') as rsh:
            rsh.write(f'''#!/bin/bash
#SBATCH --job-name=Fail2Drive_{seed}_{route_id}
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50gb
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1

# NOTE: Make sure that the time limit is enough for your model to fail the route!
# Example: Timeout may take 3-4 minutes in game time, if cluster time limit enables only 2:30 game time, the route will be resubmitted until it succeeds!

echo JOB ID $SLURM_JOB_ID
echo "Node: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

monitor_resources() {{
    while true; do
        echo "===== RESOURCE SNAPSHOT $(date -Is) ====="
        echo "--- GPU ---"
        nvidia-smi || true
        echo "--- CPU / MEM processes ---"
        ps -u "$USER" -o pid,ppid,pcpu,pmem,rss,etime,comm,args --sort=-pcpu | head -25 || true
        echo "--- Memory ---"
        free -h || true
        echo "--- Load ---"
        uptime || true
        echo "--- Disk I/O ---"
        if command -v iostat >/dev/null 2>&1; then
            iostat -xz 1 1 || true
        elif command -v pidstat >/dev/null 2>&1; then
            pidstat -d 1 1 || true
        else
            cat /proc/loadavg || true
        fi
        sleep 30
    done
}}
monitor_resources &
MONITOR_PID=$!

# NOTE: You can use this in your agent to store visualization outputs
export VIZ_PATH={viz_path}

FREE_WORLD_PORT=`comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 49} | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
echo 'World Port:' $FREE_WORLD_PORT

FREE_STREAMING_PORT=`comm -23 <(seq {carla_streaming_port_start} {carla_streaming_port_start + 49} | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
echo 'Streaming Port:' $FREE_STREAMING_PORT

export TM_PORT=`comm -23 <(seq {carla_tm_port_start} {carla_tm_port_start+49} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT

# Start CARLA inside the Apptainer image and clean it up when the job exits.
{carla_command} &
CARLA_PID=$!
cleanup() {{
    kill "$MONITOR_PID" 2>/dev/null || true
    kill "$CARLA_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    wait "$CARLA_PID" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

sleep 60  # Wait for CARLA to finish starting
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
    echo "CARLA exited before the evaluator started" >&2
    wait "$CARLA_PID"
    exit 1
fi
nvidia-smi || true

# Run the RGB sensor agent with visualization and challenge debug output disabled.
LIVE_VISU=0 \\
DEBUG_CHALLENGE=0 \\
python -u {shlex.quote(LEADERBOARD_SCRIPT)} \\
--routes={shlex.quote(route)} \\
--repetitions=1 \\
--track=SENSORS \\
--checkpoint={shlex.quote(result_file)} \\
--timeout=300 \\
--agent={shlex.quote(cfg["agent_file"])} \\
--agent-config={shlex.quote(cfg["agent_config"])} \\
--port=${{FREE_WORLD_PORT}} \\
--traffic-manager-port=${{TM_PORT}} \\
--traffic-manager-seed={seed}
''')

def get_running_jobs():
    try:
        squeue_out = subprocess.check_output(
            f'squeue --me --noheader --format "%A|%j" | grep -F "|{FAIL2DRIVE_JOB_PREFIX}" || true',
            shell=True,
        ).decode("utf-8").splitlines()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] Failed to query running jobs from slurm: {exc}")
        return set()
    return {line.split("|", 1)[0].strip() for line in squeue_out if line.strip()}

def _is_retryable_result(evaluation_data):
    checkpoint = evaluation_data.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return True

    progress = checkpoint.get("progress")
    records = checkpoint.get("records")
    if not isinstance(progress, list) or not isinstance(records, list):
        return True

    if len(progress) < 2 or progress[0] < progress[1] or len(records) == 0:
        return True

    for record in records:
        if not isinstance(record, dict):
            return True
        if record.get("status") in RETRYABLE_STATUSES:
            return True

    return False

def get_max_num_parallel_jobs():
    try:
        with open(MAX_JOBS_FILE, "r", encoding="utf-8") as f:
            max_num_parallel_jobs = int(f.read().strip())
    except (OSError, ValueError) as exc:
        print(f"[warn] Failed to read max parallel jobs from '{MAX_JOBS_FILE}': {exc}. Falling back to 1.")
        return 1
    return max_num_parallel_jobs

def filter_completed(jobs):
    filtered_jobs = []

    running_jobs = get_running_jobs()
    for job in jobs:

        # If job is running we keep it in list (other function does killing)
        if "job_id" in job:
           if job["job_id"] in running_jobs:
              filtered_jobs.append(job)
              continue

        # Keep failed jobs to resubmit
        result_file = job["result_file"]
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    evaluation_data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[warn] Could not read '{result_file}': {exc}")
                if job["tries"] > 0:
                    filtered_jobs.append(job)
                continue

            need_to_resubmit = _is_retryable_result(evaluation_data)

            if need_to_resubmit and job["tries"] > 0:
                filtered_jobs.append(job)
        # Results file doesnt exist
        elif job["tries"] > 0:
            filtered_jobs.append(job)
    return filtered_jobs

def kill_dead_jobs(jobs):
    running_jobs = get_running_jobs()
    for job in jobs:

        if "job_id" in job:
            job_id = job["job_id"]

        elif os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")

        else:
            continue

        if job_id not in running_jobs:
            continue

        log = job["log_file"]
        if not os.path.exists(job["log_file"]):
            continue

        with open(log) as f:
            lines = f.readlines()
        if len(lines)==0:
            continue

        if any(["Watchdog exception" in line for line in lines]) or \
            "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.\n" in lines or \
            "[91mStopping the route, the agent has crashed:\n" in lines or \
            "[91mError during the simulation:\n" in lines:

            subprocess.Popen(f"scancel {job_id}", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str,
                      default=os.path.join(FAIL2DRIVE_DIR, 'fail2drive_split'),
                      help='Path to folder containing the split route files')
    parser.add_argument('--out_root', type=str,
                      default=os.path.join(FAIL2DRIVE_DIR, 'results', 'fail2drive'),
                      help='Path where results should be stored')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                      help='The seeds to evaluate')
    parser.add_argument('--retries', type=int, default=3,
                      help='Maximum number of retries per route')
    parser.add_argument('--agent', type=str, default=DEFAULT_AGENT_FILE,
                      help='Path to agent entry file')
    parser.add_argument('--agent-config', type=str, default=DEFAULT_AGENT_CONFIG,
                      help='Path to agent config/checkpoint')

    args = parser.parse_args()

    if not os.path.isfile(CARLA_IMAGE):
        parser.error(f"CARLA Apptainer image not found: {CARLA_IMAGE}")
    carla_script = os.path.join(CARLA_BIND_SOURCE, "f2d_carla", "CarlaUE4.sh")
    if not os.path.isfile(carla_script):
        parser.error(f"CARLA launch script not found: {carla_script}")

    routes = sorted([x for x in os.listdir(args.routes) if x[-4:]==".xml"])

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    seeds = args.seeds
    retries = args.retries

    cfg = {
        "viz": True,
        "agent_file": os.path.abspath(os.path.expanduser(args.agent)),
        "agent_config": os.path.abspath(os.path.expanduser(args.agent_config)),
    }

    # Filling the job queue
    job_queue = []
    for seed in seeds:

        base_dir = os.path.join(out_root, str(seed))
        os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

        for route in routes:
            route_id = route.split("_")[-1][:-4]
            route_seed = int(route_id) % 1000 + (10000 * seed) # NOTE: Fail2Drive specific, pairs are route_id%1000
            route = os.path.join(args.routes, route)

            viz_path = ""
            if cfg["viz"]:
                viz_path = os.path.join(base_dir, "viz", route_id)

            result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
            log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
            err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")

            job_file = os.path.join(base_dir, "run", f'eval_{route_id}.sh')

            job = {
                "route": route,
                "route_id": route_id,
                "seed": route_seed,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "viz_path": viz_path,
                "job_file": job_file,
                "tries": retries
            }

            job_queue.append(job)

    carla_world_ports = list(range(10000, 20000, 50))
    carla_streaming_ports = list(range(20000, 30000, 50))
    carla_tm_ports = list(range(30000, 40000, 50))
    random.shuffle(carla_world_ports)
    random.shuffle(carla_streaming_ports)
    random.shuffle(carla_tm_ports)
    port_idx = 0

    # Submitting the jobs to slurm
    jobs = len(job_queue)
    progress = tqdm(total = jobs)
    while job_queue:
        kill_dead_jobs(job_queue)
        job_queue = filter_completed(job_queue)

        progress.update(jobs - len(job_queue) - progress.n)

        running_jobs = get_running_jobs()
        max_num_parallel_jobs = get_max_num_parallel_jobs()
        if len(running_jobs) >= max_num_parallel_jobs:
            time.sleep(5)
            print("Waiting for 5 seconds...")
            continue

        for job in job_queue:
            if job["tries"] <= 0:
                continue

            if "job_id" in job and job["job_id"] in running_jobs:
                continue

            if os.path.exists(job["log_file"]):
                with open(job["log_file"], "r") as f:
                    job_id = f.readline().strip().replace("JOB ID ", "")
                    if job_id in running_jobs:
                        continue

            # Need to submit this job
            carla_world_port_start = carla_world_ports[port_idx]
            carla_streaming_port_start = carla_streaming_ports[port_idx]
            carla_tm_port_start = carla_tm_ports[port_idx]
            port_idx += 1
            port_idx %= 200

            # Make bash file:
            bash_file(job, cfg, carla_world_port_start, carla_streaming_port_start, carla_tm_port_start)

            # submit
            if cfg["viz"]:
                if os.path.exists(job["viz_path"]):
                    shutil.rmtree(job["viz_path"])
                os.makedirs(job["viz_path"], exist_ok=True)

            for file in [job["result_file"], job["err_file"], job["log_file"]]:
                if os.path.exists(file):
                    os.remove(file)
            try:
                job_id = subprocess.check_output(
                    f'sbatch {job["job_file"]}', shell=True
                ).decode('utf-8').strip().rsplit(' ', maxsplit=1)[-1]
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"[warn] Failed to submit job '{job['job_file']}': {exc}")
                continue
            
            job["job_id"] = job_id
            job["tries"] -= 1

            print(f'submit {job["job_file"]} {job_id}')
            print(len(job_queue))
            break

        time.sleep(2)
