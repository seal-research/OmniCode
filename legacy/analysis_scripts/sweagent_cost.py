from pathlib import Path
import json


def compute_cost(
    results_dir: Path | str,    
):
    results_dir = Path(results_dir)

    costs = []
    tokens_sent = []
    tokens_gen = []
    api_calls = []

    for instance_dir in results_dir.iterdir():

        if not instance_dir.is_dir():
            continue

        traj_files = list(instance_dir.rglob("*.traj"))
        
        if len(traj_files) == 0:
            continue

        traj_file = traj_files[0]
        traj_data = json.loads(traj_file.read_text())
        if "info" in traj_data:
            if "model_stats" in traj_data["info"]:
                if "instance_cost" in traj_data["info"]["model_stats"]:
                    costs.append(traj_data["info"]["model_stats"]["instance_cost"])
                if "tokens_sent" in traj_data["info"]["model_stats"]:
                    tokens_sent.append(traj_data["info"]["model_stats"]["tokens_sent"])
                if "tokens_received" in traj_data["info"]["model_stats"]:
                    tokens_gen.append(traj_data["info"]["model_stats"]["tokens_received"])
                if "api_calls" in traj_data["info"]["model_stats"]:
                    api_calls.append(traj_data["info"]["model_stats"]["api_calls"])
 
    def mean(l: list): return sum(l) / len(l)

    print(f"{mean(costs)=:.4f}, {mean(tokens_sent)=:.1f}, {mean(tokens_gen)=:.1f}, {mean(api_calls)=:.1f}")


    

if __name__ == '__main__':
    import fire
    fire.Fire(compute_cost)