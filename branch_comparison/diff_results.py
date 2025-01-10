from jsondiff import diff
import json

f = open("branch_comparison/scenarios.json", "r")
SCENARIOS = json.load(f)

if __name__ == "__main__":

    for index, scenario in enumerate(SCENARIOS):
        master_file = open(f"branch_comparison/results/master_{index}.json", "r")
        master_results = json.load(master_file)

        async_file = open(f"branch_comparison/results/async_{index}.json", "r")
        async_results = json.load(async_file)

        d = diff(master_results, async_results)

        if d:
            print(f"{scenario}: {d}")
