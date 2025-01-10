from spectacles.client import LookerClient
from spectacles.runner import Runner
import json
import os

f = open("branch_comparison/scenarios.json", "r")
SCENARIOS = json.load(f)


def get_master_results(
    base_url, client_id, client_secret, project_name, validation_args
):

    looker_client = LookerClient(
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret,
    )

    runner = Runner(
        looker_client,
        project_name,
        remote_reset=True,
    )

    master_results = runner.validate_sql(**validation_args)

    return master_results


if __name__ == "__main__":

    base_url = "https://spectacles.looker.com"
    client_id = os.getenv("LOOKER_CLIENT_ID")
    client_secret = os.getenv("LOOKER_CLIENT_SECRET")

    for index, scenario in enumerate(SCENARIOS):
        results = get_master_results(base_url, client_id, client_secret, **scenario)
        with open(f"branch_comparison/results/master_{index}.json", "w") as f:
            json.dump(results, f)
