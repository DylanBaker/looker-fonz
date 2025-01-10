from httpx import AsyncClient
import asyncio
import json
from spectacles.client import LookerClient
from spectacles.runner import Runner
import os

f = open("branch_comparison/scenarios.json", "r")
SCENARIOS = json.load(f)


async def get_async_results(
    base_url, client_id, client_secret, project_name, validation_args
):

    async_client = AsyncClient()

    looker_client = LookerClient(
        async_client=async_client,
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret,
    )

    runner = Runner(looker_client, project_name, remote_reset=True)

    async_results = await runner.validate_sql(**validation_args)

    return async_results


if __name__ == "__main__":

    base_url = "https://spectacles.looker.com"
    client_id = os.getenv("LOOKER_CLIENT_ID")
    client_secret = os.getenv("LOOKER_CLIENT_SECRET")

    for index, scenario in enumerate(SCENARIOS):
        results = asyncio.run(
            get_async_results(base_url, client_id, client_secret, **scenario)
        )
        with open(f"branch_comparison/results/async_{index}.json", "w") as f:
            json.dump(results, f)
