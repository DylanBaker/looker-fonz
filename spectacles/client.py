import asyncio
import json
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Tuple

import backoff
import httpx
from aiocache import Cache, cached, serializers
from httpx import HTTPStatusError, NetworkError, RemoteProtocolError, TimeoutException

import spectacles.utils as utils
from spectacles.exceptions import LookerApiError, SpectaclesException
from spectacles.logger import GLOBAL_LOGGER as logger
from spectacles.models import JsonDict

DEFAULT_API_VERSION = 4.0
TIMEOUT_SEC = 300
LOOKML_VALIDATION_TIMEOUT = 7200
MAX_ASYNC_CONNECTIONS = 200

DEFAULT_RETRIES = 3
DEFAULT_NETWORK_RETRIES = 10
NETWORK_EXCEPTIONS = (
    NetworkError,
    TimeoutException,
    RemoteProtocolError,
)
STATUS_EXCEPTIONS = (
    HTTPStatusError,
    LookerApiError,
)
BACKOFF_EXCEPTIONS = NETWORK_EXCEPTIONS + STATUS_EXCEPTIONS


def giveup_unless_bad_gateway(exception: Exception) -> bool:
    """Give up retries if a status error encountered with any code besides 502/504."""
    if isinstance(exception, LookerApiError):
        return exception.status not in (
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        )
    elif isinstance(exception, HTTPStatusError):
        return exception.response.status_code not in (
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        )
    else:
        return False


@dataclass(frozen=True)  # Token is immutable
class AccessToken:
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str]
    expires_at: float

    def __str__(self) -> str:
        return self.access_token

    @property
    def expired(self) -> bool:
        return False if time.time() < self.expires_at else True


def log_backoff(details: dict) -> None:
    logger.debug(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries. "
        f"Error: {details['exception'].__class__.__name__}"
    )


def backoff_with_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    @backoff.on_exception(
        backoff.expo,
        STATUS_EXCEPTIONS,
        giveup=giveup_unless_bad_gateway,
        max_tries=DEFAULT_RETRIES,
        on_backoff=log_backoff,
    )
    @backoff.on_exception(
        backoff.expo,
        NETWORK_EXCEPTIONS,
        max_tries=DEFAULT_NETWORK_RETRIES,
        on_backoff=log_backoff,
    )
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


class LookerClient:
    """Wraps some endpoints of the Looker API, issues requests and handles responses.

    Args:
        base_url: Base URL for the Looker instance, e.g. https://mycompany.looker.com.
        client_id: Looker API client ID.
        client_secret: Looker API client secret.
        port: Desired API port to use for requests.
        api_version: Desired API version to use for requests.

    Attributes:
        api_url: Combined URL used as a base for request building.
        session: Persistent session to avoid re-authenticating.

    """

    def __init__(
        self,
        async_client: httpx.AsyncClient,
        base_url: str,
        client_id: str,
        client_secret: str,
        port: Optional[int] = None,
        api_version: float = DEFAULT_API_VERSION,
    ):
        self.async_client = async_client
        supported_api_versions = [4.0]
        if api_version not in supported_api_versions:
            raise SpectaclesException(
                name="unsupported-api-version",
                title="Specified API version is not supported.",
                detail=(
                    f"Version '{api_version}' is not supported. "
                    "Please use one of these supported versions instead: "
                    f"{', '.join(str(ver) for ver in sorted(supported_api_versions))}"
                ),
            )

        self.base_url: str = base_url.rstrip("/")
        if port is None and self.base_url.endswith("cloud.looker.com"):
            # GCP-hosted instance, so use default port of 443 with HTTPS
            if not self.base_url.startswith("https"):
                raise SpectaclesException(
                    name="invalid-base-url",
                    title="Looker instance base URL is not valid.",
                    detail="The URL must be an HTTPS URL.",
                )
            self.api_url: str = f"{self.base_url}/api/{api_version}/"
        else:
            self.api_url = f"{self.base_url}:{port or 19999}/api/{api_version}/"
        self.client_id: str = client_id
        self.client_secret: str = client_secret
        self.api_version: float = api_version
        self.access_token: Optional[AccessToken] = None
        self.workspace: str = "production"

        self.authenticate()

    def authenticate(self) -> None:
        """Logs in to Looker's API using a client ID/secret pair and an API version.

        Args:
            client_id: Looker API client ID.
            client_secret: Looker API client secret.
            api_version: Desired API version to use for requests.

        """
        logger.debug(f"Authenticating to the Looker as client ID '{self.client_id}'")

        url = utils.compose_url(self.api_url, path=["login"])
        body = {"client_id": self.client_id, "client_secret": self.client_secret}
        # This should not use `self.post` or it will create a recursive loop
        response = httpx.post(url=url, data=body, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-authenticate",
                title="Couldn't authenticate to the Looker API.",
                status=response.status_code,
                detail=(
                    f"Unable to authenticate with client ID '{self.client_id}'. "
                    "Check that your credentials are correct and try again."
                ),
                response=response,
            ) from error

        result = response.json()
        if "expires_at" not in result:
            # Calculate the expiration time with a one-minute buffer
            result["expires_at"] = time.time() + result["expires_in"] - 60
        self.access_token = AccessToken(**result)
        self.async_client.headers = httpx.Headers(
            {"Authorization": f"token {self.access_token}"}
        )

        looker_version = self.get_looker_release_version()
        logger.info(
            f"Connected to Looker version {looker_version} "
            f"using Looker API {self.api_version}"
        )

    async def request(
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        if self.access_token and self.access_token.expired:
            logger.debug("Looker API access token has expired, requesting a new one")
            self.authenticate()
            if self.workspace == "dev":
                await self.update_workspace("dev")
        return await self.async_client.request(method, url, *args, **kwargs)

    async def get(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, *args, **kwargs)

    async def post(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, *args, **kwargs)

    async def patch(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("PATCH", url, *args, **kwargs)

    async def put(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("PUT", url, *args, **kwargs)

    async def delete(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("DELETE", url, *args, **kwargs)

    @backoff.on_exception(
        backoff.expo,
        BACKOFF_EXCEPTIONS,
        giveup=giveup_unless_bad_gateway,
        max_tries=DEFAULT_NETWORK_RETRIES,
    )
    def get_looker_release_version(self) -> str:
        """Gets the version number of connected Looker instance.

        Returns:
            str: Looker instance release version number (e.g. 6.22.12)

        """
        logger.debug("Checking Looker instance release version")

        url = utils.compose_url(self.api_url, path=["versions"])

        response = httpx.get(
            url=url, timeout=TIMEOUT_SEC, headers=self.async_client.headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-version",
                title="Couldn't get Looker's release version.",
                status=response.status_code,
                detail=(
                    "Unable to get the release version of your Looker instance. "
                    "Please try again."
                ),
                response=response,
            ) from error

        return response.json()["looker_release_version"]  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def get_workspace(self) -> str:
        """Gets the session workspace.

        Args:
            project: Name of the Looker project to use.

        Returns:
            str: The session workspace, dev or production.
        """
        logger.debug("Getting the workspace in use by this session")
        url = utils.compose_url(self.api_url, path=["session"])
        response = await self.get(url=url, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-workspace",
                title="Couldn't get the workspace.",
                status=response.status_code,
                detail=(
                    "Unable to get the workspace in use by this session. "
                    "Please try again."
                ),
                response=response,
            ) from error
        return response.json()["workspace_id"]  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def update_workspace(self, workspace: str) -> None:
        """Updates the session workspace.

        Args:
            workspace: The workspace to switch to, either 'production' or 'dev'
        """
        logger.debug(f"Updating session to use the {workspace} workspace")
        url = utils.compose_url(self.api_url, path=["session"])
        body = {"workspace_id": workspace}
        response = await self.patch(url=url, json=body, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-update-workspace",
                title="Couldn't update the session's workspace.",
                status=response.status_code,
                detail=(
                    f"Unable to update workspace to '{workspace}'. "
                    "If you have any unsaved work on the branch "
                    "checked out by the user whose API credentials "
                    "Spectacles is using, please save it and try again."
                ),
                response=response,
            ) from error
        self.workspace = workspace

    @backoff_with_exceptions
    async def get_all_branches(self, project: str) -> List[JsonDict]:
        """Returns a list of git branches in the project repository.

        Args:
            project: Name of the Looker project to use.
        """
        logger.debug(f"Getting all Git branches in project '{project}'")
        url = utils.compose_url(
            self.api_url, path=["projects", project, "git_branches"]
        )
        response = await self.get(url=url, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-branches",
                title="Couldn't get all Git branches.",
                status=response.status_code,
                detail=(
                    f"Unable to get all Git branches in project '{project}'. "
                    "Please try again."
                ),
                response=response,
            ) from error

        return response.json()  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def checkout_branch(self, project: str, branch: str) -> None:
        """Checks out a new git branch. Only works in dev workspace.

        Args:
            project: Name of the Looker project to use.
            branch: Name of the Git branch to check out.
        """
        logger.debug(f"Setting project '{project}' branch to '{branch}'")
        url = utils.compose_url(self.api_url, path=["projects", project, "git_branch"])
        body = {"name": branch}
        response = await self.put(url=url, json=body, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-checkout-branch",
                title="Couldn't checkout Git branch.",
                status=response.status_code,
                detail=(
                    f"Unable to checkout Git branch '{branch}'. "
                    "If you have uncommitted changes on the current branch, "
                    "please commit or revert them, then try again."
                ),
                response=response,
            ) from error

    @backoff_with_exceptions
    async def reset_to_remote(self, project: str) -> None:
        """Reset a project development branch to the revision of the project that is on the remote.

        Args:
            project: Name of the Looker project to use.

        """
        logger.debug("Resetting branch to remote.")
        url = utils.compose_url(
            self.api_url, path=["projects", project, "reset_to_remote"]
        )
        response = await self.post(url=url, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-reset-remote",
                title="Couldn't checkout Git branch.",
                status=response.status_code,
                detail=(
                    "Unable to reset local Git branch "
                    "to match remote. Please try again."
                ),
                response=response,
            ) from error

    @backoff_with_exceptions
    async def get_manifest(self, project: str) -> JsonDict:
        """Gets all the dependent LookML projects defined in the manifest file.

        Args:
            project: Name of the Looker project to use.

        Returns:
            List[JsonDict]: JSON response containing all dependent projects
        """
        logger.debug("Getting manifest details")
        url = utils.compose_url(self.api_url, path=["projects", project, "manifest"])
        response = await self.get(url=url, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-manifest",
                title="Couldn't retrieve project manifest.",
                status=response.status_code,
                detail=(
                    f"Failed to retrieve manifest for project '{project}'. "
                    "Make sure you have a 'manifest.lkml' file in your project, "
                    "then try again."
                ),
                response=response,
            ) from error

        manifest = response.json()

        return manifest  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def get_active_branch(self, project: str) -> JsonDict:
        """Gets the active branch for the user in the given project.

        Args:
            project: Name of the Looker project to use.

        Returns:
            str: Name of the active branch
        """
        logger.debug(f"Getting active branch for project '{project}'")
        url = utils.compose_url(self.api_url, path=["projects", project, "git_branch"])
        response = await self.get(url=url, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-active-branch",
                title="Couldn't determine active Git branch.",
                status=response.status_code,
                detail=(
                    f"Unable to get active branch for project '{project}'. "
                    "Please check that the project exists, is configured, "
                    "and that your user has the correct permissions and try again."
                ),
                response=response,
            ) from error

        branch_name = response.json()["name"]
        logger.debug(f"The active branch is '{branch_name}'")

        return response.json()  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def get_active_branch_name(self, project: str) -> str:
        """Helper method to return only the branch name."""
        full_response = await self.get_active_branch(project)
        return full_response["name"]  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def create_branch(
        self, project: str, branch: str, ref: Optional[str] = None
    ) -> None:
        """Creates a branch in the given project.

        Args:
            project: Name of the Looker project to use.
            branch: Name of the branch to create.
            ref: The ref to create the branch from.
        """
        body = {"name": branch}
        message = f"Creating branch '{branch}' on project '{project}'"
        detail = (
            f"Unable to create branch '{branch}' "
            f"in project '{project}'. "
            "Confirm the branch doesn't already exist and try again."
        )

        if ref:
            body["ref"] = ref
            message += f" with ref '{ref}'"
            detail = (
                f"Unable to create branch '{branch}' "
                f"in project '{project}' using ref '{ref}'. "
                "Confirm the branch doesn't already exist and try again."
            )

        logger.debug(message)

        url = utils.compose_url(self.api_url, path=["projects", project, "git_branch"])
        response = await self.post(url=url, json=body, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-create-branch",
                title="Couldn't create new Git branch.",
                status=response.status_code,
                detail=detail,
                response=response,
            ) from error

    @backoff_with_exceptions
    async def hard_reset_branch(self, project: str, branch: str, ref: str) -> None:
        """Hard resets a branch to the ref prodvided.

        DANGER: hard reset will be force pushed to the remote. Unsaved changes and
            commits may be permanently lost.

        Args:
            project: Name of the Looker project to use.
            branch: Name of the branch to update.
            ref: The ref to update the branch from.
        """
        logger.debug(
            f"Hard resetting branch '{branch}' on project '{project}' to ref '{ref}'"
        )

        body = {"name": branch, "ref": ref}
        url = utils.compose_url(self.api_url, path=["projects", project, "git_branch"])
        response = await self.put(url=url, json=body, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-update-branch",
                title="Couldn't update Git branch.",
                status=response.status_code,
                detail=(
                    f"Unable to update branch '{branch}' "
                    f"in project '{project}' using ref '{ref}'. "
                    "Please try again."
                ),
                response=response,
            ) from error

    @backoff_with_exceptions
    async def delete_branch(self, project: str, branch: str) -> None:
        """Deletes a branch in the given project.

        Args:
            project: Name of the Looker project to use.
            branch: Name of the branch to delete.
        """
        logger.debug(f"Deleting branch '{branch}' in project '{project}'")

        url = utils.compose_url(
            self.api_url, path=["projects", project, "git_branch", branch]
        )
        response = await self.delete(url=url, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-delete-branch",
                title="Couldn't delete Git branch.",
                status=response.status_code,
                detail=(
                    f"Unable to delete branch '{branch}' "
                    f"in project '{project}'. Please try again."
                ),
                response=response,
            ) from error

    @backoff_with_exceptions
    async def all_lookml_tests(self, project: str) -> List[JsonDict]:
        """Gets all LookML/data tests for a given project.

        Args:
            project: Name of the Looker project to use

        Returns:
            List[JsonDict]: JSON response containing all LookML/data tests

        """
        logger.debug(f"Getting LookML tests for project {project}")
        url = utils.compose_url(
            self.api_url, path=["projects", project, "lookml_tests"]
        )
        response = await self.get(url=url, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-data-tests",
                title="Couldn't retrieve all data tests.",
                status=response.status_code,
                detail=(
                    f"Unable to retrieve all data tests for "
                    f"project '{project}'. Please try again."
                ),
                response=response,
            ) from error

        return response.json()  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def run_lookml_test(
        self, project: str, model: Optional[str] = None, test: Optional[str] = None
    ) -> List[JsonDict]:
        """Runs all LookML/data tests for a given project and model (optional)

        This command only runs tests in production, as the Looker API doesn't currently
        allow us to run data tests on a specific branch.

        Args:
            project: Name of the Looker project to use
            model: Optional name of the LookML model to restrict testing to

        Returns:
            List[JsonDict]: JSON response containing any LookML/data test errors

        """
        if model is None and test is None:
            logger.debug(f"Running all LookML tests for project '{project}'")
        elif model is None and test is not None:
            logger.debug(f"Running LookML test '{test}'")
        elif model is not None and test is None:
            logger.debug(f"Running all LookML tests for model '{model}'")
        elif model is not None and test is not None:
            logger.debug(f"Running LookML test '{test}' in model '{model}'")

        url = utils.compose_url(
            self.api_url, path=["projects", project, "lookml_tests", "run"]
        )

        params = {}
        if model is not None:
            params["model"] = model
        if test is not None:
            params["test"] = test
        response = await self.get(
            url=url,
            params=params,
            timeout=1800,  # 30m timeout for long-running tests
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-run-data-test",
                title="Couldn't run data test.",
                status=response.status_code,
                detail=(
                    f"Unable to run one or more data tests for "
                    f"project '{project}'. Please try again."
                ),
                response=response,
            ) from error

        return response.json()  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def get_lookml_models(
        self, fields: Optional[List[str]] = None
    ) -> List[JsonDict]:
        """Gets all models and explores from the LookmlModel endpoint.

        Returns:
            List[JsonDict]: JSON response containing LookML models and explores.

        """
        logger.debug(f"Getting all models and explores from {self.base_url}")
        if fields is None:
            fields = []

        params: Dict[str, Any] = {}
        if fields:
            params["fields"] = fields

        url = utils.compose_url(self.api_url, path=["lookml_models"], params=params)
        response = await self.get(url=url, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-lookml",
                title="Couldn't retrieve models and explores.",
                status=response.status_code,
                detail="Unable to retrieve LookML details. Please try again.",
                response=response,
            ) from error

        return response.json()  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def get_lookml_dimensions(
        self, model: str, explore: str
    ) -> List[Dict[str, Any]]:
        """Gets all dimensions for an explore from the LookmlModel endpoint."""
        logger.debug(f"Getting all dimensions from explore {model}/{explore}")
        params = {"fields": ["fields"]}
        url = utils.compose_url(
            self.api_url,
            path=["lookml_models", model, "explores", explore],
            params=params,
        )
        response = await self.get(url=url, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            logger.info(
                f"Error code: {response.status_code} for url {url} response: {response.text}"
            )
            raise LookerApiError(
                name="unable-to-get-dimension-lookml",
                title="Couldn't retrieve dimensions.",
                status=response.status_code,
                detail=(
                    "Unable to retrieve dimension LookML details "
                    f"for explore '{model}/{explore}'. Please try again."
                ),
                response=response,
            ) from error

        return response.json()["fields"]["dimensions"]  # type: ignore[no-any-return]

    @cached(cache=Cache.MEMORY, serializer=serializers.PickleSerializer())  # type: ignore
    @backoff_with_exceptions
    async def create_query(
        self,
        model: str,
        explore: str,
        dimensions: List[str],
        fields: Optional[List[str]] = None,
    ) -> JsonDict:
        """Creates a Looker async query for one or more specified dimensions.

        The query created is a SELECT query, selecting all dimensions specified for a
        certain model and explore. Looker builds the query using the `sql` field in the
        LookML for each dimension.

        If a Timeout exception is received, attempts to retry.

        """
        # Using old-style string formatting so that strings are formatted lazily
        logger.debug(
            "Creating async query for %s/%s/%s",
            model,
            explore,
            "*" if len(dimensions) != 1 else dimensions[0],
        )
        body = {
            "model": model,
            "view": explore,
            "fields": dimensions,
            "limit": 0,
            "filter_expression": "1=2",
        }

        params: Dict[str, List[str]] = {}
        if fields is None:
            params["fields"] = []
        else:
            params["fields"] = fields

        url = utils.compose_url(self.api_url, path=["queries"], params=params)
        response = await self.post(url=url, json=body, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-create-query",
                title="Couldn't create query.",
                status=response.status_code,
                detail=(
                    f"Failed to create query for {model}/{explore}/"
                    f'{"*" if len(dimensions) > 1 else dimensions[0]}. '
                    "Please try again."
                ),
                response=response,
            ) from error

        result = response.json()
        query_id: str = result["id"]
        logger.debug(
            "Query for %s/%s/%s created as query %s",
            model,
            explore,
            "*" if len(dimensions) != 1 else dimensions[0],
            query_id,
        )
        return result  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def create_query_task(self, query_id: str) -> str:
        """Runs a previously created query asynchronously and returns the query task ID.

        If a ClientError or TimeoutError is received, attempts to retry.

        Args:
            session: Existing asychronous HTTP session.
            query_id: ID of a previously created query to run.

        Returns:
            str: ID for the query task, used to check on the status of the query, which
                is being run asynchronously.

        """
        # Using old-style string formatting so that strings are formatted lazily
        logger.debug("Starting query %s", query_id)
        body = {"query_id": query_id, "result_format": "json_detail"}
        params = {"fields": ["id"]}
        url = utils.compose_url(self.api_url, path=["query_tasks"], params=params)

        response = await self.post(
            url=url, json=body, params={"cache": "false"}, timeout=TIMEOUT_SEC
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-launch-query",
                title="Couldn't launch query.",
                status=response.status_code,
                detail=(
                    "Failed to create query task for "
                    f"query '{query_id}'. Please try again."
                ),
                response=response,
            ) from error

        result = response.json()
        query_task_id = result["id"]
        logger.debug("Query %s is running under query task %s", query_id, query_task_id)
        return query_task_id  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def get_query_task_multi_results(
        self, query_task_ids: Tuple[str, ...]
    ) -> JsonDict:
        """Returns query task results.

        If a ClientError or TimeoutError is received, attempts to retry.

        Args:
            query_task_ids: IDs for the query tasks running asynchronously.

        Returns:
            List[JsonDict]: JSON response from the query task.

        """
        # Using old-style string formatting so that strings are formatted lazily
        logger.debug(
            "Attempting to get results for %d query tasks", len(query_task_ids)
        )
        url = utils.compose_url(self.api_url, path=["query_tasks", "multi_results"])
        response = await self.get(
            url=url,
            params={"query_task_ids": ",".join(query_task_ids)},
            timeout=TIMEOUT_SEC,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-query-results",
                title="Couldn't get results for the specified query tasks.",
                status=response.status_code,
                detail=(
                    "Failed to get the results for "
                    f"{len(query_task_ids)} query tasks. "
                    "Please try again."
                ),
                response=response,
            ) from error

        result = response.json()
        return result  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def cancel_query_task(self, query_task_id: str) -> None:
        """Cancels a query task.

        Args:
            query_task_id: ID for the query task to cancel.

        """
        logger.debug(f"Cancelling query task: {query_task_id}")
        url = utils.compose_url(self.api_url, path=["running_queries", query_task_id])
        await self.delete(url=url, timeout=TIMEOUT_SEC)

        # No raise_for_status() here because Looker API seems to give a 404
        # if you try to cancel a finished query which can happen as part of cleanup

    async def content_validation(self) -> JsonDict:
        logger.debug("Validating all content in Looker")
        url = utils.compose_url(self.api_url, path=["content_validation"])
        response = await self.get(
            url=url, timeout=3600
        )  # 1 hour timeout for content validation

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-validate-content",
                title="Couldn't validate Looks and Dashboards.",
                status=response.status_code,
                detail=("Failed to run the content validator. Please try again."),
                response=response,
            ) from error

        result = response.json()
        return result  # type: ignore[no-any-return]

    @backoff.on_exception(
        backoff.expo,
        # Omit retries on timeouts because timeout is already very long
        STATUS_EXCEPTIONS
        + (
            NetworkError,
            RemoteProtocolError,
        ),
        giveup=giveup_unless_bad_gateway,
        max_tries=DEFAULT_RETRIES,
    )
    async def lookml_validation(
        self, project: str, timeout: int = LOOKML_VALIDATION_TIMEOUT
    ) -> JsonDict:
        logger.debug(
            f"Validating LookML for project '{project}' with timeout {timeout} seconds."
        )
        url = utils.compose_url(self.api_url, path=["projects", project, "validate"])
        response = await self.post(url=url, timeout=timeout)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-validate-lookml",
                title=f"Couldn't validate LookML in project {project}.",
                status=response.status_code,
                detail=("Failed to run the LookML validator. Please try again."),
                response=response,
            ) from error

        result = response.json()
        return result  # type: ignore[no-any-return]

    @backoff.on_exception(
        backoff.expo,
        # Omit retries on timeouts because timeout is already very long
        STATUS_EXCEPTIONS
        + (
            NetworkError,
            RemoteProtocolError,
        ),
        giveup=giveup_unless_bad_gateway,
        max_tries=DEFAULT_RETRIES,
    )
    async def cached_lookml_validation(self, project: str) -> Optional[JsonDict]:
        logger.debug(f"Getting cached LookML validation results for '{project}'")
        url = utils.compose_url(self.api_url, path=["projects", project, "validate"])
        response = await self.get(url=url, timeout=1800)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-cached-lookml-validation",
                title=f"Couldn't get cached LookML validation results in project '{project}'.",
                status=response.status_code,
                detail=(
                    "Failed to get cached LookML valiation results. Please try again."
                ),
                response=response,
            ) from error

        # If no cached validation results are available, Looker returns a 204 No Content.
        # The response has no payload. We should return None in this case and handle accordingly.
        if response.status_code == 204:
            return None

        result = response.json()
        return result  # type: ignore[no-any-return]

    @cached(cache=Cache.MEMORY, serializer=serializers.PickleSerializer())  # type: ignore
    @backoff_with_exceptions
    async def all_folders(self) -> List[JsonDict]:
        logger.debug("Getting information about all folders")
        url = utils.compose_url(self.api_url, path=["folders"])
        response = await self.get(url=url, timeout=TIMEOUT_SEC)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LookerApiError(
                name="unable-to-get-folders",
                title="Couldn't obtain project folders.",
                status=response.status_code,
                detail=("Failed to get all folders."),
                response=response,
            ) from error

        result = response.json()
        return result  # type: ignore[no-any-return]

    @backoff_with_exceptions
    async def run_query(
        self, query_id: str, explore: str, model: str, dimension: Optional[str] = None
    ) -> str:
        """Returns the compiled SQL for a given query ID.

        The corresponding Looker API endpoint allows us to run queries with a variety
        of result formats, however we only use the `sql` result format, which doesn't
        run the query but does return its compiled SQL.

        If a Timeout exception is received, attempts to retry.

        """
        # Using old-style string formatting so that strings are formatted lazily
        logger.debug("Retrieving the SQL for query ID %s", query_id)

        url = utils.compose_url(self.api_url, path=["queries", query_id, "run", "sql"])
        response = await self.get(url=url, timeout=TIMEOUT_SEC)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return (
                    "-- SQL could not be generated because of errors with this query."
                )
            elif e.response.status_code == 400 and (
                "Must query at least one dimension or measure" in response.text
                or "View Not Found" in response.text
            ):
                return (
                    "-- SQL could not be generated because of errors with this query."
                )
            else:
                detail = (
                    f"Failed to retrieve compiled SQL for "
                    f"{'dimension' if dimension else 'explore'} "
                    f"'{model}/{explore}{'/' + dimension if dimension else ''}' "
                    f"with query '{query_id}'. "
                    "Please try again."
                )
                try:
                    response_json = response.json()
                    if "message" in response_json:
                        message = response_json["message"]
                        detail += f' Received the following from Looker: "{message}"'
                except json.JSONDecodeError:
                    pass
                raise LookerApiError(
                    name="unable-to-retrieve-compiled-sql",
                    title="Couldn't retrieve compiled SQL.",
                    status=response.status_code,
                    detail=detail,
                    response=response,
                ) from e

        result = response.text
        logger.debug("Retrieved compiled SQL for query %s", query_id)

        return result
