import asyncio
import itertools
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from spectacles.client import LOOKML_VALIDATION_TIMEOUT, LookerClient
from spectacles.exceptions import LookerApiError, SpectaclesException, SqlError
from spectacles.logger import GLOBAL_LOGGER as logger
from spectacles.lookml import CompiledSql, Explore, build_project, Project
from spectacles.models import JsonDict, SkipReason
from spectacles.printer import print_header
from spectacles.utils import time_hash
from spectacles.validators import (
    ContentValidator,
    DataTestValidator,
    LookMLValidator,
    SqlValidator,
)
from spectacles.validators.data_test import DATA_TEST_CONCURRENCY
from spectacles.validators.sql import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_QUERY_CONCURRENCY,
    DEFAULT_RUNTIME_THRESHOLD,
)


@dataclass
class ProjectState:
    project: str
    workspace: str
    branch: str
    commit: str


def is_commit(string: str) -> bool:
    """Tests if a string is a SHA1 hash."""
    return bool(re.match("[0-9a-f]{5,40}", string))


class LookerBranchManager:
    def __init__(
        self,
        client: LookerClient,
        project: str,
        remote_reset: bool = False,
        use_personal_branch: bool = False,
        pin_imports: Optional[Dict[str, str]] = None,
        skip_imports: Optional[List[str]] = None,
    ):
        """Context manager for Git branch checkout, creation, and deletion."""
        logger.debug(f"Setting up branch manager in project '{project}'")
        self.client = client
        self.project = project
        self.remote_reset = remote_reset
        self.pin_imports = pin_imports or {}
        self.history: List[ProjectState] = []

        self.commit: Optional[str] = None
        self.branch: Optional[str] = None
        self.is_temp_branch: bool = False
        self.use_personal_branch: bool = use_personal_branch
        self.personal_branch: Optional[str] = None
        self.import_managers: List[LookerBranchManager] = []
        self.skip_imports: List[str] = [] if skip_imports is None else skip_imports

    def __call__(
        self, ref: Optional[str] = None, ephemeral: Optional[bool] = None
    ) -> "LookerBranchManager":
        logger.debug("")
        logger.debug(
            f"Setting Git state for project '{self.project}' "
            f"@ {ref or 'production'}:\n"
        )
        self.branch = None
        self.commit = None

        if ref is None:
            pass
        elif is_commit(ref):
            self.commit = ref
        else:
            self.branch = ref

        if ephemeral is None:
            if self.commit or self.pin_imports:
                self.ephemeral = True
            else:
                self.ephemeral = False
        else:
            if self.commit and ephemeral is False:
                raise ValueError(
                    "ephemeral = False is invalid for a commit reference because "
                    "it's impossible to checkout a commit directly in Looker. "
                    "You must use a temp branch."
                )

            self.ephemeral = ephemeral

        self.is_temp_branch = False
        self.import_managers = []
        return self

    async def __aenter__(self) -> None:
        logger.indent(1)

        state: ProjectState = await self.get_project_state()
        self.workspace: str = state.workspace
        self.history = [state]
        # A branch was passed, so we check it out in dev mode.
        if self.branch:
            await self.update_workspace("dev")
            if self.ephemeral:
                new_branch = await self.checkout_ephemeral_branch(
                    "origin/" + self.branch
                )
                if not self.use_personal_branch:
                    self.branch = new_branch
            else:
                await self.client.checkout_branch(self.project, self.branch)
                if self.remote_reset:
                    await self.client.reset_to_remote(self.project)
        # A commit was passed, so we non-destructively create a temporary branch we can
        # hard reset to the commit.
        elif self.commit:
            self.branch = await self.checkout_ephemeral_branch(self.commit)
        # Neither branch nor commit were passed, so go to production.
        else:
            if self.init_state.workspace == "production":
                prod_state = self.init_state
            else:
                await self.update_workspace("production")
                prod_state = await self.get_project_state()
            self.branch = prod_state.branch
            self.commit = prod_state.commit
            if self.ephemeral:
                self.branch = await self.checkout_ephemeral_branch(prod_state.commit)

        logger.debug(
            f"Set project '{self.project}' to branch '{self.branch}' @ "
            f"{(self.commit or 'HEAD')[:6]} in {self.workspace} workspace "
            f"[ephemeral = {self.ephemeral}]"
        )

        self.imports: List[str] = await self.get_project_imports()
        logger.debug(
            f"Project '{self.project}' imports the following projects: {self.imports}"
        )

        # Create temporary branches off production for manifest dependencies
        if not self.imports:
            logger.debug(f"Project '{self.project}' doesn't import any other projects")
        elif self.workspace == "production":
            logger.debug(
                "In production, no need for temporary branches in imported projects"
            )
        else:
            logger.debug("Creating temporary branches in imported projects")
            for project in self.imports:
                if project == self.project:
                    raise SpectaclesException(
                        name="/errors/circular-project-import",
                        title="Circular project import",
                        detail=f"Project '{self.project}' imports itself",
                    )
                elif project not in self.skip_imports:
                    import_ref = self.pin_imports.get(project, None)
                    manager = LookerBranchManager(
                        self.client,
                        project,
                        pin_imports=self.pin_imports,
                        skip_imports=self.skip_imports,
                        use_personal_branch=self.use_personal_branch,
                    )
                    await manager(ref=import_ref, ephemeral=True).__aenter__()
                    self.import_managers.append(manager)
                    self.skip_imports.append(project)
                else:
                    logger.debug(
                        f"Skipping project '{project}', which is already imported"
                    )

        logger.indent(-1)
        logger.debug("")

    async def __aexit__(self, *args: Any) -> None:
        logger.debug("")
        logger.debug(f"Cleaning up Git state in '{self.project}'")
        logger.indent(1)
        message = (
            f"Restoring project '{self.project}' to branch '{self.init_state.branch}'"
        )
        if self.is_temp_branch:
            message += f" and deleting temporary branch '{self.branch}'"
        logger.debug(message)

        if self.is_temp_branch:
            if self.branch is None:
                raise TypeError(
                    "Unable to clean up temporary branch, LookerBranchManager.branch "
                    "is None"
                )
            dev_state = self.history.pop()
            await self.client.checkout_branch(self.project, dev_state.branch)
            await self.client.delete_branch(self.project, self.branch)

        for manager in self.import_managers:
            await manager.__aexit__()

        self.skip_imports = []

        if self.init_state.workspace == "production":
            await self.update_workspace("production")
        else:
            await self.update_workspace("dev")
            await self.client.checkout_branch(self.project, self.init_state.branch)

        logger.indent(-1)
        logger.debug("")

    @property
    def init_state(self) -> ProjectState:
        try:
            state = self.history[0]
        except IndexError as error:
            raise IndexError(
                "No history exists, you must enter the context manager "
                "to generate initial state."
            ) from error
        return state

    @property
    def ref(self) -> Optional[str]:
        if self.commit:
            return self.commit[:6]
        else:
            return self.branch

    async def update_workspace(self, workspace: str) -> None:
        if workspace not in ("dev", "production"):
            raise ValueError("Workspace can only be set to 'dev' or 'production'")
        if self.workspace != workspace:
            await self.client.update_workspace(workspace)
            self.workspace = workspace

    async def get_project_state(self) -> ProjectState:
        workspace = await self.client.get_workspace()
        branch_info = await self.client.get_active_branch(self.project)
        return ProjectState(
            self.project, workspace, branch_info["name"], branch_info["ref"]
        )

    async def get_project_imports(self) -> List[str]:
        try:
            manifest = await self.client.get_manifest(self.project)
        except LookerApiError:
            return []
        else:
            return [p["name"] for p in manifest["imports"] if not p["is_remote"]]

    async def checkout_personal_branch(self, ref: str) -> str:
        """Updates the user's personal branch to the git ref."""
        await self.update_workspace("dev")
        if not self.personal_branch:
            self.personal_branch = await self.get_personal_branch()
        await self.client.checkout_branch(self.project, self.personal_branch)
        await self.client.reset_to_remote(self.project)
        await self.client.hard_reset_branch(self.project, self.personal_branch, ref)
        return self.personal_branch

    async def get_personal_branch(self) -> str:
        """Finds the name of the user's personal branch."""
        branches = await self.client.get_all_branches(self.project)
        for branch in branches:
            if branch["personal"] and not branch["readonly"]:
                return str(branch["name"])
        raise ValueError(
            f"Personal branch not found for client ID {self.client.client_id} "
            f"in project '{self.project}'"
        )

    async def checkout_temp_branch(self, ref: str) -> str:
        """Creates a temporary branch off a commit or off production."""
        # Save the dev mode state so we have somewhere to delete the temp branch
        # from later. We can't delete branches from prod workspace.
        await self.update_workspace("dev")
        self.history.append(await self.get_project_state())
        name = "tmp_spectacles_" + time_hash()
        logger.debug(
            f"Branching '{name}' off '{ref}'. "
            f"Afterwards, restoring to branch '{self.init_state.branch}' in "
            f"project '{self.project}'"
        )
        await self.client.create_branch(self.project, name)
        await self.client.hard_reset_branch(self.project, name, ref)
        self.is_temp_branch = True
        return name

    async def checkout_ephemeral_branch(self, ref: str) -> str:
        """Either check out temp or personal branch and hard-reset."""
        if self.use_personal_branch:
            branch = await self.checkout_personal_branch(ref)
        else:
            branch = await self.checkout_temp_branch(ref)
        return branch


class Runner:
    """Runs validations and returns JSON-style dictionaries with validation results.

    Args:
        base_url: Base URL for the Looker instance, e.g. https://mycompany.looker.com.
        project: Name of the Looker project to use.
        branch: Name of the Git branch to check out.
        client_id: Looker API client ID.
        client_secret: Looker API client secret.
        port: Desired API port to use for requests.
        api_version: Desired API version to use for requests.

    Attributes:
        client: Looker API client used for making requests.

    """

    def __init__(
        self,
        client: LookerClient,
        project: str,
        remote_reset: bool = False,
        pin_imports: Optional[Dict[str, str]] = None,
        use_personal_branch: bool = False,
    ):
        self.project = project
        self.client = client
        self.branch_manager = LookerBranchManager(
            client,
            project,
            remote_reset,
            pin_imports=pin_imports or {},
            use_personal_branch=use_personal_branch,
        )

    async def validate_sql(
        self,
        ref: Optional[str] = None,
        filters: Optional[List[str]] = None,
        fail_fast: bool = True,
        incremental: bool = False,
        target: Optional[str] = None,
        concurrency: int = DEFAULT_QUERY_CONCURRENCY,
        profile: bool = False,
        runtime_threshold: int = DEFAULT_RUNTIME_THRESHOLD,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        ignore_hidden_fields: bool = False,
    ) -> JsonDict:
        if filters is None:
            filters = ["*/*"]
        else:
            # Only build the full project from the API if we're using a wildcard filter and not in incremental mode
            get_full_project = any("*" in f for f in filters) or incremental

        validator = SqlValidator(self.client, concurrency, runtime_threshold)
        ephemeral = True if incremental else None
        # Create explore-level tests for the desired ref
        async with self.branch_manager(ref=ref, ephemeral=ephemeral):
            base_ref = self.branch_manager.ref  # Resolve the full ref after checkout
            logger.debug("Compiling SQL for Explores at the base ref")
            project = await build_project(
                self.client,
                name=self.project,
                filters=filters,
                include_dimensions=True,
                ignore_hidden_fields=ignore_hidden_fields,
                get_full_project=get_full_project,
            )
            base_explores: Set[CompiledSql] = set()
            if incremental:
                compiled_explores = await asyncio.gather(
                    *(
                        validator.compile_explore(explore)
                        for explore in project.iter_explores()
                    )
                )
                base_explores = set(compiled_explores)

        if incremental:
            # Create explore tests for the target
            async with self.branch_manager(ref=target, ephemeral=True):
                target_ref = self.branch_manager.ref
                if target_ref == base_ref:
                    raise SpectaclesException(
                        name="incremental-same-ref",
                        title="Incremental comparison to the same Git state.",
                        detail=(
                            f"The base ref ({ref or 'production'}) and "
                            f"target ref ({target or 'production'}) point to the "
                            f"same commit ({base_ref}), "
                            "so incremental comparison isn't possible."
                        ),
                    )
                logger.debug("Compiling SQL for Explores at the target ref")
                target_project = await build_project(
                    self.client,
                    name=self.project,
                    filters=filters,
                    include_dimensions=True,
                    ignore_hidden_fields=ignore_hidden_fields,
                )
                target_explores: Set[CompiledSql] = set()

                compiled_explores = await asyncio.gather(
                    *(
                        validator.compile_explore(explore)
                        for explore in target_project.iter_explores()
                    )
                )
                target_explores = set(compiled_explores)

            # Determine which explore tests are identical between target and base
            # Iterate instead of set operations so we have control of which test, and
            # corresponding which `lookml_ref` is used
            explores: Tuple[Explore, ...] = tuple()
            for compiled in base_explores:
                explore = project.get_explore(
                    compiled.model_name, compiled.explore_name
                )
                if explore is None:
                    raise TypeError(
                        "Couldn't find the explore "
                        f"{compiled.model_name}.{compiled.explore_name}"
                    )
                if compiled in target_explores:
                    # Mark explores with the same compiled SQL (test) as skipped
                    explore.skipped = SkipReason.UNMODIFIED
                else:
                    # Test explores with unique SQL for base ref
                    explores += (explore,)

            logger.debug(f"Found {len(explores)} explores with unique SQL")
        else:
            explores = tuple(project.iter_explores())

        n_tested_explores = len([e for e in explores if not e.skipped])
        n_total_explores = len(base_explores or explores)
        print_header(
            f"Testing {n_tested_explores}/{n_total_explores} explores "
            + ("[fail fast] " if fail_fast else "")
            + f"[concurrency = {concurrency}]"
        )

        async with self.branch_manager(ref=ref):
            await validator.search(explores, fail_fast, chunk_size, profile=profile)

        # Create dimension tests for the desired ref when explores errored
        if not fail_fast and incremental:
            errored_dimensions = project.iter_dimensions(errored=True)
            # For errored dimensions, create dimension tests for the target ref
            async with self.branch_manager(ref=target, ephemeral=True):
                target_ref = self.branch_manager.ref
                logger.debug("Compiling SQL for dimensions at the target ref")
                compiled_dimensions = await asyncio.gather(
                    *(
                        validator.compile_dimension(dimension)
                        for dimension in project.iter_dimensions(errored=True)
                    )
                )
                target_dimensions = set(compiled_dimensions)

            # Keep only the errors that don't exist on the target branch
            logger.debug(
                "Removing errors that would exist in project "
                f"@ {target or 'production'}"
            )

            # Namespace SQL with the dimension name, just in case
            target_sql = tuple(
                (compiled.dimension_name, compiled.sql)
                for compiled in target_dimensions
            )
            for dimension in errored_dimensions:
                for error in dimension.errors:
                    if (
                        isinstance(error, SqlError)
                        and (dimension.name, error.metadata["sql"]) in target_sql
                    ):
                        error.ignore = True

        results = project.get_results(validator="sql", fail_fast=fail_fast)
        return results

    async def validate_data_tests(
        self,
        ref: Optional[str] = None,
        filters: Optional[List[str]] = None,
        concurrency: int = DATA_TEST_CONCURRENCY,
    ) -> JsonDict:
        if filters is None:
            filters = ["*/*"]
        async with self.branch_manager(ref):
            validator = DataTestValidator(self.client)
            logger.info(
                "Building LookML project hierarchy for project "
                f"'{self.project}' @ {self.branch_manager.ref}"
            )
            project = await build_project(
                self.client, name=self.project, filters=filters
            )
            explore_count = project.count_explores()
            print_header(
                f"Running data tests based on {explore_count} "
                f"{'explore' if explore_count == 1 else 'explores'}"
            )
            tests = await validator.get_tests(project)
            await validator.validate(tests, concurrency)

        results = project.get_results(validator="data_test")
        return results

    async def validate_lookml(
        self,
        ref: Optional[str],
        severity: str,
        timeout: int = LOOKML_VALIDATION_TIMEOUT,
    ) -> JsonDict:
        async with self.branch_manager(ref=ref):
            validator = LookMLValidator(self.client)
            print_header(f"Validating LookML in project {self.project} [{severity}]")
            results = await validator.validate(self.project, severity, timeout)
        return results

    async def validate_content(
        self,
        ref: Optional[str] = None,
        filters: Optional[List[str]] = None,
        incremental: bool = False,
        target: Optional[str] = None,
        exclude_personal: bool = False,
        folders: Optional[List[str]] = None,
    ) -> JsonDict:
        logger.debug(
            f"Validating content. ref={ref}, filters={filters}, incremental={incremental}",
        )
        if filters is not None or filters == ["*/*"]:
            # Only build the full project from the API if we're using a wildcard filter and not in incremental mode
            get_full_project = any("*" in f for f in filters if f != "*/*") or (
                not incremental
            )
            logger.debug(f"get_full_project = {get_full_project}")

        if folders is None:
            folders = []

        async with self.branch_manager(ref=ref):
            validator = ContentValidator(
                self.client,
                exclude_personal,
                folders,
            )
            logger.info(
                "Building LookML project hierarchy for project "
                f"'{self.project}' @ {self.branch_manager.ref}"
            )
            project = await build_project(
                self.client,
                name=self.project,
                filters=filters,
                include_all_explores=True,
            )
            explore_count = project.count_explores()
            print_header(
                f"Validating content based on {explore_count} "
                f"{'explore' if explore_count == 1 else 'explores'}"
                + (" [incremental mode] " if incremental else "")
            )
            await validator.validate(project)
            results = project.get_results(validator="content", filters=filters)

        if incremental and (self.branch_manager.branch or self.branch_manager.commit):
            logger.debug("Starting another content validation against the target ref")
            async with self.branch_manager(ref=target):
                logger.debug(
                    "Building LookML project hierarchy for project "
                    f"'{self.project}' @ {self.branch_manager.ref}"
                )
                target_project = await build_project(
                    self.client,
                    name=self.project,
                    filters=filters,
                    include_all_explores=True,
                )
                await validator.validate(target_project)
                target_results = target_project.get_results(
                    validator="content", filters=filters
                )

            return self._incremental_results(base=results, target=target_results)
        else:
            return results

    @staticmethod
    def _incremental_results(base: JsonDict, target: JsonDict) -> JsonDict:
        """Returns a new result with only the additional errors in `additional`."""
        diff: JsonDict = {
            "validator": "content",
            # Start with models and explores we know passed for the base ref
            "tested": [test for test in base["tested"] if test["status"] != "failed"],
            "errors": [],
        }

        # Build a list of disputed tests where dupes by model and explore are allowed
        tests = []
        for error in base["errors"]:
            if error in target["errors"]:
                status = "passed"
            else:
                status = "failed"
                diff["errors"].append(error)

            test = dict(model=error["model"], explore=error["explore"], status=status)
            tests.append(test)

        def key_by(x: dict[str, Any]) -> Tuple[str, str]:
            return (x["model"], x["explore"])

        if tests:
            # Dedupe the list of tests, grouping by model and explore and taking the min
            # To do this, we group by model and explore and sort by `passed`
            tests = sorted(
                tests, key=lambda x: (x["model"], x["explore"], x["status"] != "failed")
            )
            for _, group in itertools.groupby(tests, key=key_by):
                items = list(group)
                diff["tested"].append(items[0])

        # Re-sort the final list
        diff["tested"] = sorted(diff["tested"], key=key_by)

        # Recompute the overall state of the test suite
        passed = min(
            (test["status"] != "failed" for test in diff["tested"]), default=True
        )
        diff["status"] = "passed" if passed else "failed"

        return diff
