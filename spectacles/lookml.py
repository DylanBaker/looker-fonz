import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

from spectacles.client import LookerClient
from spectacles.exceptions import LookMlNotFound, ValidationError
from spectacles.logger import GLOBAL_LOGGER as logger
from spectacles.models import JsonDict, SkipReason
from spectacles.project_select import is_selected


class LookMlObject:
    name: str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def queried(self) -> bool:
        raise NotImplementedError

    @queried.setter
    def queried(self, value: bool) -> None:
        raise NotImplementedError


class Dimension(LookMlObject):
    def __init__(
        self,
        name: str,
        model_name: str,
        explore_name: str,
        type: str,
        tags: List[str],
        sql: str,
        is_hidden: bool,
        url: Optional[str] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.explore_name = explore_name
        self.type = type
        self.tags = tags
        self.sql = sql
        self.url = url
        self.is_hidden = is_hidden
        self._queried: bool = False
        self.errors: List[ValidationError] = []

        if (
            re.search(r"spectacles\s*:\s*ignore", sql, re.IGNORECASE)
            or "spectacles: ignore" in tags
        ):
            self.ignore = True
        else:
            self.ignore = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"type={self.type}, "
            f"errored={self.errored})"
        )

    @property
    def queried(self) -> bool:
        return self._queried

    @queried.setter
    def queried(self, value: bool) -> None:
        self._queried = value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented

        return (
            self.name == other.name
            and self.model_name == other.model_name
            and self.explore_name == other.explore_name
            and self.type == other.type
            and self.url == other.url
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented

        return (self.model_name, self.explore_name, self.name) < (
            other.model_name,
            other.explore_name,
            other.name,
        )

    @property
    def errored(self) -> Optional[bool]:
        return bool(self.errors) if self.queried else None

    @errored.setter
    def errored(self, value: bool) -> None:
        raise AttributeError(
            "Cannot assign to 'errored' property of a Dimension instance. "
            "For a dimension to be considered errored, it must have a ValidationError "
            "in its 'errors' attribute."
        )

    @classmethod
    def from_json(
        cls, json_dict: Dict[str, Any], model_name: str, explore_name: str
    ) -> "Dimension":
        name = json_dict["name"]
        type = json_dict["type"]
        tags = json_dict["tags"]
        sql = json_dict["sql"]
        url = json_dict["lookml_link"]
        is_hidden = json_dict["hidden"]
        return cls(name, model_name, explore_name, type, tags, sql, is_hidden, url)


class Explore(LookMlObject):
    def __init__(
        self, name: str, model_name: str, dimensions: Optional[List[Dimension]] = None
    ):
        self.name = name
        self.model_name = model_name
        self.dimensions = [] if dimensions is None else dimensions
        self.errors: List[ValidationError] = []
        self.successes: List[JsonDict] = []
        self.skipped: Optional[SkipReason] = None
        self._queried: bool = False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Explore):
            return NotImplemented

        return (
            self.name == other.name
            and self.model_name == other.model_name
            and self.dimensions == other.dimensions
        )

    @property
    def queried(self) -> bool:
        if self.dimensions:
            return any(dimension.queried for dimension in self.dimensions)
        else:
            return self._queried

    @queried.setter
    def queried(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Value for queried must be boolean.")
        if self.dimensions:
            for dimension in self.dimensions:
                dimension.queried = value
        else:
            self._queried = value

    @property
    def errored(self) -> Optional[bool]:
        if self.queried:
            return bool(self.errors) or any(
                dimension.errored for dimension in self.dimensions
            )
        else:
            return None

    @errored.setter
    def errored(self, value: bool) -> None:
        raise AttributeError(
            "Cannot assign to 'errored' property of an Explore instance. "
            "For an explore to be considered errored, it must have a ValidationError "
            "in its 'errors' attribute or contain dimensions in an errored state."
        )

    def get_errored_dimensions(self) -> Generator[Dimension, None, None]:
        for dimension in self.dimensions:
            if dimension.errored:
                yield dimension

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any], model_name: str) -> "Explore":
        name = json_dict["name"]
        return cls(name, model_name)

    def add_dimension(self, dimension: Dimension) -> None:
        self.dimensions.append(dimension)

    @property
    def number_of_errors(self) -> int:
        if self.errored:
            if self.errors:
                errors = len(self.errors)
            else:
                errors = sum(
                    len(dimension.errors)
                    for dimension in self.dimensions
                    if dimension.errored
                )
            return errors
        else:
            return 0


@dataclass(eq=True, frozen=True)
class CompiledSql:
    model_name: str
    explore_name: str
    sql: str
    dimension_name: Optional[str] = None

    @classmethod
    def from_explore(cls, explore: Explore, sql: str) -> "CompiledSql":
        return CompiledSql(
            model_name=explore.model_name, explore_name=explore.name, sql=sql
        )

    @classmethod
    def from_dimension(cls, dimension: Dimension, sql: str) -> "CompiledSql":
        return CompiledSql(
            model_name=dimension.model_name,
            explore_name=dimension.explore_name,
            dimension_name=dimension.name,
            sql=sql,
        )


class Model(LookMlObject):
    def __init__(self, name: str, project_name: str, explores: List[Explore]):
        self.name = name
        self.project_name = project_name
        self.explores = explores
        self.errors: List[ValidationError] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, explores={self.explores})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Model):
            return NotImplemented

        return (
            self.name == other.name
            and self.project_name == other.project_name
            and self.explores == other.explores
        )

    @property
    def errored(self) -> Optional[bool]:
        if self.queried:
            return bool(self.errors) or any(
                explore.errored for explore in self.explores
            )
        else:
            return None

    @errored.setter
    def errored(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Value for errored must be boolean.")
        if not self.explores:
            raise AttributeError(
                "Cannot assign to 'errored' property because this model does not have any explores."
            )
        for explore in self.explores:
            explore.errored = value

    @property
    def queried(self) -> bool:
        return any(explore.queried for explore in self.explores)

    @queried.setter
    def queried(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Value for queried must be boolean.")
        for explore in self.explores:
            explore.queried = value

    def get_explore(self, name: str) -> Optional[Explore]:
        return next(
            (explore for explore in self.explores if explore.name == name), None
        )

    def get_errored_explores(self) -> Generator[Explore, None, None]:
        for explore in self.explores:
            if explore.errored:
                yield explore

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "Model":
        name = json_dict["name"]
        project = json_dict["project_name"]
        explores = [
            Explore.from_json(d, model_name=name) for d in json_dict["explores"]
        ]
        return cls(name, project, explores)

    @property
    def number_of_errors(self) -> int:
        return len(self.errors) + sum(
            [explore.number_of_errors for explore in self.explores if explore.errored]
        )


class Project(LookMlObject):
    def __init__(self, name: str, models: Sequence[Model]) -> None:
        self.name = name
        self.models = models
        self._is_complete = False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Project):
            return NotImplemented

        return self.name == other.name and self.models == other.models

    def count_explores(self) -> int:
        """Returns the number of explores in the project, excluding skipped explores."""
        return len([explore for explore in self.iter_explores() if not explore.skipped])

    def iter_models(self, errored: bool = False) -> Iterable[Model]:
        for model in self.models:
            if errored:
                if model.errored:
                    yield model
            else:
                yield model

    def iter_explores(self, errored: bool = False) -> Iterable[Explore]:
        for model in self.iter_models():
            for explore in model.explores:
                if errored:
                    if explore.errored:
                        yield explore
                else:
                    yield explore

    def iter_dimensions(self, errored: bool = False) -> Iterable[Dimension]:
        for explore in self.iter_explores():
            for dimension in explore.dimensions:
                if errored:
                    if dimension.errored:
                        yield dimension
                else:
                    yield dimension

    @property
    def is_complete_project(self) -> bool:
        return self._is_complete

    @is_complete_project.setter
    def is_complete_project(self, value: bool) -> None:
        self._is_complete = value

    @property
    def errored(self) -> Optional[bool]:
        if self.queried:
            return any(model.errored for model in self.models)
        else:
            return None

    @errored.setter
    def errored(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Value for errored must be boolean.")
        if not self.models:
            raise AttributeError(
                "Cannot assign to 'errored' property because this project does not have any models."
            )
        for model in self.models:
            model.errored = value

    @property
    def queried(self) -> bool:
        return any(model.queried for model in self.models)

    @queried.setter
    def queried(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Value for queried must be boolean.")
        for model in self.models:
            model.queried = value

    def get_model(self, name: str) -> Optional[Model]:
        return next((model for model in self.models if model.name == name), None)

    def get_explore(self, model: str, name: str) -> Optional[Explore]:
        model_object = self.get_model(model)
        if not model_object:
            return None
        else:
            return model_object.get_explore(name)

    def get_results(
        self,
        validator: str,
        fail_fast: Optional[bool] = None,
        filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        successes: List[Dict[str, Any]] = []
        tested = []

        for model in self.models:
            # Add model level content validation errors.
            # We create an explore "tested" record for those that
            # aren't in the LookML tree.
            distinct_explores = set()

            for error in model.errors:
                if filters is not None and not is_selected(
                    model.name, error.explore, filters
                ):
                    continue
                distinct_explores.add(error.explore)
                errors.append(error.to_dict())

            model_tested = [
                {"model": model.name, "explore": e, "status": "failed"}
                for e in distinct_explores
            ]
            tested.extend(model_tested)

            for explore in model.explores:
                if filters is not None and not is_selected(
                    model.name,
                    explore.name,
                    filters,  # pyright: ignore[reportGeneralTypeIssues]
                ):
                    continue

                test: Dict[str, Any] = {
                    "model": model.name,
                    "explore": explore.name,
                    "status": "passed",  # To be overwritten if needed
                }

                if explore.skipped:
                    test["status"] = "skipped"
                    test["skip_reason"] = explore.skipped.value
                elif explore.errored and validator != "sql":
                    test["status"] = "failed"
                    errors.extend([e.to_dict() for e in explore.errors])
                elif explore.errored and fail_fast is True:
                    test["status"] = "failed"
                    errors.append(explore.errors[0].to_dict())
                elif explore.errored:
                    dimension_errors = [e for d in explore.dimensions for e in d.errors]
                    # If an explore has explore-level errors but not dimension-level
                    # errors, return those instead. Skip anything marked as ignored.
                    relevant_errors = [
                        e.to_dict()
                        for e in dimension_errors + explore.errors
                        if not e.ignore
                    ]
                    if relevant_errors:
                        test["status"] = "failed"
                        errors.extend(relevant_errors)

                if explore.successes:
                    successes.extend([success for success in explore.successes])

                tested.append(test)

        passed = min((test["status"] != "failed" for test in tested), default=True)
        return {
            "validator": validator,
            "status": "passed" if passed else "failed",
            "tested": tested,
            "errors": errors,
            "successes": successes,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, models={self.models})"

    @property
    def number_of_errors(self) -> int:
        return sum([model.number_of_errors for model in self.models if model.errored])


async def build_explore_dimensions(
    client: LookerClient,
    explore: Explore,
    ignore_hidden_fields: bool = False,
) -> None:
    """Creates Dimension objects for all dimensions in a given explore."""
    dimensions_json = await client.get_lookml_dimensions(
        explore.model_name, explore.name
    )

    dimensions: List[Dimension] = []
    for dimension_json in dimensions_json:
        dimension: Dimension = Dimension.from_json(
            dimension_json, explore.model_name, explore.name
        )
        if dimension.url is not None:
            dimension.url = client.base_url + dimension.url
        if not dimension.ignore and not (dimension.is_hidden and ignore_hidden_fields):
            dimensions.append(dimension)

    explore.dimensions = dimensions
    if len(explore.dimensions) == 0:
        logger.warning(
            f"Warning: Explore '{explore.name}' does not have any non-ignored "
            "dimensions and will not be validated."
        )
        explore.skipped = SkipReason.NO_DIMENSIONS


async def build_project(
    client: LookerClient,
    name: str,
    filters: Optional[List[str]] = None,
    include_dimensions: bool = False,
    ignore_hidden_fields: bool = False,
    include_all_explores: bool = False,
    get_full_project: bool = True,
) -> Project:
    """Creates an object (tree) representation of a LookML project."""
    if filters is None:
        filters = ["*/*"]
    is_complete_project = False

    if get_full_project:
        models = []
        fields = ["name", "project_name", "explores"]
        for lookmlmodel in await client.get_lookml_models(fields=fields):
            model = Model.from_json(lookmlmodel)
            if model.project_name == name:
                models.append(model)

        if not models:
            raise LookMlNotFound(
                name="project-models-not-found",
                title="No configured models found for the specified project.",
                detail=(
                    f"Go to {client.base_url}/projects and confirm "
                    "a) at least one model exists for the project and "
                    "b) it has an active configuration."
                ),
            )
        is_complete_project = True

    else:
        # Create a project with only the models specified in the filters
        logger.debug("Building project with only the filtered models")
        models: Dict[str, Model] = {}
        for filter in filters:
            model, explore = filter.split("/")
            if model not in models:
                models[model] = Model(name=model, project_name=name, explores=[])
            if explore not in models[model].explores:
                models[model].explores.append(Explore(name=explore, model_name=model))
        project = Project(name=name, models=models.values())
        models = project.models

    # Prune to selected explores for non-content validators
    if not include_all_explores:
        tasks: List[asyncio.Task[Any]] = []
        for model in models:
            model.explores = [
                explore
                for explore in model.explores
                if is_selected(model.name, explore.name, filters)
            ]
            if include_dimensions:
                for explore in model.explores:
                    task = asyncio.create_task(
                        build_explore_dimensions(client, explore, ignore_hidden_fields),
                        name=f"build_explore_dimensions_{explore.name}",
                    )
                    tasks.append(task)

        await asyncio.gather(*tasks)

    # Include empty models when including all explores
    if include_all_explores:
        project = Project(name, models)
    else:
        project = Project(name, [m for m in models if len(m.explores) > 0])

    # Indicates whether the project has all of the models/explores or just the selected ones
    project.is_complete_project = is_complete_project
    return project
