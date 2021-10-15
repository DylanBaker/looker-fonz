import jsonschema
import pytest
from unittest.mock import Mock, patch
from spectacles.client import LookerClient
from spectacles.exceptions import ContentError, DataTestError, SqlError
from spectacles.runner import Runner
from utils import build_validation


@pytest.mark.vcr(match_on=["uri", "method", "raw_body"])
@pytest.mark.parametrize("fail_fast", [True, False])
def test_validate_sql_should_work(looker_client, fail_fast):
    runner = Runner(looker_client, "eye_exam")
    result = runner.validate_sql(
        filters=["eye_exam/users", "eye_exam/users__fail"], fail_fast=fail_fast
    )
    assert result["status"] == "failed"
    assert result["tested"][0]["passed"]
    assert not result["tested"][1]["passed"]
    if fail_fast:
        assert len(result["errors"]) == 1
    else:
        assert len(result["errors"]) > 1


@pytest.mark.vcr(match_on=["uri", "method", "raw_body"])
def test_validate_content_should_work(looker_client):
    runner = Runner(looker_client, "eye_exam")
    result = runner.validate_content(filters=["eye_exam/users", "eye_exam/users__fail"])
    assert result["status"] == "failed"
    assert result["tested"][0]["passed"]
    assert not result["tested"][1]["passed"]
    assert len(result["errors"]) > 0


@pytest.mark.vcr(match_on=["uri", "method", "raw_body"])
def test_validate_data_tests_should_work(looker_client):
    runner = Runner(looker_client, "eye_exam")
    result = runner.validate_data_tests(
        filters=["eye_exam/users", "eye_exam/users__fail"]
    )
    assert result["status"] == "failed"
    assert result["tested"][0]["passed"]
    assert not result["tested"][1]["passed"]
    assert len(result["errors"]) > 0


@patch("spectacles.validators.data_test.DataTestValidator.get_tests")
@patch("spectacles.validators.data_test.DataTestValidator.validate")
@patch("spectacles.runner.build_project")
@patch("spectacles.runner.LookerBranchManager")
def test_validate_data_tests_returns_valid_schema(
    mock_branch_manager,
    mock_build_project,
    mock_validate,
    mock_get_tests,
    project,
    model,
    explore,
    schema,
):
    error_message = "An error ocurred"

    def add_error_to_project(tests):
        project.models[0].explores[0].queried = True
        project.models[0].explores[0].errors = [
            DataTestError("", "", error_message, "", "", "")
        ]

    model.explores = [explore]
    project.models = [model]
    mock_build_project.return_value = project
    mock_validate.side_effect = add_error_to_project
    runner = Runner(client=Mock(spec=LookerClient), project="eye_exam")
    result = runner.validate_data_tests()
    assert result["status"] == "failed"
    assert result["errors"][0]["message"] == error_message
    jsonschema.validate(result, schema)


@patch("spectacles.validators.content.ContentValidator.validate")
@patch("spectacles.runner.build_project")
@patch("spectacles.runner.LookerBranchManager")
def test_validate_content_returns_valid_schema(
    mock_branch_manager,
    mock_build_project,
    mock_validate,
    project,
    model,
    explore,
    schema,
):
    error_message = "An error ocurred"

    def add_error_to_project(tests):
        project.models[0].explores[0].queried = True
        project.models[0].explores[0].errors = [
            ContentError("", "", error_message, "", "", "", "", "")
        ]

    model.explores = [explore]
    project.models = [model]
    mock_build_project.return_value = project
    mock_validate.side_effect = add_error_to_project
    runner = Runner(client=Mock(spec=LookerClient), project="eye_exam")
    result = runner.validate_content()
    assert result["status"] == "failed"
    assert result["errors"][0]["message"] == error_message
    jsonschema.validate(result, schema)


@patch("spectacles.validators.sql.SqlValidator.create_tests")
@patch("spectacles.validators.sql.SqlValidator.run_tests")
@patch("spectacles.runner.build_project")
@patch("spectacles.runner.LookerBranchManager")
def test_validate_sql_returns_valid_schema(
    mock_branch_manager,
    mock_build_project,
    mock_run_tests,
    mock_create_tests,
    project,
    model,
    explore,
    schema,
):
    error_message = "An error ocurred"

    def add_error_to_project(tests, profile):
        project.models[0].explores[0].queried = True
        project.models[0].explores[0].errors = [SqlError("", "", "", "", error_message)]

    model.explores = [explore]
    project.models = [model]
    mock_build_project.return_value = project
    mock_run_tests.side_effect = add_error_to_project
    runner = Runner(client=Mock(spec=LookerClient), project="eye_exam")
    result = runner.validate_sql()
    assert result["status"] == "failed"
    assert result["errors"][0]["message"] == error_message
    jsonschema.validate(result, schema)


def test_incremental_same_results_should_not_have_errors():
    main = build_validation("content")
    additional = build_validation("content")
    incremental = Runner._incremental_results(main, additional)
    assert incremental["status"] == "passed"
    assert incremental["errors"] == []
    assert incremental["tested"] == [
        dict(model="ecommerce", explore="orders", passed=True),
        dict(model="ecommerce", explore="sessions", passed=True),
        dict(model="ecommerce", explore="users", passed=True),
    ]


def test_incremental_with_fewer_errors_than_main():
    main = build_validation("content")
    additional = build_validation("content")
    additional["tested"][2]["passed"] = True
    additional["errors"] = []
    incremental = Runner._incremental_results(main, additional)
    assert incremental["status"] == "passed"
    assert incremental["errors"] == []
    assert incremental["tested"] == [
        dict(model="ecommerce", explore="orders", passed=True),
        dict(model="ecommerce", explore="sessions", passed=True),
        dict(model="ecommerce", explore="users", passed=True),
    ]


def test_incremental_with_more_errors_than_main():
    main = build_validation("content")
    additional = build_validation("content")
    additional["tested"][1]["passed"] = False
    extra_errors = [
        dict(
            model="ecommerce",
            explore="users",
            test=None,
            message="Another error occurred",
            metadata={},
        ),
        dict(
            model="ecommerce",
            explore="sessions",
            test=None,
            message="An error occurred",
            metadata={},
        ),
    ]
    additional["errors"].extend(extra_errors)
    incremental = Runner._incremental_results(main, additional)
    assert incremental["status"] == "failed"
    assert incremental["errors"] == extra_errors
    assert incremental["tested"] == [
        dict(model="ecommerce", explore="orders", passed=True),
        dict(model="ecommerce", explore="sessions", passed=False),
        dict(model="ecommerce", explore="users", passed=False),
    ]


def test_incremental_with_fewer_tested_explores_than_main():
    main = build_validation("content")
    additional = build_validation("content")
    _ = additional["tested"].pop(0)
    extra_error = dict(
        model="ecommerce",
        explore="users",
        test=None,
        message="Another error occurred",
        metadata={},
    )
    additional["errors"].append(extra_error)
    incremental = Runner._incremental_results(main, additional)
    assert incremental["status"] == "failed"
    assert incremental["errors"] == [extra_error]
    assert incremental["tested"] == [
        dict(model="ecommerce", explore="sessions", passed=True),
        dict(model="ecommerce", explore="users", passed=False),
    ]
