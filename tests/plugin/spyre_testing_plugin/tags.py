# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared JUnit result-tagging helpers.

Both this repo's tests (via the autouse fixture in tests/conftest.py) and the
upstream vLLM tests (via pytest_collection_modifyitems in this plugin) stamp the
same `model__<id>` / `testtype__<tier>` tags. Upstream tests are collected from
outside tests/, so the conftest fixture never binds to them; the collection hook
tags them instead. Keeping the extraction here lets both callers share one
definition of "which param names name a model" and one tier source.

Tags are emitted as JUnit `<property name="tag" value="key__value"/>` elements
(a single property name `tag`, value `key__value`), matching the convention the
ClickHouse ingest reads.
"""

import os

# Parametrize argnames whose value names the model under test. Scalars hold the
# model id directly; `model_info` is a vLLM model-info object (its `.name` is the
# id); `model_ref_output` is a `(model_id, expected_output)` tuple whose first
# element is the id (tests/e2e/test_compile.py). Extend this as new model-bearing
# param names appear rather than guessing from arbitrary tuples.
MODEL_PARAM_NAMES = ("model", "model_path", "model_info", "model_ref_output")


def model_from_params(params):
    """Best-effort model id from a test's parametrization, or None."""
    for name in MODEL_PARAM_NAMES:
        if name not in params:
            continue
        value = params[name]
        if value is None:
            continue
        # (model_id, ...) tuple: the id is the first element.
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        # vLLM model-info object: prefer its .name attribute.
        name_attr = getattr(value, "name", None)
        return name_attr if name_attr is not None else str(value)
    return None


def test_tier():
    """Suite tier for this run, forwarded by CI (run-matrix-config resolves it
    via `make print-test-type` and exports SPYRE_TEST_TIER). Empty on a local
    run. Read live rather than cached at import so a test that monkeypatches the
    env still sees the change."""
    return os.environ.get("SPYRE_TEST_TIER", "")


def result_tags(params):
    """The (name, value) JUnit property pairs for a test with these params.

    A test with no recognized model param and a run with no tier yields an empty
    list, so callers can append unconditionally.
    """
    tags = []
    model = model_from_params(params)
    if model:
        tags.append(("tag", f"model__{model}"))
    tier = test_tier()
    if tier:
        tags.append(("tag", f"testtype__{tier}"))
    return tags
