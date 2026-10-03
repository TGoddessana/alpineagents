"""FrozenDict and FrozenList (``alpineagents._frozen``): the read-only JSON containers inside history entries."""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from alpineagents import ModelEvent, State
from alpineagents._frozen import FrozenDict, FrozenList, freeze, thaw
from alpineagents.types import RawBlock, ToolCall

NESTED = {"name": "calc", "args": [1, 2, {"deep": [3, {"x": None}]}], "flag": True, "ratio": 0.5}


def frozen_dict() -> FrozenDict:
    return FrozenDict(NESTED)


def frozen_list() -> FrozenList:
    return FrozenList([3, 1, 2, {"a": [1]}])


# ================================================================ every mutator raises


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("name", "x"),
        lambda d: d.__setitem__("new", 1),
        lambda d: d.__delitem__("name"),
        lambda d: d.update({"name": "x"}),
        lambda d: d.update(name="x"),
        lambda d: d.pop("name"),
        lambda d: d.pop("missing", None),
        lambda d: d.popitem(),
        lambda d: d.clear(),
        lambda d: d.setdefault("name", 1),
        lambda d: d.setdefault("new", 1),
    ],
    ids=[
        "setitem",
        "setitem-new",
        "delitem",
        "update-mapping",
        "update-kwargs",
        "pop",
        "pop-default",
        "popitem",
        "clear",
        "setdefault-existing",
        "setdefault-new",
    ],
)
def test_every_dict_mutator_raises_type_error(mutate):
    d = frozen_dict()
    before = json.dumps(d)
    with pytest.raises(TypeError, match="read-only"):
        mutate(d)
    assert json.dumps(d) == before


def test_in_place_or_raises_type_error_on_a_dict():
    d = frozen_dict()
    with pytest.raises(TypeError, match="read-only"):
        d |= {"name": "x"}
    assert d == NESTED


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.__setitem__(0, 9),
        lambda x: x.__setitem__(slice(0, 2), [9]),
        lambda x: x.__delitem__(0),
        lambda x: x.__delitem__(slice(0, 2)),
        lambda x: x.append(4),
        lambda x: x.extend([4]),
        lambda x: x.insert(0, 4),
        lambda x: x.remove(3),
        lambda x: x.pop(),
        lambda x: x.pop(0),
        lambda x: x.clear(),
        lambda x: x.sort(),
        lambda x: x.reverse(),
    ],
    ids=[
        "setitem",
        "setitem-slice",
        "delitem",
        "delitem-slice",
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "pop-index",
        "clear",
        "sort",
        "reverse",
    ],
)
def test_every_list_mutator_raises_type_error(mutate):
    x = frozen_list()
    before = json.dumps(x)
    with pytest.raises(TypeError, match="read-only"):
        mutate(x)
    assert json.dumps(x) == before


def test_in_place_add_and_multiply_raise_type_error_on_a_list():
    x = frozen_list()
    with pytest.raises(TypeError, match="read-only"):
        x += [4]
    with pytest.raises(TypeError, match="read-only"):
        x *= 2
    assert x == [3, 1, 2, {"a": [1]}]


def test_the_error_says_how_to_change_a_copy_or_extra_data():
    with pytest.raises(TypeError) as e:
        frozen_dict()["name"] = "x"
    message = str(e.value)
    assert "FrozenDict is read-only" in message
    assert "setting key 'name'" in message
    assert "edit_extra_data()" in message
    with pytest.raises(TypeError) as e:
        frozen_list().append(1)
    assert "FrozenList is read-only" in str(e.value) and "append()" in str(e.value)


def test_nested_containers_are_frozen_too():
    d = frozen_dict()
    assert isinstance(d["args"], FrozenList)
    assert isinstance(d["args"][2], FrozenDict)
    assert isinstance(d["args"][2]["deep"], FrozenList)
    assert isinstance(d["args"][2]["deep"][1], FrozenDict)
    with pytest.raises(TypeError):
        d["args"].append(4)
    with pytest.raises(TypeError):
        d["args"][2]["deep"][1]["x"] = 1
    assert d == NESTED


def test_a_frozen_container_is_deep_frozen_however_it_is_built():
    assert isinstance(FrozenDict(a=[1, {"b": 2}])["a"][1], FrozenDict)
    assert isinstance(FrozenDict([("a", [1])])["a"], FrozenList)
    assert isinstance(FrozenList([[1], {"a": 1}])[0], FrozenList)
    assert isinstance(FrozenList([[1], {"a": 1}])[1], FrozenDict)
    assert isinstance(FrozenDict.fromkeys(["a", "b"], 0), FrozenDict)


# ================================================================ reading works like dict and list


def test_they_are_a_dict_and_a_list():
    d, x = frozen_dict(), frozen_list()
    assert isinstance(d, dict) and isinstance(x, list)
    assert d["name"] == "calc" and d.get("missing") is None and d.get("missing", 7) == 7
    assert list(d) == ["name", "args", "flag", "ratio"]
    assert list(d.items())[0] == ("name", "calc")
    assert "name" in d and "nope" not in d and len(d) == 4
    assert x[0] == 3 and x[-1] == {"a": [1]} and len(x) == 4 and 3 in x
    assert list(x) == [3, 1, 2, {"a": [1]}]
    assert sorted(x[:3]) == [1, 2, 3]  # sorted() is the way to sort a frozen list
    assert x.index(1) == 1 and x.count(2) == 1


def test_equality_with_plain_values_works_both_ways():
    d, x = frozen_dict(), frozen_list()
    assert d == NESTED and NESTED == d
    assert x == [3, 1, 2, {"a": [1]}] and [3, 1, 2, {"a": [1]}] == x
    assert d != {"name": "other"} and x != [3]
    assert d == FrozenDict(NESTED) and FrozenDict() == {}
    assert FrozenList() == [] and FrozenList([1]) != (1,)  # a list is not equal to a tuple, frozen or not


def test_they_are_not_hashable_like_dict_and_list():
    with pytest.raises(TypeError):
        hash(frozen_dict())
    with pytest.raises(TypeError):
        hash(frozen_list())


def test_unpacking_and_dict_conversion_work():
    d = FrozenDict({"path": "a.py", "line": 3})

    def f(path, line):
        return (path, line)

    assert f(**d) == ("a.py", 3)  # a tool call's args reach a function this way
    assert dict(d) == {"path": "a.py", "line": 3}
    assert {**d, "extra": 1} == {"path": "a.py", "line": 3, "extra": 1}
    assert [*frozen_list()][:2] == [3, 1]


def test_repr_is_the_plain_repr():
    assert repr(FrozenDict({"a": [1]})) == "{'a': [1]}"
    assert repr(FrozenList([1, {"a": 2}])) == "[1, {'a': 2}]"


def test_copies_made_by_operators_slices_and_copy_are_plain_and_editable():
    d, x = frozen_dict(), frozen_list()
    merged = d | {"extra": 1}
    assert type(merged) is dict and merged["extra"] == 1
    assert type(d.copy()) is dict
    d_copy = d.copy()
    d_copy["name"] = "changed"  # editable, and the original is untouched
    assert d["name"] == "calc"
    assert type(x[:2]) is list and type(x + [9]) is list and type(x * 2) is list
    sliced = x[:2]
    sliced.append(5)
    assert len(x) == 4


# ================================================================ json, pickle, deepcopy


def test_json_dumps_and_loads_work():
    d = frozen_dict()
    text = json.dumps(d)
    assert json.loads(text) == NESTED
    assert json.dumps(d, sort_keys=True) == json.dumps(NESTED, sort_keys=True)
    assert json.dumps(frozen_list()) == '[3, 1, 2, {"a": [1]}]'
    assert json.dumps({"call": d, "list": frozen_list()}) == json.dumps({"call": NESTED, "list": [3, 1, 2, {"a": [1]}]})


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_round_trips_into_frozen_containers(protocol):
    for original in (frozen_dict(), frozen_list()):
        loaded = pickle.loads(pickle.dumps(original, protocol=protocol))
        assert type(loaded) is type(original)
        assert loaded == original
        if isinstance(loaded, FrozenDict):
            assert isinstance(loaded["args"], FrozenList)
            with pytest.raises(TypeError):
                loaded["name"] = "x"
        else:
            assert isinstance(loaded[3], FrozenDict)
            with pytest.raises(TypeError):
                loaded.append(1)


def test_deepcopy_and_copy_work_and_stay_frozen():
    # Documented behavior: a copy of a frozen container is frozen again (so a deep copy of a history entry keeps
    # its values read-only); thaw() is the way to get an editable copy.
    d = frozen_dict()
    deep = copy.deepcopy(d)
    assert deep == d and deep is not d and type(deep) is FrozenDict
    assert deep["args"] is not d["args"]
    with pytest.raises(TypeError):
        deep["args"].append(1)
    shallow = copy.copy(d)
    assert shallow == d and type(shallow) is FrozenDict
    x = copy.deepcopy(frozen_list())
    assert x == frozen_list() and type(x) is FrozenList
    with pytest.raises(TypeError):
        x.append(1)
    # deepcopy of a plain container holding frozen ones works too
    box = copy.deepcopy({"args": d})
    assert box == {"args": NESTED}


# ================================================================ freeze and thaw


def test_freeze_converts_dicts_lists_and_tuples_deeply():
    value = {"a": [1, (2, 3), {"b": [4]}], "c": ("x",)}
    frozen = freeze(value)
    assert type(frozen) is FrozenDict
    assert type(frozen["a"]) is FrozenList and type(frozen["a"][1]) is FrozenList  # a tuple becomes a list
    assert type(frozen["a"][2]) is FrozenDict and type(frozen["a"][2]["b"]) is FrozenList
    assert type(frozen["c"]) is FrozenList
    assert frozen == {"a": [1, [2, 3], {"b": [4]}], "c": ["x"]}
    assert type(value) is dict and type(value["a"]) is list  # the argument is not changed


def test_freeze_leaves_scalars_and_other_objects_alone():
    marker = object()
    assert freeze(None) is None and freeze(True) is True and freeze(3) == 3 and freeze(0.5) == 0.5
    assert freeze("text") == "text"
    assert freeze(marker) is marker
    assert freeze({1, 2}) == {1, 2} and type(freeze({1, 2})) is set  # not JSON: left as it is
    assert freeze(b"bytes") == b"bytes"


def test_freeze_returns_an_already_frozen_value_as_it_is():
    d, x = frozen_dict(), frozen_list()
    assert freeze(d) is d and freeze(x) is x


def test_thaw_gives_an_editable_deep_copy():
    d = frozen_dict()
    plain = thaw(d)
    assert type(plain) is dict and type(plain["args"]) is list and type(plain["args"][2]) is dict
    assert type(plain["args"][2]["deep"][1]) is dict
    assert plain == NESTED
    plain["name"] = "changed"
    plain["args"].append(99)
    plain["args"][2]["deep"][1]["x"] = "set"
    assert d == NESTED  # the frozen original did not change
    assert thaw([1, (2, 3)]) == [1, [2, 3]] and type(thaw((1,))) is list
    assert thaw("text") == "text" and thaw(3) == 3


def test_freeze_thaw_round_trip_keeps_the_value():
    assert thaw(freeze(NESTED)) == NESTED
    assert freeze(thaw(frozen_dict())) == frozen_dict()


# ================================================================ where they show up


def test_tool_call_args_are_frozen_but_equal_to_the_plain_dict():
    call = ToolCall("edit", {"path": "a.py", "lines": [1, 2], "opts": {"force": True}}, "c1")
    assert isinstance(call.args, FrozenDict) and isinstance(call.args["lines"], FrozenList)
    assert isinstance(call.args["opts"], FrozenDict)
    assert call.args == {"path": "a.py", "lines": [1, 2], "opts": {"force": True}}
    assert json.loads(json.dumps(call.args)) == dict(call.args)
    with pytest.raises(TypeError, match="read-only"):
        call.args["path"] = "b.py"
    with pytest.raises(TypeError, match="read-only"):
        call.args["lines"].append(3)
    assert call == ToolCall("edit", {"path": "a.py", "lines": [1, 2], "opts": {"force": True}}, "c1")


def test_the_dict_a_call_was_built_from_is_not_shared_with_it():
    source = {"items": [1]}
    call = ToolCall("t", source, "c1")
    source["items"].append(2)
    source["new"] = 1
    assert call.args == {"items": [1]}


def test_raw_block_and_model_event_data_are_frozen():
    raw = RawBlock("anthropic", {"type": "thinking", "parts": ["a"]})
    assert isinstance(raw.data, FrozenDict) and isinstance(raw.data["parts"], FrozenList)
    with pytest.raises(TypeError):
        raw.data["type"] = "x"
    event = ModelEvent("fallback", "a failed", {"from": "a", "chain": ["a", "b"]})
    assert isinstance(event.data, FrozenDict) and isinstance(event.data["chain"], FrozenList)
    with pytest.raises(TypeError):
        event.data["chain"].append("c")


def test_state_extra_data_and_finish_answers_are_frozen():
    state = State(extra_data={"notes": {"bug": ["b.py"]}})
    assert isinstance(state.extra_data, FrozenDict)
    assert isinstance(state.extra_data["notes"]["bug"], FrozenList)
    with pytest.raises(TypeError, match="edit_extra_data"):
        state.extra_data["x"] = 1
    with pytest.raises(TypeError):
        state.extra_data["notes"]["bug"].append("c.py")
    state.finish({"summary": "ok", "files": ["a", "b"]})
    assert isinstance(state.answer, FrozenDict) and isinstance(state.answer["files"], FrozenList)
    assert state.answer == {"summary": "ok", "files": ["a", "b"]}
    with pytest.raises(TypeError):
        state.answer["files"].append("c")
