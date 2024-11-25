"""Tools for working with nested structures of lists, tuples or dictionaries."""

from collections.abc import Generator, Sequence
from types import EllipsisType
from typing import Any, Optional

from jaxtyping import ArrayLike, PyTree


def walk_tree(tree: PyTree) -> Generator[tuple[list, Any]]:
    """Walk through a tree and yield the indices and values of the leaves.

    Parameters
    ----------
    tree :
        The tree to walk through. Can only be a nested tree of lists, tuples and/or
        dictionaries.

    Yields
    ------
    indices, value
        The indices as a list and values of the leaves.
    """
    cursor = []
    generators_list = []
    value = tree
    key = None
    while True:
        if isinstance(value, dict | list | tuple):
            cursor.append(key)
            if isinstance(value, dict):
                generator = iter(value.items())
            else:
                generator = enumerate(value)
            generators_list.append(generator)

        key, value = next(generators_list[-1], (None, None))
        cursor[-1] = key
        if key is None:
            generators_list.pop()
            cursor.pop()
            if len(generators_list) == 0:
                break
            continue
        elif not isinstance(value, dict | list | tuple):
            yield cursor, value


def nested_indexing(
    tree: PyTree,
    indices: str | int | Sequence[str | int],
    add: Optional[ArrayLike] = None,
    at: int | EllipsisType | slice | None | Sequence[EllipsisType | slice | int] = None,
) -> Any:
    """Return the element of a nested structure of lists or tuples.

    Parameters
    ----------
    tree :
        The nested structure of lists or tuples.
    indices :
        The indices of the element to return.
    add : optional
        The element to add to the nested structure of lists or tuples.
    at : optional
        Specifies the position where to add the element.

    Returns
    -------
    element
        The element of the nested structure of lists or tuples.

    """
    element = tree
    if not isinstance(indices, tuple | list):
        indices = [indices]
    for depth, index in enumerate(indices):
        if depth == len(indices) - 1:
            if add is not None:
                if at is not None:
                    element[index] = element[index].at[at].add(add)
                else:
                    element[index] = element[index] + add
        element = element[index]
    return element
