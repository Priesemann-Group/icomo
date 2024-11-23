"""Tools to create compartmental models and ODE systems."""

import logging
from collections.abc import Sequence
from types import EllipsisType

import graphviz
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

# import jax.numpy as jnp

logger = logging.getLogger(__name__)


def erlang_kernel(
    comp: ArrayLike, rate: ArrayLike, inflow: ArrayLike = 0
) -> tuple[Array, ArrayLike]:
    r"""Model the compartments delayed by an Erlang kernel.

    Utility function to model an Erlang kernel for a compartmental model. The shape is
    determined by the length of the last dimension of comp. For example, if comp C
    is an array of shape (...,3), the function implements the following system of ODEs:

    .. math::
       \begin{align*}
       \mathrm{rate\_indiv} &= 3 \cdot \mathrm{rate},&\\
        \frac{\mathrm dC^{(1)}(t)}{\mathrm dt} &= \mathrm{inflow} &-\mathrm{
        rate\_indiv}
        \cdot
            C^{(1)},\\
        \frac{\mathrm dC^{(2)}(t)}{\mathrm dt} &= \mathrm{rate\_indiv}  \cdot
            C^{(1)} &-\mathrm{rate\_indiv}  \cdot C^{(2)},\\
        \frac{\mathrm dC^{(3)}(t)}{\mathrm dt} &= \mathrm{rate\_indiv}  \cdot
            C^{(2)} &-\mathrm{rate\_indiv}  \cdot C^{(3)},\\
        \mathrm{outflow} &= \mathrm{rate\_indiv}  \cdot C^{(3)}
       \end{align*}
    ..

    Parameters
    ----------
    comp: jnp.ndarray of shape (..., n)
        The compartment on which the Erlang kernel is applied. The last dimension n is
        the length of the kernel.
    rate: float or ndarray
        The rate of the kernel, 1/rate is the mean time spent in total in all
        compartments

    Returns
    -------
    d_comp: jnp.ndarray of shape (..., n)
        The derivatives erlangerlangof the compartments
    outflow: float or ndarray
        The outflow from the last compartment
    """
    if not isinstance(comp, ArrayLike):
        raise AttributeError(
            f"comp has to be an array-like object, not of type {type(comp)}"
        )
    length = jnp.shape(comp)[-1]
    d_comp = jnp.zeros_like(comp)
    rate_indiv = rate * length
    d_comp = d_comp.at[..., 0].add(inflow - rate_indiv * comp[..., 0])
    d_comp = d_comp.at[..., 1:].add(
        -rate_indiv * comp[..., 1:] + rate_indiv * comp[..., :-1]
    )

    outflow = rate_indiv * comp[..., -1]

    return d_comp, outflow


class CompModel:
    """Class to help building a compartmental model.

    The model is built by adding flows between compartments. The model is then compiled
    into a function that can be used in an ODE.
    """

    def __init__(self, y_dict: PyTree[ArrayLike] = None):
        """Initialize the CompModel class.

        Parameters
        ----------
        y_dict: dict
            Dictionary of compartments. Keys are the names of the compartments and
            values are floats or ndarrays that represent their value.
        """
        if y_dict is None:
            y_dict = {}
        self.y = y_dict
        self._init_graph()

    def _init_graph(self):
        self.graph = graphviz.Digraph("CompModel")
        for key in self.y.keys():
            self.graph.node(key)
        self.graph.attr(rankdir="LR")

    @property
    def y(self) -> PyTree[Array]:
        """Returns the compartments of the model.

        Returns
        -------
        y: dict
            Dictionary of compartments. Keys are the names of the compartments and
            values are floats or ndarrays that represent their value.
        """
        return self._y

    @y.setter
    def y(self, y_dict: PyTree[ArrayLike]) -> None:
        """Set the compartments of the model.

        Also resets the derivatives of the compartments to zero.

        Parameters
        ----------
        y_dict: dict
            Dictionary of compartments. Keys are the names of the compartments and
            values are floats or ndarrays that represent their value.
        """
        self._y = jax.tree_util.tree_map(lambda x: jnp.array(x), y_dict)
        self._dy = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.y)
        self._init_graph()

    @property
    def dy(self):
        """Returns the derivative of the compartments.

        This should be returned by the function that defines the system of ODEs.

        Returns
        -------
        dComp: dict
        """
        return self._dy

    def flow(
        self,
        start_comp: str | int | Sequence[str | int],
        end_comp: str | int | Sequence[str | int],
        rate: ArrayLike,
        label: str | None = None,
        end_comp_is_erlang: bool = False,
    ) -> None:
        """
        Add a flow from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str or list
            Key of the start compartment.
        end_comp: str
            Key of the end compartment
        rate: float or ndarray
            rate of the flow to add between compartments, is multiplied by start_comp,
            so it should be broadcastable whith it.
        label: str, optional
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.
        end_comp_is_erlang: bool, default: False
            If True, end_comp points to a compartment with an Erlang distributed
            dwelling time, i.e., as the last dimension of the compartment is used
            for the Erlang distribution modeling. The flow is then only added to the
            first element of the last dimension, i.e. to self.y[end_comp][...,0].

        Returns
        -------
        None
        """
        self.add_deriv(start_comp, -rate * nested_indexing(self.y, start_comp))
        self.add_deriv(
            end_comp, rate * nested_indexing(self.y, start_comp), end_comp_is_erlang
        )

        self.graph.edge(start_comp, end_comp, label=label)

    def erlang_flow(
        self,
        start_comp: str | int | Sequence[str | int],
        end_comp: str | int | Sequence[str | int],
        rate: ArrayLike,
        label: str | None = None,
        end_comp_is_erlang: bool = False,
    ) -> None:
        """Add a flow with erlang kernel from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str or list
            start compartments of the flow. The length of last dimension the list is
            the shape of the
            Erlang kernel.
        end_comp: str or list
            end compartment of the flow
        rate: float or ndarray
            rate of the flow, equal to the inverse of the mean time spent in the Erlang
            kernel.
        label: str, optional
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.
        end_comp_is_erlang: bool, default: False
            If True, end_comp points to a compartment with an Erlang distributed
            dwelling time, i.e., as the last dimension of the compartment is used
            for the Erlang distribution modeling. The flow is then only added to the
            first element of the last dimension, i.e. to self.y[end_comp][...,0].
        """
        d_comp, outflow = erlang_kernel(
            comp=nested_indexing(self.y, start_comp), rate=rate
        )
        self.add_deriv(start_comp, d_comp)
        self.add_deriv(end_comp, outflow, end_comp_is_erlang)

        self.graph.edge(start_comp, end_comp, label=label)

    def delayed_copy(
        self,
        comp_to_copy: str | int | Sequence[str | int],
        delayed_comp: str | int | Sequence[str | int],
        tau_delay: ArrayLike,
    ) -> None:
        """Add a delayed copy of a compartment."""
        d_delayed_comp = delayed_copy_kernel(
            initial_comp=nested_indexing(self.y, comp_to_copy),
            delayed_comp=nested_indexing(self.y, delayed_comp),
            tau_delay=tau_delay,
        )
        self.add_deriv(delayed_comp, d_delayed_comp)

    def add_deriv(
        self,
        y_key: str | int | Sequence[str | int],
        additive_dy: ArrayLike,
        end_comp_is_erlang: bool = False,
    ) -> None:
        """Add a derivative to a compartment.

        Add a derivative to a compartment. This is useful if the derivative is not
        directly modelled by a flow between compartments.

        Parameters
        ----------
        y_key: str
            Key of the compartment
        additive_dy: float or ndarray
            Derivative to add to the compartment
        comp_is_list: bool, default: False

        """
        nested_indexing(
            tree=self.dy,
            indices=y_key,
            add=additive_dy,
            at=(Ellipsis, 0) if end_comp_is_erlang else None,
        )

    def view_graph(self, on_display: bool = True) -> None:
        """Display a graph of the compartmental model.

        Requires Graphviz (a non-python software) to be installed
        (https://www.graphviz.org/). It is also available in the conda-forge channel.
        See https://github.com/xflr6/graphviz?tab=readme-ov-file#installation.

        Parameters
        ----------
        on_display : bool
            If True, the graph is displayed in the notebook, otherwise it is saved as a
            pdf in the current folder and opened with the default pdf viewer.
        """
        if on_display:
            try:
                from IPython.display import display

                display(self.graph)
            except Exception:
                self.graph.view()
        else:
            self.graph.view()


def nested_indexing(
    tree: PyTree,
    indices: str | int | Sequence[str | int],
    add: ArrayLike | None = None,
    at: int | EllipsisType | slice | None | Sequence[EllipsisType | slice | int] = None,
):
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
    element : object
        The element of the nested structure of lists or tuples.

    """
    element = tree
    if not isinstance(indices, tuple | list):
        indices = [indices]
    for depth, index in enumerate(indices):
        if not depth == len(indices) - 1:
            element = element[index]
        else:
            if add is None:
                return element[index]
            else:
                if at is not None:
                    element[index] = element[index].at[at].add(add)
                else:
                    element[index] = element[index] + add


def delayed_copy_kernel(
    initial_comp: ArrayLike, delayed_comp: ArrayLike, tau_delay: ArrayLike
) -> Array:
    """Return the derivative to model the delayed copy of a compartment.

    The delay has the form of an Erlang kernel with shape parameter
    delayed_comp.shape[-1].

    Parameters
    ----------
    initial_comp : jnp.ndarray, shape: (...)
        The compartment that is copied
    delayed_comp : jnp.ndarray, shape (..., n)
        Compartment that is a delayed copies of initial_var, the last element of the
        last dimension is the compartment which has the same total content as
        initial_comp over time, but is delayed by tau_delay.
    tau_delay : float or ndarray
        The mean delay of the copy.

    Returns
    -------
    d_delayed_vars : list of floats or list of ndarrays
        The derivatives of the delayed compartments

    """
    length = jnp.shape(delayed_comp)[-1]
    inflow = initial_comp / tau_delay * length
    d_delayed_vars, outflow = erlang_kernel(inflow, delayed_comp[:], 1 / tau_delay)
    return d_delayed_vars
