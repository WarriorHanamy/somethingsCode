from hypothesis import given, strategies as st
from demo_test.math_utils import add, async_mul
import pytest


@given(st.integers(), st.integers())
def test_add_commutative(x, y):
    # 交换律
    assert add(x, y) == add(y, x)


@given(
    st.integers(min_value=-10_000, max_value=10_000),
    st.integers(min_value=-10_000, max_value=10_000),
    st.integers(min_value=-10_000, max_value=10_000),
)
def test_add_associative(a, b, c):
    # 结合律
    assert add(add(a, b), c) == add(a, add(b, c))


@pytest.mark.asyncio
@given(
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-1000, max_value=1000),
)
async def test_async_mul_linear(x, y):
    # 简单线性性质：x*y + x*y == x*(2*y)
    left = await async_mul(x, y)
    right = await async_mul(x, 2 * y)
    assert left + left == right
