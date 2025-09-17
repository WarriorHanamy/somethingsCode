import pytest
from demo_test.math_utils import add, safe_div, async_mul


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (-1, 1, 0),
        (0, 0, 0),
    ],
)
def test_add(a, b, expected):
    assert add(a, b) == expected


def test_safe_div_normal():
    assert safe_div(6, 3) == 2


def test_safe_div_zero_division():
    with pytest.raises(ZeroDivisionError):
        safe_div(1, 0)


@pytest.mark.asyncio
async def test_async_mul():
    assert await async_mul(3, 4) == 12
