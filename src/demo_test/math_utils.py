import asyncio

def add(a: int, b: int) -> int:
    return a + b

def safe_div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("b must not be zero")
    return a / b

async def async_mul(a: int, b: int) -> int:
    # 模拟异步开销
    await asyncio.sleep(0)
    return a * b
