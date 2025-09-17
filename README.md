# somethingsCode

This project just contains something I like, majorly wouble the codes. And the environment is required to be totally online.

Wishes:
- online CI/CD.
- JupyterNotebook mainly.




## Test
```shell
# 运行测试（-q 安静模式）
uv run pytest -q
```


```shell
# 运行覆盖率
uv run coverage run -m pytest
# 生成报告
uv run coverage report
```



## Pre-Commit
After you have done
```shell
pre-commit install
```
, the git commit will automatically first pre-commit. If there is any error, the commit process wil terminate and cannot completed normally.
