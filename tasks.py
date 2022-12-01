from invoke import task
from test import test_env


@task
def test(c):
    test_env.main()