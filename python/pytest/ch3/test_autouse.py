import pytest
import time

@pytest.fixture(autouse=True, scope="session")
def footer_session_scope():
    yield
    now = time.time()
    print("--")
    print(
        "finitshed : {}.".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        )
    )
    print("--------------")


@pytest.fixture(autouse=True)
def footer_function_scope():
    start = time.time()
    yield
    stop = time.time()
    delta = stop - start
    print("\ntest dutation : {:0.3} seconds.".format(delta))


def test_1():
    time.sleep(1)


def test_2():
    time.sleep(1.23)
