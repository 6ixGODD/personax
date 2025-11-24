import personax


def test_version() -> None:
    assert hasattr(personax, "__version__")
    assert isinstance(personax.__version__, str)
    assert len(personax.__version__) > 0


def test_title() -> None:
    assert hasattr(personax, "__title__")
    assert isinstance(personax.__title__, str)
    assert len(personax.__title__) > 0
