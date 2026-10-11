from showroom_coach import __version__
from showroom_coach.cli import main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
