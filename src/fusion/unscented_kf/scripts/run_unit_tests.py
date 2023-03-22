import coverage
import pytest

cov = coverage.Coverage(config_file=True)
cov.start()

pytest.main(['-x', '.', '-vv'])

cov.stop()
cov.save()
cov.html_report()

###run it in terminal vscode adds python libs for some reason ...