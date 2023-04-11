conda activate scheduling_problem
coverage run -m pytest
coverage html --skip-empty --omit="*test.py" -d .\tests\coverage
Start-Process tests\coverage\index.html