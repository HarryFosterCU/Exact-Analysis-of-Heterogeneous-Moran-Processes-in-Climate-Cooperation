# Heterogeneous Evolutionary Process

## Software

To install a python virtual env:

```
$ python -m venv env
```

To activate the python env:

```
$ source env/bin/activate
```

To install dependencies:

```
$ python -m pip install -r requirements.txt
```

To check formatting of code according to [black](https://github.com/psf/black)
run:

```
$ python -m black --check src
```

To format code according to black:

```
$ python -m black src
```

### Benchmarks

To run all benchmarks:

    $ python -m pytest -vv .\src\benchmarks\

To run benchmarks for a specific benchmark file:

    $ python -m pytest -vv .\src\benchmarks\<file_name>.py

To compare benchmarks to another branch:

    $ ... pytest --benchmark-compare=<file_name_comparing_to> --benchmark-compare-fail=min:5%

To output benchmark data to a specific named file:

    $ ... pytest --benchmark-only --benchmark-save=<file_name>
