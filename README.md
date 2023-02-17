# ssb-gis-utils

## Developer information

### Git LFS
The data in the testdata directory is stored with [Git LFS](https://git-lfs.com/).
Make sure `git-lfs` is installed and that you have run the command `git lfs install`
at least once. You only need to run this once per user account.

### Dependencies
[Poetry](https://python-poetry.org/) is used for dependency management. Install
poetry and run the command below from the root directory to install the dependencies.  
```shell
poetry install --no-root
```

### Tests
Use the following command from the root directory to run the tests: 
```shell
poetry run pytest  # from root directory
```

### Formatting
Format the code with `black`  and `isort` by running the following command from the
root directory:
```shell
poetry run black .
poetry run isort .
```
