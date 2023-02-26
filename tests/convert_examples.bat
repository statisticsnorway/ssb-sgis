
del /S examples_files\*
del /S ../examples_files\*

poetry run jupytext --to ipynb examples.py

jupyter nbconvert --execute --to notebook --inplace examples.ipynb
jupyter nbconvert --to markdown examples.ipynb

move examples.md ..\examples.md

move examples_files ..\examples_files

del /S examples_files\*
del /S examples.ipynb
