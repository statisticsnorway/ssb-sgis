import os
import shutil
import subprocess


if "sgis" in os.listdir():
    os.chdir("docs")
elif "ssb-sgis" in os.listdir():
    os.chdir("ssb-sgis/docs")
else:
    n = 0
    while "ssb-sgis" not in os.listdir():
        os.chdir("..")
        n += 1
        if n == 10:
            break
    os.chdir("ssb-sgis/docs")


def py_to_md(
    file: str, png_folder_suffix: str = "_files", move_n_folders_up: int = 0
) -> None:
    """Converts .py to .ipynb, runs the notebook, saves to markdown, deletes ipynb

    Optionally moves the markdown file and the pictures up n folders.
    """
    py_file = file + ".py"
    nb_file = file + ".ipynb"
    md_file = file + ".md"
    png_folder = f"{file}{png_folder_suffix}/"

    if os.path.exists(png_folder):
        shutil.rmtree(png_folder)
    if os.path.exists("../" + png_folder):
        shutil.rmtree("../" + png_folder)

    subprocess.call(["poetry", "run", "jupytext", "--to", "ipynb", py_file])
    subprocess.call(
        ["jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", nb_file]
    )
    subprocess.call(["jupyter", "nbconvert", "--to", "markdown", nb_file])

    if move_n_folders_up:
        for _ in range(move_n_folders_up):
            shutil.move(md_file, "../" + md_file)
            if os.path.exists(png_folder):
                shutil.move(png_folder, "../" + png_folder)
            md_file = "../" + md_file
            png_folder = "../" + png_folder
            print(md_file)
            print(png_folder)

    os.remove(nb_file)


def clean_up_md(file: str):
    unwanted_text = """
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
"""

    with open(file) as f:
        filedata = f.read()
        filedata = filedata.replace(unwanted_text, "")
    with open(file, "w") as f:
        f.write(filedata)


file = "network_analysis_examples"
py_to_md(file, move_n_folders_up=0)
clean_up_md("../" + file + ".md")

file = "network_analysis_demo_template"
py_to_md(file, move_n_folders_up=0)
clean_up_md("../" + file + ".md")
