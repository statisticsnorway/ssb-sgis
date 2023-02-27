import os
import shutil
import subprocess


if "ssb-gis-utils" in os.listdir():
    os.chdir("ssb-gis-utils/tests")
elif "gis_utils" in os.listdir():
    os.chdir("tests")


def py_to_md(
    file: str, png_folder_suffix: str = "_files", move_n_folders_up: int = 0
) -> None:
    """Converts .py to .ipynb, runs the notebook, saves to markdown, deletes ipynb

    Optionally moves the markdown file and the pictures one folder up
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


py_to_md("network_analysis_examples", move_n_folders_up=1)
py_to_md("network_analysis_demo_template", move_n_folders_up=1)
