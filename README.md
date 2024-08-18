# feedback-prize-english-language-learning
This project is part of the [Feedback Prize - English Language Learning on Kaggle](https://www.kaggle.com/competitions/feedback-prize-english-language-learning).


The goal of this project is to develop models that accurately assess the quality of English language learners' writing, in line with the objectives of the Kaggle competition.



# Source folder
[Go to the source folder](./src/feedback_prize_english_language_learning/lib)

## Source folder structure
lib/

├── data/

│ ├── init.py

│ ├── data_module.py

│ └── data_utils.py

├── models/

│ ├── BertRegression.py

│ └── init.py

├── visualizations/

│ ├── init.py

│ └── eda.py

├── init.py

├── config.py

├── consts.py

├── trainer.py

└── utils.py



To get started, explore the files in the `lib` directory which contains the core logic of the project.

## Test Results Summary

| Test Type  | Test Metric | DataLoader 0 Value |
|------------|-------------|-------------------|
| Vocabulary | RMSE        | 0.42812013626098633  |
| Vocabulary | Loss        | 0.10702415555715561  |
| Grammar    | RMSE        | 0.47667704796791077  |
| Grammar    | Loss        | 0.11766650527715683  |


# Setup developer environment

To start, you need to setup your local machine.

## Getting Started with code running
```
poetry install
poetry shell
poetry run pytest
poetry run mypy .
poetry add --dev pre-commit
poetry run pre-commit install
./pre-commit.sh  src/
<!-- poetry run pre-commit run --all-files -->
poetry run pylint feedback_prize_english_language_learning
poetry run bandit -r feedback_prize_english_language_learning


```

## Setup venv

You need to setup virtual environment, simplest way is to run from project root directory:

```bash
$ . ./setup_dev_env.sh
$ source venv/bin/activate
```
This will create a new venv and run `pip install -r requirements-dev.txt`.
Last line shows how to activate the environment.

## Install pre-commit

To ensure code quality we use pre-commit hook with several checks. Setup it by:

```
pre-commit install
```

All updated files will be reformatted and linted before the commit.

To reformat and lint all files in the project, use:

`pre-commit run --all-files`

The used linters are configured in `.pre-commit-config.yaml`. You can use `pre-commit autoupdate` to bump tools to the latest versions.

## Autoreload within notebooks

When you install project's package add below code (before imports) in your notebook:
```
# Load the "autoreload" extension
%load_ext autoreload
# Change mode to always reload modules: you change code in src, it gets loaded
%autoreload 2
```
Read more about different modes in [documentation](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html).

All code should be in `src/` to make reusability and review straightforward, keep notebooks simple for exploratory data analysis.
See also [Cookiecutter Data Science opinion](https://drivendata.github.io/cookiecutter-data-science/#notebooks-are-for-exploration-and-communication).

# Project documentation

In `docs/` directory are Sphinx RST/Markdown files.

To build documentation locally, in your configured environment, you can use `build_docs.sh` script:

```bash
$ ./build_docs.sh
```

Then open `public/index.html` file.

Please read the official [Sphinx documentation](https://www.sphinx-doc.org/en/master/) for more details.



### Github Actions Documentation

By default **Github Actions** pipelines have `documentation` workflow which will build sphinx documentation automatically on main branch - and it will push it to a branch - it can be hosted on **Github Pages** if you enable it.

To access it, you need to enable it, on **Github repository -> Settings -> Pages** page select **Deploy from a branch** and select **gh-pages**. Link will appear here after deployment.

**WARNING:** Only on Github Enterprise you can make it private so only people with repository access can view it.

Please read more about it [here](https://docs.github.com/en/pages/quickstart).

# Jupyter notebooks and jupytext

To make notebooks more friendly for code review and version control we use `jupytext` to sync notebooks with python files. If you have not used it before, please read [jupytext documentation](https://jupytext.readthedocs.io/en/latest/).

There is pre-commit hook which automatically generates and syncs notebooks with python files on each commit.

Please ensure you do not edit/modify manually or by other means generated py:percent files as they will conflict with jupytext change detection and lead to endless loop.
Treat them as read-only files and edit only notebooks.# Semantic version bump

To bump version of the library please use `bump2version` which will update all version strings.

NOTE: Configuration is in `.bumpversion.cfg` and **this is a main file defining version which should be updated only with bump2version**.

For convenience there is bash script which will create commit, to use it call:

```bash
# to create a new commit by increasing one semvar:
$ ./bump_version.sh minor
$ ./bump_version.sh major
$ ./bump_version.sh patch
# to see what is going to change run:
$ ./bump_version.sh --dry-run major
```
Script updates **VERSION** file and setup.cfg automatically uses that version.

You can configure it to update version string in other files as well - please check out the bump2version configuration file.


