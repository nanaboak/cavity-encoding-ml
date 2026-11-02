# Contributing

Thank you for helping to improve `cavity-encoding-ml`. Bug reports, questions, ideas and code are all welcome.

## Report a problem or ask a question

Open an issue at https://github.com/nanaboak/cavity-encoding-ml/issues. Please add the following.

- What you did, with a small piece of code or data that shows the problem.
- What you expected, and what happened instead.
- Your Python version and the output of `pip show cavity-encoding-ml`.

Do not upload confidential production data. Synthetic or anonymised data are enough.

## Suggest a change

1. Fork the repository and create a branch.
2. Install the package with the test tools: `pip install -e ".[dev]"`.
3. Make your change and add a test for it in `tests/`.
4. Run `pytest` and check that all tests pass.
5. Add a line to `CHANGELOG.md` and open a pull request with a short description.

## Style

Keep functions small, give them a docstring and use clear names. Results of the method must stay the same unless a change is discussed in an issue first, because the package reproduces a published method.

## Getting help

If you are unsure where to start, open an issue and ask. Beginners are welcome.
