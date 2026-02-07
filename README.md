# UV Python Project - Local Setup Guide

This repository uses **uv**, a fast and modern Python package manager, to manage dependencies and run the project locally. This guide takes you from **cloning the repository** to **running the project on your machine** with zero guesswork.

## Prerequisites

Ensure the following are installed:

- **Python 3.9+**
- **Git**

Verify:
```bash
python --version
git --version
```

## About UV

`uv` is a next-generation Python package manager that:

* Replaces `pip`, `pip-tools`, and `virtualenv`
* Automatically manages virtual environments
* Uses `pyproject.toml` as the single source of truth
* Is extremely fast (Rust-based)
* You do not need to create or activate a virtual environment manually

## Install UV

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify installation:
```bash
uv --version
```

## Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

## Project Structure

```
.
├── pyproject.toml
├── uv.lock
├── src/
│   └── main.py
├── .env.example
└── README.md
```

## Install Dependencies

From the project root:
```bash
uv sync
```

This will:

- Create a virtual environment automatically
- Install all dependencies
- Use `uv.lock` for reproducible builds

## Environment Variables (Optional)

If environment variables are required, copy `.env.example` to `.env` and update the values as needed.

## Run the Project

From the project root:
```bash
uv run python src/main.py
```

**OR**
```bash
uv run main.py
```

This will run the `main.py` script in the `src` directory.

and the window will pop where you can drag drip the music files and see the music visualization 
