# Hugging Face Flashcard Generator

A simple, CPU-friendly flashcard generator using Hugging Face Transformers.

## 1. Create and activate the virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Generate flashcards

```powershell
python flashcard_generator.py notes.txt
```

The result is saved as:

```text
flashcards.csv
```

## 4. Choose the number of cards

```powershell
python flashcard_generator.py notes.txt -n 5
```

## 5. Debug model output

If you ever get 0 cards:

```powershell
python flashcard_generator.py notes.txt --debug
```

This prints the raw Hugging Face model response.

## 6. CPU-friendly model

For a smaller model:

```powershell
python flashcard_generator.py notes.txt --model google/flan-t5-small
```

For better quality on a machine that can handle it:

```powershell
python flashcard_generator.py notes.txt --model google/flan-t5-base
```

## Project files

- `flashcard_generator.py` - main program
- `notes.txt` - sample input
- `requirements.txt` - Python dependencies
- `flashcards.csv` - generated output
