import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_MODEL = "google/flan-t5-base"


# ---------------------------------------------------------
# READ INPUT FILE
# ---------------------------------------------------------

def read_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()

    if not text:
        raise SystemExit(f"Input file is empty: {path}")

    return text


# ---------------------------------------------------------
# SPLIT TEXT INTO SENTENCES
# ---------------------------------------------------------

def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------

def build_generator(model_name: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using model: {model_name}")
    print("Device:", "GPU" if device == "cuda" else "CPU")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = model.to(device)

    return tokenizer, model, device


# ---------------------------------------------------------
# GENERATE ONE FLASHCARD
# ---------------------------------------------------------

def generate_one_card(tokenizer, model, device, fact):

    # ---------------------------------------------
    # Generate QUESTION
    # ---------------------------------------------

    question_prompt = f"""
Create ONE simple study question from this fact.

FACT:
{fact}

Rules:
- Ask about the main information in the fact.
- Use only information from the fact.
- Return ONLY the question.
- Do not provide the answer.
- Do not repeat the entire fact.
"""

    question_inputs = tokenizer(
        question_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    question_outputs = model.generate(
        **question_inputs,
        max_new_tokens=50,
        num_beams=4,
        do_sample=False
    )

    question = tokenizer.decode(
        question_outputs[0],
        skip_special_tokens=True
    ).strip()

    # Remove unwanted prefixes
    question = re.sub(
        r"^(question|q)\s*:\s*",
        "",
        question,
        flags=re.IGNORECASE
    ).strip()

    print("Question:", question)

    # ---------------------------------------------
    # Validate question
    # ---------------------------------------------

    if not question:
        return None

    if len(question) < 5:
        return None

    if len(question) > 200:
        return None

    # Make sure it ends with ?
    if not question.endswith("?"):
        question += "?"

    # ---------------------------------------------
    # Generate ANSWER
    # ---------------------------------------------

    answer_prompt = f"""
Answer the question using ONLY the fact below.

FACT:
{fact}

QUESTION:
{question}

Rules:
- Return ONLY the answer.
- Keep the answer short.
- Do not repeat the question.
- Do not add explanations.
"""

    answer_inputs = tokenizer(
        answer_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    answer_outputs = model.generate(
        **answer_inputs,
        max_new_tokens=60,
        num_beams=4,
        do_sample=False
    )

    answer = tokenizer.decode(
        answer_outputs[0],
        skip_special_tokens=True
    ).strip()

    # Remove unwanted prefixes
    answer = re.sub(
        r"^(answer|a)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    print("Answer:", answer)

    # ---------------------------------------------
    # Validate answer
    # ---------------------------------------------

    if not answer:
        return None

    if len(answer) < 2:
        return None

    if answer.lower() in [
        "answer",
        "question",
        "unknown",
        "i don't know"
    ]:
        return None

    return question, answer


# ---------------------------------------------------------
# GENERATE ALL FLASHCARDS
# ---------------------------------------------------------

def generate_cards(generator, facts, count):

    tokenizer, model, device = generator

    cards = []
    used_questions = set()

    # Never ask for more cards than available facts
    total = min(count, len(facts))

    for i in range(total):

        fact = facts[i]

        print()
        print("=" * 50)
        print(f"Creating card {i + 1}/{total}")
        print("Fact:", fact)
        print("=" * 50)

        card = generate_one_card(
            tokenizer,
            model,
            device,
            fact
        )

        if card is None:
            print("Could not create this card.")
            continue

        question, answer = card

        # -----------------------------------------
        # DUPLICATE CHECK
        # -----------------------------------------

        question_key = question.lower().strip()

        if question_key in used_questions:
            print("Duplicate question skipped.")
            continue

        used_questions.add(question_key)

        cards.append((question, answer))

    return cards


# ---------------------------------------------------------
# CLEAN CARDS
# ---------------------------------------------------------

def clean_cards(cards):

    cleaned = []
    seen = set()

    for question, answer in cards:

        question = question.strip()
        answer = answer.strip()

        key = (
            question.lower(),
            answer.lower()
        )

        if question and answer and key not in seen:

            seen.add(key)

            cleaned.append(
                (question, answer)
            )

    return cleaned


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

def save_csv(cards, output_path):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Question",
            "Answer"
        ])

        writer.writerows(cards)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate flashcards using Hugging Face FLAN-T5."
    )

    parser.add_argument(
        "input",
        help="Path to TXT file"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="flashcards.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "-n",
        "--cards-per-chunk",
        type=int,
        default=5,
        help="Number of flashcards to generate"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name"
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Validate number
    # ---------------------------------------------

    if args.cards_per_chunk < 1:
        raise SystemExit(
            "Number of cards must be at least 1."
        )

    # ---------------------------------------------
    # Read notes
    # ---------------------------------------------

    text = read_text(args.input)

    print()
    print("Input characters:", len(text))

    # ---------------------------------------------
    # Split into facts
    # ---------------------------------------------

    facts = split_sentences(text)

    print("Facts found:", len(facts))

    for i, fact in enumerate(facts, 1):
        print(f"{i}. {fact}")

    # ---------------------------------------------
    # Load model
    # ---------------------------------------------

    generator = build_generator(args.model)

    # ---------------------------------------------
    # Generate cards
    # ---------------------------------------------

    cards = generate_cards(
        generator,
        facts,
        args.cards_per_chunk
    )

    # ---------------------------------------------
    # Clean
    # ---------------------------------------------

    cards = clean_cards(cards)

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    save_csv(
        cards,
        args.output
    )

    print()
    print("=" * 50)
    print("FLASHCARD GENERATION COMPLETE")
    print("=" * 50)

    print(
        f"Saved {len(cards)} flashcards to {args.output}"
    )

    # ---------------------------------------------
    # Display cards
    # ---------------------------------------------

    print()

    for i, (question, answer) in enumerate(cards, 1):

        print(f"Card {i}")
        print("Q:", question)
        print("A:", answer)
        print()


# ---------------------------------------------------------
# PROGRAM START
# ---------------------------------------------------------

if __name__ == "__main__":
    main()