import re
import os
from typing import Dict, Any
from pathlib import Path
import logging

from pdf_extracter import process_pdf 

import nltk
# nltk.data.path  # Shows where it's trying to load from
nltk.download('punkt_tab', download_dir='punkt_tab')
nltk.data.path.append('punkt_tab')
from nltk.tokenize import sent_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def chunk_and_enumerate_sentences(pdf_result: Dict[str, Any]) -> str:
    """
    Takes the result from `process_pdf`, and returns sentence-enumerated text.
    Format: <1> Sentence one. <2> Sentence two.
    """
    combined_text = result["combined_text"]
    if not combined_text:
        logger.warning("No combined_text found in PDF result.")
        return ""

    sentences = sent_tokenize(combined_text)
    enumerated_text = ""
    for i, sentence in enumerate(sentences, 1):
        enumerated_text += f"<{i}> {sentence.strip()}\n"
    return enumerated_text


def save_parsed_sentences(parsed_text: str, output_path: str = "parsed_sentences.txt") -> None:
    """
    Save the sentence-enumerated text to a file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(parsed_text)
        logger.info(f"Parsed sentences saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving parsed sentences: {e}")


if __name__ == "__main__":
    input_pdf_path = "C:/Users/tejup/Downloads/extraction_purpose2.pdf"  # Replace with actual PDF path

    if not os.path.exists(input_pdf_path):
        logger.error(f"PDF file not found: {input_pdf_path}")
    else:
        result = process_pdf(input_pdf_path)
        parsed_text = chunk_and_enumerate_sentences(result)
        save_parsed_sentences(parsed_text)
