"""Text normalization: spoken numbers to digits, altitude extraction helpers."""
import re
from word2number import w2n
import logging

def spoken_numbers_to_digits(text):
    logger = logging.getLogger(__name__)
    try:
        text = re.sub(r'\bnative\b', 'negative', text, flags=re.IGNORECASE)
        text = re.sub(r'\boh\b', 'zero', text, flags=re.IGNORECASE)
        text = re.sub(r'\bo\b', 'zero', text, flags=re.IGNORECASE)
        number_pattern = re.compile(
            r'\b(?P<sign>negative |minus )?(?P<number>(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
            r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|point|and)\s?)+)\b',
            re.IGNORECASE
        )
        def repl(match):
            phrase = match.group('number').strip()
            if not phrase:
                return match.group(0)
            sign = match.group('sign')
            try:
                num = w2n.word_to_num(phrase)
                if sign:
                    num = -abs(num)
                return f" {num} "
            except Exception as e:
                logger.warning(f"word2number failed for phrase '{phrase}': {e}")
                return match.group(0)
        return number_pattern.sub(repl, text)
    except Exception as e:
        logger.error(f"Error in spoken_numbers_to_digits: {e}")
        return text

def extract_altitude_from_text(text):
    """
    Extracts an altitude value from a string ONLY if the word 'altitude' is present.
    Returns the altitude as an integer, or None if not found.
    """
    logger = logging.getLogger(__name__)
    try:
        text = text.lower()
        # This regex now ONLY looks for a number following the word "altitude"
        match = re.search(r'altitude\s*(?:to|is|at)?\s*([a-zA-Z0-9\-\s]+)', text)
        if match:
            candidate = match.group(1).strip()
            digit_match = re.search(r'(\d+)', candidate)
            if digit_match:
                return int(digit_match.group(1))
            try:
                # Try to convert number words to digits (e.g., "fifty")
                return w2n.word_to_num(candidate)
            except ValueError:
                logger.warning(f"word2number failed for altitude candidate '{candidate}'")
        return None
    except Exception as e:
        logger.error(f"Error in extract_altitude_from_text: {e}")
        return None
