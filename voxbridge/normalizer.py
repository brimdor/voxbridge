"""Text normalization module for VoxBridge TTS.

Pre-processes text before TTS synthesis, converting numbers, dates, times,
currencies, phone numbers, ordinals, and abbreviations into spoken form.
Works as a preprocessor pipe that any TTS backend can use.

Example:
    ```python
    from voxbridge.normalizer import Normalizer

    norm = Normalizer()
    print(norm.normalize("$1,234.56"))        # "one thousand two hundred thirty four dollars and fifty six cents"
    print(norm.normalize("June 15, 2026"))     # "June fifteenth, twenty twenty six"
    print(norm.normalize("5:30 p.m."))          # "five thirty PM"
    print(norm.normalize("1-800-555-0199"))     # "one eight hundred five five five zero one nine nine"
    print(norm.normalize("1st, 2nd, 3rd"))       # "first, second, third"
    ```
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Number-to-word mapping
_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}

_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}

_ORDINAL_ONES = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth",
}

_ORDINAL_TENS = {
    2: "twentieth", 3: "thirtieth", 4: "fortieth", 5: "fiftieth",
    6: "sixtieth", 7: "seventieth", 8: "eightieth", 9: "ninetieth",
}

_MONTHS = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June", "july": "July",
    "august": "August", "september": "September", "october": "October",
    "november": "November", "december": "December",
}

_ABBREVIATIONS = {
    "mr.": "mister", "mrs.": "missus", "ms.": "miss",
    "dr.": "doctor", "prof.": "professor", "sr.": "senior",
    "jr.": "junior", "st.": "street", "ave.": "avenue",
    "blvd.": "boulevard", "dept.": "department", "inc.": "incorporated",
    "ltd.": "limited", "corp.": "corporation", "co.": "company",
    "est.": "established", "gov.": "governor", "gen.": "general",
    "sgt.": "sergeant", "lt.": "lieutenant", "capt.": "captain",
    "cmdr.": "commander", "adm.": "admiral", "pres.": "president",
    "rep.": "representative", "sen.": "senator", "rev.": "reverend",
    "hon.": "honorable", "vs.": "versus", "etc.": "et cetera",
    "approx.": "approximately", "apt.": "apartment",
    "asap": "as soon as possible", "fyi": "for your information",
    "imo": "in my opinion", "btw": "by the way",
    "brb": "be right back", "lol": "laugh out loud",
    "rofl": "rolling on the floor laughing", "idk": "I don't know",
    "tbh": "to be honest", "imo": "in my opinion", "imho": "in my humble opinion",
    "faq": "frequently asked questions", "aka": "also known as",
    "pdf": "PDF", "url": "URL", "html": "HTML", "css": "CSS",
    "api": "API", "sdk": "SDK", "cpu": "CPU", "gpu": "GPU",
    "ram": "RAM", "ssd": "SSD", "usb": "USB", "wifi": "WiFi",
}


def _number_to_words(n: int) -> str:
    """Convert an integer to its English word representation."""
    if n < 0:
        return "minus " + _number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        if n % 10 == 0:
            return _TENS[n // 10]
        return _TENS[n // 10] + " " + _ONES[n % 10]
    if n < 1000:
        if n % 100 == 0:
            return _ONES[n // 100] + " hundred"
        return _ONES[n // 100] + " " + _number_to_words(n % 100)
    if n < 1_000_000:
        if n % 1000 == 0:
            return _number_to_words(n // 1000) + " thousand"
        return _number_to_words(n // 1000) + " thousand " + _number_to_words(n % 1000)
    if n < 1_000_000_000:
        if n % 1_000_000 == 0:
            return _number_to_words(n // 1_000_000) + " million"
        return _number_to_words(n // 1_000_000) + " million " + _number_to_words(n % 1_000_000)
    if n < 1_000_000_000_000:
        if n % 1_000_000_000 == 0:
            return _number_to_words(n // 1_000_000_000) + " billion"
        return _number_to_words(n // 1_000_000_000) + " billion " + _number_to_words(n % 1_000_000_000)
    return str(n)  # fallback for very large numbers


def _ordinal_to_words(n: int) -> str:
    """Convert an integer to its English ordinal word."""
    if n < 0:
        return str(n)  # no ordinal for negatives
    if n < 20:
        if n in _ORDINAL_ONES:
            return _ORDINAL_ONES[n]
        return _ONES[n] + "th"
    if n < 100:
        if n % 10 == 0:
            return _ORDINAL_TENS.get(n // 10, _TENS[n // 10] + "th")
        return _TENS[n // 10] + " " + _ordinal_to_words(n % 10)
    # For larger numbers, use "Nth" form
    if n % 100 in (11, 12, 13):
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return _number_to_words(n) + suffix


def _digit_by_digit(number_str: str) -> str:
    """Spell out each digit of a number individually."""
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    return " ".join(digit_words.get(ch, ch) for ch in number_str)


class Normalizer:
    """Text normalizer that preprocesses text for TTS synthesis.

    Converts currencies, dates, times, phone numbers, ordinals, and
    abbreviations into spoken form. Can also extract and process
    expression tags (<laugh>, <breath>, etc.) for use with ExpressionProcessor.

    Example:
        ```python
        norm = Normalizer()
        text = norm.normalize("Meeting on June 15, 2026 at 5:30 p.m.")
        # "Meeting on June fifteenth, twenty twenty six at five thirty PM"
        ```
    """

    def __init__(
        self,
        *,
        currency: bool = True,
        dates: bool = True,
        phone_numbers: bool = True,
        time: bool = True,
        ordinals: bool = True,
        abbreviations: bool = True,
        expressions: bool = True,
    ):
        """Initialize the normalizer with feature flags.

        Args:
            currency: Expand currency expressions
            dates: Convert dates to spoken form
            phone_numbers: Spell out phone numbers digit by digit
            time: Convert time expressions to spoken form
            ordinals: Convert ordinals (1st, 2nd, 3rd) to words
            abbreviations: Expand common abbreviations
            expressions: Extract and preserve expression tags
        """
        self.currency = currency
        self.dates = dates
        self.phone_numbers = phone_numbers
        self.time = time
        self.ordinals = ordinals
        self.abbreviations = abbreviations
        self.expressions = expressions

    def normalize(self, text: str) -> str:
        """Apply all normalization steps to text.

        Processing order:
        1. Extract expression tags (preserve them)
        2. Normalize currency
        3. Normalize time
        4. Normalize phone numbers
        5. Normalize dates
        6. Normalize ordinals
        7. Expand abbreviations
        8. Re-insert expression tags

        Args:
            text: Raw input text

        Returns:
            Normalized text ready for TTS
        """
        # Step 1: Extract expression tags to preserve them
        extracted = {}
        if self.expressions:
            text, extracted = self._extract_expression_tags(text)

        # Step 2: Normalize currency
        if self.currency:
            text = self._normalize_currency(text)

        # Step 3: Normalize time (before dates since "5:30 p.m." contains colon)
        if self.time:
            text = self._normalize_time(text)

        # Step 4: Normalize phone numbers
        if self.phone_numbers:
            text = self._normalize_phone_numbers(text)

        # Step 5: Normalize dates
        if self.dates:
            text = self._normalize_dates(text)

        # Step 6: Normalize ordinals
        if self.ordinals:
            text = self._normalize_ordinals(text)

        # Step 7: Expand abbreviations
        if self.abbreviations:
            text = self._expand_abbreviations(text)

        # Step 8: Re-insert expression tags
        if extracted:
            text = self._reinsert_expression_tags(text, extracted)

        return text

    # --- Expression tag handling ---

    _EXPRESSION_TAG_RE = re.compile(
        r"<(laugh|breath|sigh|cough|gasp|groan|chuckle|whisper|shout|pause)(?:\s+([^>]*))?\s*/?>",
        re.IGNORECASE,
    )

    def _extract_expression_tags(self, text: str) -> tuple[str, dict[str, str]]:
        """Extract expression tags from text, replacing with placeholders.

        Returns:
            Tuple of (text_with_placeholders, {placeholder: original_tag_text})
        """
        placeholders = {}
        counter = 0

        def _replace(m: re.Match) -> str:
            nonlocal counter
            tag_name = m.group(1).lower()
            tag_content = m.group(2) or ""
            key = f"__EXPR_{counter}__"
            placeholders[key] = f"<{tag_name}" + (f" {tag_content}" if tag_content else "") + ">"
            counter += 1
            return key

        text = self._EXPRESSION_TAG_RE.sub(_replace, text)
        return text, placeholders

    def _reinsert_expression_tags(self, text: str, placeholders: dict[str, str]) -> str:
        """Re-insert expression tags from placeholders."""
        for key, tag in placeholders.items():
            text = text.replace(key, tag)
        return text

    # --- Currency ---

    _CURRENCY_RE = re.compile(
        r"(?:\$|€|£|¥)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)"
    )
    _CURRENCY_EURO_RE = re.compile(
        r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:€|EUR)"
    )

    def _normalize_currency(self, text: str) -> str:
        """Normalize currency expressions to spoken form."""
        # $12,458.75 → "twelve thousand four hundred fifty eight dollars and seventy five cents"
        def _usd_replace(m: re.Match) -> str:
            amount_str = m.group(1).replace(",", "")
            try:
                amount = float(amount_str)
            except ValueError:
                return m.group(0)
            # Check if has cents
            if "." in m.group(1):
                dollars = int(amount)
                cents = round((amount - dollars) * 100)
                result = _number_to_words(dollars) + (" dollars" if dollars != 1 else " dollar")
                if cents > 0:
                    result += " and " + _number_to_words(cents) + (" cents" if cents != 1 else " cent")
                return result
            else:
                dollars = int(amount)
                return _number_to_words(dollars) + (" dollars" if dollars != 1 else " dollar")

        text = self._CURRENCY_RE.sub(_usd_replace, text)

        # Handle euro amounts: 12€ → "twelve euros"
        def _euro_replace(m: re.Match) -> str:
            amount_str = m.group(1).replace(",", "")
            try:
                amount = float(amount_str)
            except ValueError:
                return m.group(0)
            if "." in m.group(1):
                euros = int(amount)
                cents = round((amount - euros) * 100)
                if cents > 0:
                    return _number_to_words(euros) + " euros and " + _number_to_words(cents) + " cents"
                return _number_to_words(euros) + " euros"
            return _number_to_words(int(amount)) + " euros"

        text = self._CURRENCY_EURO_RE.sub(_euro_replace, text)
        return text

    # --- Time ---

    _TIME_RE = re.compile(
        r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?|AM|PM|am|pm)?\b",
        re.IGNORECASE,
    )

    def _normalize_time(self, text: str) -> str:
        """Normalize time expressions to spoken form."""
        def _time_replace(m: re.Match) -> str:
            hour = int(m.group(1))
            minute = int(m.group(2))
            ampm = m.group(3)

            # Convert hour words
            if minute == 0:
                result = _number_to_words(hour) + (" hundred hours" if ampm is None else "")
            elif minute < 10:
                result = _number_to_words(hour) + " oh " + _number_to_words(minute)
            else:
                result = _number_to_words(hour) + " " + _number_to_words(minute)

            if ampm:
                # Normalize am/pm
                ampm_clean = ampm.upper().replace(".", "")
                if ampm_clean in ("AM", "PM"):
                    result += " " + ampm_clean

            return result

        text = self._TIME_RE.sub(_time_replace, text)
        return text

    # --- Phone numbers ---

    _PHONE_RE = re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )

    def _normalize_phone_numbers(self, text: str) -> str:
        """Normalize phone numbers to digit-by-digit spoken form."""
        def _phone_replace(m: re.Match) -> str:
            number = m.group(0)
            # Remove all non-digit characters except leading +
            digits_only = re.sub(r"[^\d]", "", number)
            # Handle country code
            if digits_only.startswith("1") and len(digits_only) == 11:
                digits_only = digits_only[1:]  # strip the 1
            return _digit_by_digit(digits_only)

        text = self._PHONE_RE.sub(_phone_replace, text)
        return text

    # --- Dates ---

    _MONTH_DAY_YEAR_RE = re.compile(
        r"\b("
        r"January|February|March|April|May|June|July|"
        r"August|September|October|November|December"
        r")\s+(\d{1,2}),?\s+(\d{4})\b",
        re.IGNORECASE,
    )
    _MDY_NUMERIC_RE = re.compile(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"
    )

    def _normalize_dates(self, text: str) -> str:
        """Normalize date expressions to spoken form."""
        # "June 15, 2026" → "June fifteenth, twenty twenty six"
        def _month_day_year_replace(m: re.Match) -> str:
            month = m.group(1)
            day = int(m.group(2))
            year = int(m.group(3))
            return f"{month} {_ordinal_to_words(day)}, {Normalizer._year_to_words(year)}"

        text = self._MONTH_DAY_YEAR_RE.sub(_month_day_year_replace, text)
        return text

    @staticmethod
    def _year_to_words(year: int) -> str:
        """Convert a year to spoken form: 2026 → 'twenty twenty six'."""
        if year < 100:
            return _number_to_words(year)
        if year < 1000:
            return _number_to_words(year)
        # 2000 → "two thousand", 2001 → "two thousand one"
        # 2010+ → split into century + remainder
        if 2000 <= year <= 2009:
            if year == 2000:
                return "two thousand"
            return "two thousand " + _number_to_words(year - 2000)
        # Other centuries: e.g. 1950 → "nineteen fifty"
        upper = year // 100
        lower = year % 100
        century_word = _number_to_words(upper * 100)
        if lower == 0:
            return century_word
        if lower < 10:
            return century_word + " oh " + _number_to_words(lower)
        return century_word + " " + _number_to_words(lower)

    # --- Ordinals ---

    _ORDINAL_RE = re.compile(
        r"\b(\d+)(st|nd|rd|th)\b",
        re.IGNORECASE,
    )

    def _normalize_ordinals(self, text: str) -> str:
        """Normalize ordinal numbers to words: 1st → first, 2nd → second."""
        def _ordinal_replace(m: re.Match) -> str:
            n = int(m.group(1))
            return _ordinal_to_words(n)

        text = self._ORDINAL_RE.sub(_ordinal_replace, text)
        return text

    # --- Abbreviations ---

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        # Case-insensitive expansion
        for abbr, expansion in _ABBREVIATIONS.items():
            text = re.sub(re.escape(abbr), expansion, text, flags=re.IGNORECASE)
        return text