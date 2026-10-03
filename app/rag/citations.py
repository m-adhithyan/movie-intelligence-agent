import re


SOURCE_REFERENCE_PATTERN = re.compile(r"\[(\d+)\]")


def extract_source_numbers(answer: str) -> list[int]:
    """
    Extract unique source numbers referenced in an answer.

    Example:
        "The artist discusses creation [1][3]."
        -> [1, 3]
    """
    if not answer:
        return []

    matches = SOURCE_REFERENCE_PATTERN.findall(answer)

    source_numbers = []

    for match in matches:
        number = int(match)

        if number not in source_numbers:
            source_numbers.append(number)

    return source_numbers


def validate_source_numbers(
    answer: str,
    source_count: int,
) -> tuple[list[int], list[int]]:
    """
    Return valid and invalid source references.

    Example:
        answer references [1][3][9]
        source_count = 5

        -> ([1, 3], [9])
    """
    if source_count < 0:
        raise ValueError("source_count cannot be negative.")

    source_numbers = extract_source_numbers(answer)

    valid = [
        number
        for number in source_numbers
        if 1 <= number <= source_count
    ]

    invalid = [
        number
        for number in source_numbers
        if number < 1 or number > source_count
    ]

    return valid, invalid