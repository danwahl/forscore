import xml.etree.ElementTree as ET
import math
import re

PAD_TOKEN = "<|fim_pad|>"


def parse_full_response(text: str) -> dict:
    """Parse the full XML response including think and answer blocks."""
    try:
        # Remove any leading/trailing whitespace and padding tokens
        text = text.replace(PAD_TOKEN, "").strip()

        # Wrap in root element since XML needs single root
        wrapped = f"<root>{text}</root>"
        root = ET.fromstring(wrapped)

        think_elem = root.find("think")
        answer_elem = root.find("answer")

        result = {
            "think": think_elem.text if think_elem is not None else None,
            "answer": answer_elem.text.strip() if answer_elem is not None and answer_elem.text and answer_elem.text.strip() else None,
        }

        return result
    except:
        return None


def parse_subset_from_answer(answer_text: str) -> list[int]:
    """
    Parse subset from answer text.

    Accepts multiple formats:
    - List format: [3, 7, 12] or [3,7,12]
    - Comma-separated: 3, 7, 12 or 3,7,12
    - Space-separated: 3 7 12

    Returns:
        List of integers or None if invalid
    """
    if not answer_text:
        return None

    # Remove whitespace
    text = answer_text.strip()

    # Try list format first - remove brackets
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    # Try to extract all numbers using regex
    try:
        # Find all integers in the string
        numbers = [int(x) for x in re.findall(r'-?\d+', text)]
        return numbers if numbers else None
    except (ValueError, AttributeError):
        return None


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for proper XML format with think and answer blocks."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response in responses:
        parsed = parse_full_response(response)
        if parsed is None:
            results.append(0.0)
        else:
            try:
                # Check if we have both think and answer
                has_think = parsed["think"] is not None and len(parsed["think"].strip()) > 0
                has_answer = parsed["answer"] is not None
                results.append(1.0 if has_think and has_answer else 0.0)
            except:
                results.append(0.0)

    return results


def subset_correct_reward_func(
    prompts, completions, target, numbers, **kwargs
) -> list[float]:
    """
    Reward for finding a valid subset that sums to target.

    Checks:
    1. All elements in answer are from the original list
    2. Sum of answer equals target
    3. No duplicates in answer (each element used at most once)

    This works regardless of whether the solution is unique.
    """
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, target_sum, number_list in zip(responses, target, numbers):
        parsed = parse_full_response(response)
        if parsed is None or parsed["answer"] is None:
            results.append(0.0)
        else:
            predicted_subset = parse_subset_from_answer(parsed["answer"])
            if predicted_subset is None or len(predicted_subset) == 0:
                results.append(0.0)
            else:
                try:
                    # Check validity:
                    # 1. All numbers in predicted subset are from the original list
                    # 2. No duplicates (each number used at most once)
                    # 3. Sum equals target

                    # Count occurrences in both lists
                    from collections import Counter
                    predicted_counts = Counter(predicted_subset)
                    available_counts = Counter(number_list)

                    # Check all predicted numbers are available
                    all_available = all(
                        predicted_counts[num] <= available_counts[num]
                        for num in predicted_counts
                    )

                    # Check sum
                    correct_sum = sum(predicted_subset) == target_sum

                    is_valid = all_available and correct_sum
                    results.append(1.0 if is_valid else 0.0)
                except Exception:
                    results.append(0.0)

    return results


def token_count_reward_func(
    prompts, completion_ids, target_count=512, decay_rate=256.0, **kwargs
) -> list[float]:
    """Penalty for responses that exceed target length using exponential decay."""
    results = []

    for tokens in completion_ids:
        count = len(tokens)

        # No penalty if within target count
        if count <= target_count:
            results.append(1.0)
        else:
            # Exponential decay penalty for exceeding target
            excess = count - target_count
            penalty = math.exp(-excess / decay_rate)
            results.append(penalty)

    return results
