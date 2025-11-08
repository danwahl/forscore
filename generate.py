#!/usr/bin/env python3
"""
Generate subset sum datasets for training language models.

This script generates super-increasing sequences with random subsets that
sum to a target value. Super-increasing sequences guarantee unique solutions.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple
from datasets import Dataset


def generate_super_increasing_sequence(
    length: int, start_range: Tuple[int, int] = (1, 10), growth_range: Tuple[int, int] = (1, 20)
) -> List[int]:
    """
    Generate a super-increasing sequence.

    A sequence is super-increasing if each element is greater than the sum of
    all previous elements. This property guarantees a unique solution for any
    achievable target sum.

    Args:
        length: Number of elements in the sequence
        start_range: Range for the first element
        growth_range: Range for growth beyond minimum value

    Returns:
        Super-increasing sequence as a list of integers
    """
    sequence = [random.randint(*start_range)]

    for _ in range(1, length):
        # Each element must be > sum of all previous elements
        min_val = sum(sequence) + 1
        max_val = min_val + random.randint(*growth_range)
        sequence.append(random.randint(min_val, max_val))

    return sequence


def select_random_subset(
    sequence: List[int], subset_size_range: Tuple[int, int]
) -> List[int]:
    """
    Select a random subset from the sequence.

    Args:
        sequence: The super-increasing sequence
        subset_size_range: (min, max) size of subset to select

    Returns:
        Random subset as a list of integers
    """
    min_size, max_size = subset_size_range
    # Ensure we don't try to select more elements than available
    # and leave at least 1 element out
    max_size = min(max_size, len(sequence) - 1)
    max_size = max(max_size, min_size)  # Ensure max >= min

    subset_size = random.randint(min_size, min(max_size, len(sequence)))
    indices = sorted(random.sample(range(len(sequence)), subset_size))
    return [sequence[i] for i in indices]


def format_problem(numbers: List[int], target: int) -> str:
    """
    Format the subset sum problem as a string.

    Args:
        numbers: The list of available numbers
        target: The target sum to find

    Returns:
        Formatted problem string
    """
    return f"""Given the list: {numbers}

Find a subset that sums to {target}

What subset of numbers sums to exactly {target}?"""


def generate_dataset(
    num_examples: int,
    sequence_length_range: Tuple[int, int],
    subset_size_range: Tuple[int, int],
    start_range: Tuple[int, int],
    growth_range: Tuple[int, int],
    seed: int = 42
) -> Dataset:
    """
    Generate a dataset of subset sum problems.

    Args:
        num_examples: Number of examples to generate
        sequence_length_range: (min, max) length of super-increasing sequences
        subset_size_range: (min, max) size of subsets to select
        start_range: Range for the first element of sequences
        growth_range: Range for growth beyond minimum value
        seed: Random seed for reproducibility

    Returns:
        HuggingFace Dataset
    """
    random.seed(seed)
    examples = []

    for i in range(num_examples):
        # Generate super-increasing sequence
        seq_length = random.randint(*sequence_length_range)
        numbers = generate_super_increasing_sequence(seq_length, start_range, growth_range)

        # Select random subset as solution
        solution = select_random_subset(numbers, subset_size_range)
        target = sum(solution)

        # Format problem
        problem = format_problem(numbers, target)

        # Create example
        example = {
            "problem": problem,
            "numbers": numbers,
            "target": target,
            "solution": solution,  # The subset that sums to target
            "sequence_length": seq_length,
            "subset_size": len(solution),
        }

        examples.append(example)

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_examples} examples...")

    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(
        description="Generate subset sum datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of examples to generate"
    )

    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=5,
        help="Minimum sequence length"
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=15,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--min-subset-size",
        type=int,
        default=2,
        help="Minimum subset size"
    )

    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=10,
        help="Maximum subset size"
    )

    parser.add_argument(
        "--start-range-min",
        type=int,
        default=1,
        help="Minimum value for first element"
    )

    parser.add_argument(
        "--start-range-max",
        type=int,
        default=10,
        help="Maximum value for first element"
    )

    parser.add_argument(
        "--growth-range-min",
        type=int,
        default=1,
        help="Minimum growth beyond required minimum"
    )

    parser.add_argument(
        "--growth-range-max",
        type=int,
        default=20,
        help="Maximum growth beyond required minimum"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for datasets"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (auto-generated if not specified)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Generate dataset name if not provided
    if args.dataset_name is None:
        dataset_name = f"subset_sum_l{args.min_sequence_length}-{args.max_sequence_length}_s{args.min_subset_size}-{args.max_subset_size}_n{args.num_examples}"
    else:
        dataset_name = args.dataset_name

    print(f"Generating dataset: {dataset_name}")
    print(f"  Examples: {args.num_examples}")
    print(f"  Sequence length: {args.min_sequence_length}-{args.max_sequence_length}")
    print(f"  Subset size: {args.min_subset_size}-{args.max_subset_size}")
    print(f"  Start range: {args.start_range_min}-{args.start_range_max}")
    print(f"  Growth range: {args.growth_range_min}-{args.growth_range_max}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate dataset
    dataset = generate_dataset(
        num_examples=args.num_examples,
        sequence_length_range=(args.min_sequence_length, args.max_sequence_length),
        subset_size_range=(args.min_subset_size, args.max_subset_size),
        start_range=(args.start_range_min, args.start_range_max),
        growth_range=(args.growth_range_min, args.growth_range_max),
        seed=args.seed
    )

    # Save dataset
    output_path = Path(args.output_dir) / dataset_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))

    # Print sample
    print("\n" + "="*80)
    print("Sample example:")
    print("="*80)
    sample = dataset[0]
    print(sample["problem"])
    print(f"\nSolution: {sample['solution']}")
    print(f"Target sum: {sample['target']}")
    print(f"Verification: sum({sample['solution']}) = {sum(sample['solution'])}")
    print("="*80)

    print(f"\n✓ Dataset saved successfully!")
    print(f"  Location: {output_path}")
    print(f"  Total examples: {len(dataset)}")


if __name__ == "__main__":
    main()
