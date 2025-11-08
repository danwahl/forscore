#!/usr/bin/env python3
"""
Unit tests for subset sum generation using independent solving logic.

This test module verifies the correctness of subset sum generation by:
1. Verifying super-increasing property of sequences
2. Independently solving subset sum using greedy algorithm
3. Comparing results with the generated solutions
"""

import unittest
import random
from typing import List, Optional

from generate import (
    generate_super_increasing_sequence,
    select_random_subset,
    format_problem,
    generate_dataset,
)


class IndependentSubsetSumSolver:
    """
    Independent greedy solver for super-increasing subset sum.

    For super-increasing sequences (where each element > sum of all previous),
    there is exactly one solution, which can be found using a greedy algorithm
    working backwards from the largest element.
    """

    def __init__(self, numbers: List[int]):
        self.numbers = sorted(numbers)  # Sort for greedy algorithm

    def solve(self, target: int) -> Optional[List[int]]:
        """
        Find the unique subset that sums to target using greedy algorithm.

        Args:
            target: Target sum to achieve

        Returns:
            List of numbers that sum to target, or None if no solution exists
        """
        if target == 0:
            return []

        if target < 0:
            return None

        subset = []
        remaining = target

        # Work backwards from largest element
        for num in reversed(self.numbers):
            if num <= remaining:
                subset.append(num)
                remaining -= num

                if remaining == 0:
                    return sorted(subset)

        # No solution found
        return None if remaining != 0 else sorted(subset)


class TestSubsetSumGeneration(unittest.TestCase):
    """Test subset sum generation with independent verification."""

    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)

    def test_super_increasing_property(self):
        """Test that generated sequences are super-increasing."""
        for trial in range(20):
            with self.subTest(trial=trial):
                length = random.randint(5, 15)
                sequence = generate_super_increasing_sequence(length)

                # Verify super-increasing property
                for i in range(1, len(sequence)):
                    sum_before = sum(sequence[:i])
                    current = sequence[i]
                    self.assertGreater(
                        current,
                        sum_before,
                        f"Element at index {i} ({current}) should be > sum of previous ({sum_before})\n"
                        f"Sequence: {sequence}"
                    )

    def test_sequence_length(self):
        """Test that generated sequences have correct length."""
        for length in [3, 5, 10, 15, 20]:
            sequence = generate_super_increasing_sequence(length)
            self.assertEqual(len(sequence), length)

    def test_subset_from_sequence(self):
        """Test that selected subsets only contain elements from the sequence."""
        for trial in range(20):
            with self.subTest(trial=trial):
                sequence = generate_super_increasing_sequence(10)
                subset = select_random_subset(sequence, subset_size_range=(2, 8))

                # All elements in subset should be from sequence
                for element in subset:
                    self.assertIn(
                        element,
                        sequence,
                        f"Element {element} not found in sequence {sequence}"
                    )

    def test_subset_no_duplicates(self):
        """Test that subsets don't contain duplicates."""
        for trial in range(20):
            with self.subTest(trial=trial):
                sequence = generate_super_increasing_sequence(10)
                subset = select_random_subset(sequence, subset_size_range=(2, 8))

                # No duplicates
                self.assertEqual(
                    len(subset),
                    len(set(subset)),
                    f"Subset contains duplicates: {subset}"
                )

    def test_subset_size_range(self):
        """Test that subset sizes are within specified range."""
        sequence = generate_super_increasing_sequence(15)

        for _ in range(20):
            subset = select_random_subset(sequence, subset_size_range=(3, 10))
            self.assertGreaterEqual(len(subset), 3)
            self.assertLessEqual(len(subset), 10)

    def test_greedy_solver_basic(self):
        """Test independent greedy solver on known cases."""
        # Test case 1: Simple super-increasing sequence
        numbers = [1, 2, 4, 8, 16]
        solver = IndependentSubsetSumSolver(numbers)

        # 7 = 1 + 2 + 4
        solution = solver.solve(7)
        self.assertIsNotNone(solution)
        self.assertEqual(sum(solution), 7)
        self.assertEqual(sorted(solution), [1, 2, 4])

        # 11 = 1 + 2 + 8
        solution = solver.solve(11)
        self.assertIsNotNone(solution)
        self.assertEqual(sum(solution), 11)
        self.assertEqual(sorted(solution), [1, 2, 8])

        # No solution for impossible target
        solution = solver.solve(32)
        self.assertIsNone(solution)

    def test_greedy_solver_consistency(self):
        """Test that greedy solver finds correct solutions for generated sequences."""
        for trial in range(30):
            with self.subTest(trial=trial):
                # Generate super-increasing sequence
                sequence = generate_super_increasing_sequence(
                    length=random.randint(5, 12)
                )

                # Select random subset as target
                subset = select_random_subset(sequence, subset_size_range=(2, 8))
                target = sum(subset)

                # Solve independently
                solver = IndependentSubsetSumSolver(sequence)
                solution = solver.solve(target)

                # Solution should exist and be correct
                self.assertIsNotNone(
                    solution,
                    f"No solution found for target {target} from sequence {sequence}"
                )
                self.assertEqual(
                    sum(solution),
                    target,
                    f"Solution sum ({sum(solution)}) != target ({target})"
                )

                # For super-increasing, solution should be unique and match our subset
                self.assertEqual(
                    sorted(solution),
                    sorted(subset),
                    f"Solution {solution} doesn't match expected subset {subset}\n"
                    f"Sequence: {sequence}, Target: {target}"
                )

    def test_format_problem_structure(self):
        """Test that problem formatting contains all necessary information."""
        numbers = [3, 7, 12, 25, 45]
        target = 82

        problem = format_problem(numbers, target)

        # Check key components
        self.assertIn(str(numbers), problem)
        self.assertIn(str(target), problem)
        self.assertIn("subset", problem.lower())
        self.assertIn("sum", problem.lower())

    def test_dataset_generation_count(self):
        """Test that dataset generation creates correct number of examples."""
        dataset = generate_dataset(
            num_examples=50,
            sequence_length_range=(5, 10),
            subset_size_range=(2, 5),
            start_range=(1, 10),
            growth_range=(1, 20),
            seed=123
        )

        self.assertEqual(len(dataset), 50)

    def test_dataset_example_structure(self):
        """Test that dataset examples have correct structure."""
        dataset = generate_dataset(
            num_examples=10,
            sequence_length_range=(5, 10),
            subset_size_range=(2, 5),
            start_range=(1, 10),
            growth_range=(1, 20),
            seed=456
        )

        example = dataset[0]

        # Check required fields
        self.assertIn("problem", example)
        self.assertIn("numbers", example)
        self.assertIn("target", example)
        self.assertIn("solution", example)
        self.assertIn("sequence_length", example)
        self.assertIn("subset_size", example)

        # Verify types
        self.assertIsInstance(example["problem"], str)
        self.assertIsInstance(example["numbers"], list)
        self.assertIsInstance(example["target"], int)
        self.assertIsInstance(example["solution"], list)

    def test_dataset_correctness(self):
        """Test that all examples in dataset are correct."""
        dataset = generate_dataset(
            num_examples=20,
            sequence_length_range=(5, 10),
            subset_size_range=(2, 7),
            start_range=(1, 10),
            growth_range=(1, 20),
            seed=789
        )

        for i, example in enumerate(dataset):
            with self.subTest(example=i):
                numbers = example["numbers"]
                target = example["target"]
                solution = example["solution"]

                # Verify super-increasing property
                for j in range(1, len(numbers)):
                    self.assertGreater(numbers[j], sum(numbers[:j]))

                # Verify solution is valid
                self.assertEqual(sum(solution), target)

                # All solution elements should be from numbers
                for element in solution:
                    self.assertIn(element, numbers)

                # Verify using independent solver
                solver = IndependentSubsetSumSolver(numbers)
                independent_solution = solver.solve(target)

                self.assertIsNotNone(independent_solution)
                self.assertEqual(
                    sorted(independent_solution),
                    sorted(solution),
                    f"Independent solver found different solution\n"
                    f"Numbers: {numbers}\n"
                    f"Target: {target}\n"
                    f"Generated: {solution}\n"
                    f"Independent: {independent_solution}"
                )

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical datasets."""
        dataset1 = generate_dataset(
            num_examples=10,
            sequence_length_range=(5, 10),
            subset_size_range=(2, 5),
            start_range=(1, 10),
            growth_range=(1, 20),
            seed=999
        )

        dataset2 = generate_dataset(
            num_examples=10,
            sequence_length_range=(5, 10),
            subset_size_range=(2, 5),
            start_range=(1, 10),
            growth_range=(1, 20),
            seed=999
        )

        # Compare all examples
        for i in range(len(dataset1)):
            self.assertEqual(
                dataset1[i]["numbers"],
                dataset2[i]["numbers"],
                f"Numbers differ at index {i}"
            )
            self.assertEqual(
                dataset1[i]["target"],
                dataset2[i]["target"],
                f"Targets differ at index {i}"
            )
            self.assertEqual(
                dataset1[i]["solution"],
                dataset2[i]["solution"],
                f"Solutions differ at index {i}"
            )

    def test_various_difficulty_levels(self):
        """Test generation with various difficulty parameters."""
        configs = [
            # Easy: small sequences, slow growth
            {
                'sequence_length_range': (3, 5),
                'subset_size_range': (2, 3),
                'growth_range': (1, 5)
            },
            # Medium: moderate sequences
            {
                'sequence_length_range': (7, 10),
                'subset_size_range': (3, 6),
                'growth_range': (5, 15)
            },
            # Hard: large sequences, fast growth
            {
                'sequence_length_range': (12, 20),
                'subset_size_range': (5, 15),
                'growth_range': (10, 50)
            },
        ]

        for config in configs:
            with self.subTest(config=config):
                dataset = generate_dataset(
                    num_examples=5,
                    sequence_length_range=config['sequence_length_range'],
                    subset_size_range=config['subset_size_range'],
                    start_range=(1, 10),
                    growth_range=config['growth_range'],
                    seed=random.randint(1, 10000)
                )

                # Verify all examples are valid
                for example in dataset:
                    # Check super-increasing
                    numbers = example["numbers"]
                    for i in range(1, len(numbers)):
                        self.assertGreater(numbers[i], sum(numbers[:i]))

                    # Verify solution
                    solver = IndependentSubsetSumSolver(numbers)
                    solution = solver.solve(example["target"])
                    self.assertIsNotNone(solution)
                    self.assertEqual(sorted(solution), sorted(example["solution"]))

    def test_edge_case_minimum_sequence(self):
        """Test edge case with minimum sequence length."""
        sequence = generate_super_increasing_sequence(2)
        self.assertEqual(len(sequence), 2)
        self.assertGreater(sequence[1], sequence[0])

    def test_edge_case_subset_leaves_elements(self):
        """Test that subset selection leaves at least one element out."""
        sequence = generate_super_increasing_sequence(10)

        # Max subset size should be len-1 to leave at least one element
        for _ in range(10):
            subset = select_random_subset(sequence, subset_size_range=(2, 15))
            self.assertLess(
                len(subset),
                len(sequence),
                "Subset should not include all elements"
            )

    def test_large_numbers(self):
        """Test that sequences with large growth factors work correctly."""
        sequence = generate_super_increasing_sequence(
            length=10,
            start_range=(100, 200),
            growth_range=(50, 100)
        )

        # Verify super-increasing with larger numbers
        for i in range(1, len(sequence)):
            self.assertGreater(sequence[i], sum(sequence[:i]))

        # Test solver with large numbers
        subset = select_random_subset(sequence, subset_size_range=(3, 7))
        target = sum(subset)

        solver = IndependentSubsetSumSolver(sequence)
        solution = solver.solve(target)

        self.assertIsNotNone(solution)
        self.assertEqual(sorted(solution), sorted(subset))


if __name__ == '__main__':
    unittest.main(verbosity=2)
