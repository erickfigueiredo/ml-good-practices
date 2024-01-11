import math


def print_progress_bar(step, total_steps):
    """
    Prints a progress bar to track the completion of a process.

    Args:
    - step (int): Current step in the process.
    - total_steps (int): Total number of steps in the process.

    Returns:
    - str: Formatted progress bar.
    """
    completion_percentage = (step + 1) / total_steps * 100
    formatted_completion_percentage = math.floor(completion_percentage / 3.333)

    return f'[{"=" * formatted_completion_percentage}{"." * (30 - formatted_completion_percentage)}]'
