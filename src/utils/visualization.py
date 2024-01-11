import math


def print_progress_bar(step, total_steps):
    completion_percentage = (step + 1) / total_steps * 100
    formatted_completion_percentage = math.floor(completion_percentage / 3.333)

    return f'[{"="*(formatted_completion_percentage)}{"."*(30 - formatted_completion_percentage)}]'
