# Twitter Sentiment Analysis

A sentiment categorization system for tweets is designed using classical machine learning algorithms (no deep learning). The dataset comprises of 1.6M tweets (available [here](https://www.kaggle.com/kazanova/sentiment140)) automatically labeled, and thus, noisy. This is part of [Natural Language Processing](https://www.cse.iitd.ac.in/~mausam/courses/col772/autumn2021/) course taken by [Prof Mausam](https://www.cse.iitd.ac.in/~mausam/).

## Running Mode

Predictions

```bash
python run_assignment1.py --input_path <path_to_input> --solution_path <path_to_solution>
```

Testing

```bash
python run_checker.py --ground_truth_path <path_to_ground_truth> --solution_path <path_to_solution>
```