# MLOps Tech Test

For the tasks in this test, a simple prediction model has been created using XGBoost.
Few caveats:
- The EDA was not deep-dived into.
- It is likely that with better feature engineering the accuracy could be improved.
- Performance metrics were logged at the end only.
- Since the link for the dataset was public, it was used in 'https://' format for simplicity
- Since the predictions are not idempotent, no tests were written checking or verifying the prediction results.
- The path was altered in the tasks to facilitate import of user-defined scripts
- Poetry was used as the dependency manager.

### To run the tasks:
##### Task1
```shell
python3 tasks/task_1.py
```
##### Task2
```shell
python3 tasks/task_2.py
```

### To run the tests:
##### Task1
```shell
python3 -m pytest -vv
```