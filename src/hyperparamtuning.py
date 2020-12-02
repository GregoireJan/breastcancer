import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple, AnyStr
from sklearn import datasets, metrics,  model_selection
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import mlflow
import mlflow.sklearn
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt import Trials, hp

from modeler import Modeler
import click



def regression_metrics(actual: pd.Series,
                       pred: pd.Series) -> Dict:
    """Return a collection of regression metrics as a Series.

    Args:
        actual: series of actual/true values
        pred: series of predicted values

    Returns:
        Series with the following values in a labeled index:
        MAE, RMSE
    """
    return {
        "ACCURACY": accuracy_score(actual,pred),
        "F1": metrics.f1_score(actual,pred)}

def fit_and_log_cv(model,
                   x_train: Union[pd.DataFrame, np.array],
                   y_train: Union[pd.Series, np.array],
                   x_test: Union[pd.DataFrame, np.array],
                   y_test: Union[pd.Series, np.array],
                   params: Dict[str, Any],
                   nested: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Fit a model and log it along with train/CV metrics.
  
  Args:
      x_train: feature matrix for training/CV data
      y_train: label array for training/CV data
      x_test: feature matrix for test data
      y_test: label array for test data
      nested: if true, mlflow run will be started as child
          of existing parent
  """
  with mlflow.start_run(nested=nested) as run:
    # Fit CV models; extract predictions and metrics
    print(type(params))
    print(params)
    model_cv = model(**params)
    y_pred_cv = model_selection.cross_val_predict(model_cv, x_train, y_train)
    metrics_cv = {
      f"val_{metric}": value
      for metric, value in regression_metrics(y_train, y_pred_cv).items()}

    # Fit and log full training sample model; extract predictions and metrics
    mlflow.sklearn.autolog()
    model = model(**params)
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    metrics_test = {
      f"test_{metric}": value
      for metric, value in regression_metrics(y_test, y_pred_test).items()}
    
    metrics = {**metrics_test, **metrics_cv}
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    return metrics

def build_train_objective(model,
                          x_train: Union[pd.DataFrame, np.array],
                          y_train: Union[pd.Series, np.array],
                          x_test: Union[pd.DataFrame, np.array],
                          y_test: Union[pd.Series, np.array],
                          metric: str):
        """Build optimization objective function fits and evaluates model.

        Args:
        x_train: feature matrix for training/CV data
        y_train: label array for training/CV data
        x_test: feature matrix for test data
        y_test: label array for test data
        metric: name of metric to be optimized
        
        Returns:
            Optimization function set up to take parameter dict from Hyperopt.
        """

        def train_func(params):
            """Train a model and return loss metric."""
            metrics = fit_and_log_cv(model,
            x_train, y_train, x_test, y_test, params, nested=True)
            return {'status': hyperopt.STATUS_OK, 'loss': -metrics[metric]}

        return train_func

def log_best(run: mlflow.entities.Run,
             metric: str) -> None:
    """Log the best parameters from optimization to the parent experiment.

    Args:
        run: current run to log metrics
        metric: name of metric to select best and log
    """

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))
    best_run = min(runs, key=lambda run: -run.data.metrics[metric])

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])

##############################################################################

@click.command()
@click.option('--name', type=str, default='')
@click.option('--maxeval', type=int, default=10)
@click.option('--metric', type=str, default='val_F1')
def main(name,maxeval,metric):
    """Triggers experiment looping through ML algorithms

    Args:
        name: name of experiment
        maxeval: maximum number of evaluation
        metric: name of metric to minimize cost function
    """
    mlflow.set_experiment(name)
    MAX_EVALS = maxeval
    METRIC = metric

    space = [{
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,26)),
        'n_estimators': hp.choice('n_estimators', range(100,500)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])},
        {'var_smoothing':hp.uniform('var_smoothing', 0.000000001,0.000001)}]

    X_train, X_test, y_train, y_test = Modeler().prepro()

    for index, algo in enumerate([RandomForestClassifier,GaussianNB]):
        with mlflow.start_run(run_name=str(algo)) as run:
            trials = Trials()
            train_objective = build_train_objective(algo,X_train, y_train, X_test, y_test, METRIC)
            hyperopt.fmin(fn=train_objective,
                            space=space[index],
                            algo=hyperopt.tpe.suggest,
                            max_evals=MAX_EVALS,
                            trials=trials)
            log_best(run, METRIC)
            # search_run_id = run.info.run_id
            # experiment_id = run.info.experiment_id
            mlflow.end_run()


if __name__ == '__main__':
    main()