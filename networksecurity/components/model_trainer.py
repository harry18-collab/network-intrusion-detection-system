import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)#type:ignore

    # --------- NO MLFLOW / DAGSHUB TRACKING NOW ----------
    # def track_mlflow(self, best_model, classificationmetric):
    #     pass 

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 128, 256],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params,
        )
        
        # Print all model scores for comparison
        print("\n📄 MODEL COMPARISON REPORT:")
        print("-" * 40)
        for model_name, score in model_report.items():
            print(f"{model_name:<20}: {score:.4f} ({score*100:.2f}%)")
        print("-" * 40)

        # best model score and name
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        # train metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(
            y_true=y_train, y_pred=y_train_pred
        )

        # test metrics
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(
            y_true=y_test, y_pred=y_test_pred
        )

        # preprocessor load
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # model save
        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        # yahan pe pehle galti thi: class ka naam pass kar rahe the, object ki jagah
        save_object(
            self.model_trainer_config.trained_model_file_path, obj=network_model
        )

        # model pusher ke liye plain model bhi save kar dete hain
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        # Print model performance metrics
        print("\n" + "="*60)
        print(f" BEST MODEL: {best_model_name}")
        print(f" BEST MODEL SCORE: {best_model_score:.4f}")
        print("="*60)
        
        print("\n📈 TRAINING METRICS:")
        print(f"   F1-Score:  {classification_train_metric.f1_score:.4f} ({classification_train_metric.f1_score*100:.2f}%)")
        print(f"   Precision: {classification_train_metric.precision_score:.4f} ({classification_train_metric.precision_score*100:.2f}%)")
        print(f"   Recall:    {classification_train_metric.recall_score:.4f} ({classification_train_metric.recall_score*100:.2f}%)")
        
        print("\n TESTING METRICS:")
        print(f"   F1-Score:  {classification_test_metric.f1_score:.4f} ({classification_test_metric.f1_score*100:.2f}%)")
        print(f"   Precision: {classification_test_metric.precision_score:.4f} ({classification_test_metric.precision_score*100:.2f}%)")
        print(f"   Recall:    {classification_test_metric.recall_score:.4f} ({classification_test_metric.recall_score*100:.2f}%)")
        print("="*60 + "\n")

        # artifact return
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(
                x_train, y_train, x_test, y_test
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)#type:ignore
