import pandas as pd
from amelio_cp.optimisation.optimisation_methods import OptimisationMethods
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import SMOTE


# %% Base Model for classification
class ClassifierModel:
    def __init__(self):

        self.name = None  # can name the model to call it then (i.e., SVRModel("Model A")),
        # or can only initiate then such as model_A = SVRModel()
        self.model = None  # will store the best model, should be updated each time
        self.scaler = StandardScaler()  # scaler used in data scaling
        self.X_train = (
            None  # features of training dataset, start with nothing, but will be completed
            # each time w/ a new sample
        )
        self.X_train_scaled = None  # scaled features of training dataset
        self.y_train = None  # labels of training dataset, IDEM
        self.X_test = None  # features of testing dataset
        self.X_test_scaled = None  # scaled features of testing dataset
        self.y_test = None  # labels of testing dataset
        self.params_distributions = {  # default param distributions, can be updated in child class
            "C": [1, 1000],
            "gamma": [0.001, 0.1],
            "degree": [2, 5],
            "kernel": ["linear", "poly", "rbf"],
        }
        self.optim_method = None
        self.primary_scoring = "accuracy"
        self.secondary_scoring = "f1"
        self.best_params = (
            None  # stores the best parameters, and updates it everytime the addition of a sample allows better results
        )
        self.shap_analysis = None  # stores the shap analysis objects, if needed

        self.dist_from_bound = None  # distance from decision boundary in classification models using decision_function

        # Random states of the model
        self.random_state = 42  # setting a default rdm state
        self.random_state_split = self.random_state  # sets a random state for data split
        self.random_state_optim = self.random_state  # sets a random state for the optimisation
        self.random_state_cv = self.random_state  # sets a random state for the CV

    # Functions for setting model's random states
    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value
        # Always propagate main random_state to the specific ones; they can still
        # be overridden individually afterward.
        self._random_state_split = value
        self._random_state_optim = value
        self._random_state_cv = value
        print("All random_state have been changed!")

    @property
    def random_state_split(self):
        return getattr(self, "_random_state_split", self.random_state)

    @random_state_split.setter
    def random_state_split(self, value):
        self._random_state_split = value

    @property
    def random_state_cv(self):
        return getattr(self, "_random_state_cv", self.random_state)

    @random_state_cv.setter
    def random_state_cv(self, value):
        self._random_state_cv = value

    @property
    def random_state_optim(self):
        return getattr(self, "_random_state_optim", self.random_state)

    @random_state_optim.setter
    def random_state_optim(self, value):
        self._random_state_optim = value

    # Specific function to add the training data
    def add_train_data(self, X, y):
        """Function that will add new samples to the training set."""
        self.X_train, self.y_train = self._add_template(X, y, self.X_train, self.y_train)

    # Specific function to add the testing data
    def add_test_data(self, X, y):
        """Function that will add new samples to the training set."""
        self.X_test, self.y_test = self._add_template(X, y, self.X_test, self.y_test)

    def rescale(self):
        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)

    # Function that splits and adds datasets
    def add_data(self, X, y, test_size, enabling_smote=False):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state_split
        )
        print("Split has been done.", flush=True)

        if enabling_smote is True:
            # check if training data set is imbalance, if so, use SMOTE to balance it
            # IR = Imbalance Ratio
            IR = max(y_train.value_counts()[0], y_train.value_counts()[1]) / min(
                y_train.value_counts()[0], y_train.value_counts()[1]
            )

            if IR > 2:
                smote = SMOTE(sampling_strategy="auto", random_state=self.random_state)
                x_train, y_train = smote.fit_resample(x_train, y_train)

        self.add_train_data(x_train, y_train)
        self.add_test_data(x_test, y_test)
        self.rescale()  # rescaling the data after adding it

    # Function that handles and correctly stores the data
    @staticmethod
    def _add_template(X_given, y_given, X_model, y_model):
        X_given = pd.DataFrame(X_given)  # pandas conversion
        y_given = pd.Series(y_given)

        if X_model is None:  # if nothing, will just take it
            X_model = X_given
            y_model = y_given
        else:  # if already with something in, will append the new sample
            X_model = pd.concat([X_model, X_given], ignore_index=True)
            y_model = pd.concat([y_model, y_given], ignore_index=True)
        return X_model, y_model

    # Function that optimises and trains the model
    def train_and_tune(self, method: str, n_iter=100):
        """Tune hyperparameters with choosen method and fit the model."""
        if self.X_train_scaled is None or self.y_train is None:  # Check if there is some data
            raise ValueError("No data available for training.")

        # Creating the optimisation loop
        if method == "random":
            search = OptimisationMethods.random_search(model=self, n_iter=n_iter, k_folds=5)
            search.fit(self.X_train_scaled, self.y_train)  # training
            print("Random search optimisation completed.")

        elif method == "bayesian_search":
            search = OptimisationMethods.bayesian_search(model=self, n_iter=n_iter, k_folds=5)
            search.fit(self.X_train_scaled, self.y_train)  # training
            print("Bayesian Search optimisation completed.")

        elif method == "bayesian_optim":
            search = OptimisationMethods.bayesian_optim(self, n_iter=n_iter)
            print("Bayesian optimisation completed.")

        else:
            raise ValueError("Unknown optimisation method. Choose 'random', 'bayesian' or 'bayesian_optim'.")

        self.model = search.best_estimator_  # recover the best model
        self.best_params = search.best_params_  # recover the best hp

        # Evaluate
        preds = self.model.predict(self.X_train_scaled)  # quick check to see if model OK (no overfitting)
        acc = accuracy_score(self.y_train, preds)  # IDEM
        print(f"Best Params: {self.best_params}")
        print(f"Accuracy on training data: {acc:.4f}")

        # TODO: adding ROC-AUC value

        # Evaluate with K-Fold CV for stability
        # K-Fold CV setup
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_acc = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, cv=cv_splitter, scoring=self.primary_scoring
        )
        print(f"📊 CV accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

        return {
            "Accuracy": acc,
            "CV accuracy": cv_acc.mean(),
        }

    def test_model(self):
        y_pred = self.model.predict(self.X_test_scaled)
        score = self.model.score(self.X_test_scaled, self.y_test)  # returns accuracy
        print(f"Model's score on testing data: {score:.4f}")
        return y_pred, score

    def save(self, path):
        """Save model and training data."""
        joblib.dump(
            {
                "name": self.name,
                "model": self.model,
                "X_train": self.X_train,
                "X_train_scaled": self.X_train_scaled,
                "y_train": self.y_train,
                "X_test": self.X_test,
                "X_test_scaled": self.X_test_scaled,
                "y_test": self.y_test,
                "best_params": self.best_params,
                "shap_analysis": self.shap_analysis,
            },
            path,
        )
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a saved model."""
        data = joblib.load(path)
        obj = cls()
        obj.name = data["name"]
        obj.model = data["model"]
        obj.X_train = data["X_train"]
        obj.X_train_scaled = data["X_train_scaled"]
        obj.y_train = data["y_train"]
        obj.X_test = data["X_test"]
        obj.X_test_scaled = data["X_test_scaled"]
        obj.y_test = data["y_test"]
        obj.best_params = data["best_params"]
        obj.shap_analysis = data["shap_analysis"]
        print(f"📂 Model loaded from {path}")
        return obj
