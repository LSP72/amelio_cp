import os
import pandas as pd
from pandas import DataFrame
import amelio_cp.processing.old_featurExtractor as mat_to_df


class Process:
    def __init__(self):
        pass

    def load_csv(file_path: str) -> DataFrame:
        df = pd.read_csv(file_path, index_col=False)
        return df

    # ─────────────────────────────────────────
    #  DATAFRAME PREPARATION
    # ─────────────────────────────────────────
    @staticmethod
    def calculate_ROM(df: DataFrame):

        print(df.columns)

        df["ROM_Hip_Sag"] = df["Max_Hip_flx/ext"] - df["Min_Hip_flx/ext"]
        df["ROM_Hip_Frontal"] = df["Max_Hip_abd/add"] - df["Min_Hip_abd/add"]
        df["ROM_Hip_Trns"] = df["Max_Hip_int/ext rot"] - df["Min_Hip_int/ext rot"]
        df["ROM_Knee_Sag"] = df["Max_Knee_flx/ext"] - df["Min_Knee_flx/ext"]
        df["ROM_Ankle_Sag"] = df["Max_Ankle_flx/ext"] - df["Min_Ankle_flx/ext"]

        return df

    @staticmethod
    def load_gps_data(file_path: str) -> DataFrame:
        df = pd.read_csv(file_path, index_col=False)
        df.drop(columns=["Unnamed: 0", "Post"], inplace=True)
        df.rename(columns={"Pre": "GPS"}, inplace=True)
        return df

    @staticmethod
    def load_demographic_data(file_path: str):

        df = pd.read_excel(file_path)
        df["BMI"] = df["masse"] / ((df["taille"] / 100) ** 2)

        # The numbers are in the order so 1 is walker.
        df.replace(["walker", "cane", "none"], [1, 2, 3], inplace=True)

        # ----- Add Label -----
        df.drop(["delta6MWT", "deltaV"], axis=1, inplace=True)
        # This line excludes some of the features that are in the demographic excel file.
        df.drop(["Patient", "masse", "taille", "sex", "Diagnostique"], axis=1, inplace=True)

        return df

    def load_data(
        self, data_dir: str, output_dir: str, gps_path: str, demographic_path: str, separate_legs: bool = True
    ):
        """
        Load data from raw data

        Parameters
        ----------
        data_dir : str
            Dir to the folder with all .mat files.
        output_dir : str
            Dir to the output folder.
        demographic_path : str
            Path to demographic data Excel file.
        gps_path : str
            Path to gait analysis scores Excel file.
        separate_legs : bool, optional
            Condition used about the extraction,
            by default is True.

        Returns
        -------
        all_data : pd.DataFrame
            DataFrame with all the features and labels.
        """

        # file info
        measurements = ["pctSimpleAppuie", "distFoulee", "vitCadencePasParMinute"]
        joint_names = ["Hip", "Knee", "Ankle"]
        side = ["Right", "Left"]

        # Extracting variables from .mat files
        kinematic_variables = mat_to_df.min_max_feature_extractor(
            directory=data_dir,
            output_dir=output_dir,
            measurements=measurements,
            separated_legs=separate_legs,
            joint_names=joint_names,
        )

        # Calculating ROM from the function above
        kinematic_variables = self.calculate_ROM(kinematic_variables)

        # # ----- fixing the values of cadence -----
        # kinematic_variables["vitCadencePasParMinute"] *= 2

        # ----- Add GPS to the features -----
        #       Gait Profile Score
        gps = self.load_gps_data(gps_path)
        all_data = pd.concat((kinematic_variables, gps), axis=1)

        # ----- Add participants demographic variables -----
        demographic_variables = self.load_demographic_data(demographic_path)

        all_data = pd.concat((all_data, demographic_variables), axis=1)

        # ----- Calculating the value of Cadence -----
        all_data["vitCadencePasParMinute"] = 60 * all_data["VIT_PRE"]/all_data["distFoulee"] # VIT_PRE being in m/s

        # TODO: concat in one funct

        return all_data

    @staticmethod
    def MCID_VIT_GPS(pre_data, post_data, threshold) -> list:
        delta = post_data - pre_data
        MCID = []
        for i in delta:
            if i >= threshold:
                MCID.append(1)
            else:
                MCID.append(0)
        return pd.Series(MCID, index=pre_data.index)

    @staticmethod
    def calculate_MCID(pre_data, post_data, variable, gmfcs_data=None) -> list:
        if variable == "VIT":
            return Process.MCID_VIT_GPS(pre_data, post_data, 0.1)

        elif variable == "GPS":
            return Process.MCID_VIT_GPS(pre_data, post_data, 1.6)

        elif variable == "6MWT":
            GMFCS_MCID = {1: range(4, 29), 2: range(4, 29), 3: range(9, 20), 4: range(10, 28)}
            delta_6MWT = post_data - pre_data
            MCID_6MWT = []
            for i in range(len(delta_6MWT)):
                if delta_6MWT.iloc[i] >= max(GMFCS_MCID[gmfcs_data.iloc[i]]):
                    MCID_6MWT.append(1)
                else:
                    MCID_6MWT.append(0)
            return pd.Series(MCID_6MWT, index=gmfcs_data.index)

        else:
            raise ValueError("Variable not recognized. Use 'VIT', or '6MWT', or 'GPS'.")

    # Not used as the data are directly loaded from an xlsx files
    @staticmethod
    def _calculate_gsi(data: DataFrame, weight: DataFrame):
        """This function caluclates the Global Strength Index as such:
                    GSI = (1/n)*sum(muscle_torques/weight)

                    with n = number of muscles considered

        Parameters
        ----------
        data : DataFrame
            only the strength values to consider for the calculation of the gsi

        weight : Series
            values of weight to normalised the sum of strength

        Returns
        -------
        DataFrame
            with the GSI for each individual (i.e., row)
        """

        all_gsi = []
        for i in range(len(data)):
            muscle_strenghts = data.iloc[i]
            sum = muscle_strenghts.sum()
            gsi = sum / weight.iloc[i]
            all_gsi.append(gsi)

        all_gsi_df = pd.Series(all_gsi, index=data.index)

        return all_gsi_df

    @staticmethod
    def return_gsi(file_path: str, separate_legs: bool = True):
        """
        Parameters
        ----------
        file_path : str
            Path of the excel file with all the data
            Assuming that:
            Right values should be in the 3-to-7 columns
            Left values should be in the 8-to-13 columns

        separate_legs : bool, optional
            Enables to calculate the gsi for both legs (if False)
            or for each leg (if True),
                        by default True

        Returns
        -------
        DataFrame
            Returns a df with the gsi for each patient (i.e., row)

        Raises
        ------
        TypeError
            if neither True nor False was assigned to separate_legs
        """

        data = pd.read_excel(file_path)
        weight = data["weight"]

        if separate_legs == True:
            right_gsi = Process._calculate_gsi(data.iloc[:, 2:7], weight)
            left_gsi = Process._calculate_gsi(data.iloc[:, 7:], weight)

            return right_gsi, left_gsi

        elif separate_legs == False:
            gsi = Process._calculate_gsi(data.iloc[:, 2:], weight)
            return gsi

        else:
            raise TypeError("'separate_legs' was not correctly set, it should be either True or False.")

    @staticmethod
    def prepare_dataframe(all_data: DataFrame, condition_to_predict: str, model_name: str):
        """This function prepare the data, to have the features on one side (X)
        and the labels on the other side (y)

        Parameters
        ----------
        all_data : DataFrame
            The whole matrix with features and labels.
        condition_to_predict : str
            Which condition to predict (i.e., 'VIT' or '6MWT' or 'GPS').
        model_name : str
            Model to use (i.e., 'svc' or 'svr').

        Returns
        -------
        tuple
            Features matrix (X) and labels vector (y).

        Raises
        ------
        ValueError
            _description_
        """

        conditions = ["VIT", "6MWT", "GPS"]

        if condition_to_predict not in conditions:
            raise ValueError("Condition to predict not recognized. Choose either 'VIT', '6MWT', or 'GPS'.")

        conditions.remove(condition_to_predict)

        # Find columns ending with '_POST' except the condition_to_predict_POST column
        post_columns = [
            col for col in all_data.columns if col.endswith("_POST") and col != condition_to_predict + "_POST"
        ]
        print(f"Post columns that will be removed: {post_columns}")
        # post_to_remove = [condition + "_POST" for condition in conditions]

        all_data = all_data.drop(columns=post_columns)
        all_data = all_data.dropna(axis=0)

        if condition_to_predict == "6MWT":
            gmfcs_data = all_data["GMFCS"]
        else:
            gmfcs_data = None

        if model_name == "svc":
            y = Process.calculate_MCID(
                all_data[condition_to_predict + "_PRE"],
                all_data[condition_to_predict + "_POST"],
                condition_to_predict,
                gmfcs_data,
            )
            X = all_data.drop(columns=[condition_to_predict + "_POST"])
        elif model_name == "svr":
            y = all_data[condition_to_predict + "_POST"]
            X = all_data.drop(columns=[condition_to_predict + "_POST"])
        else:
            raise ValueError("Model name not recognized. Choose either 'svc' or 'svr'.")

        y = y.rename(condition_to_predict + "_MCID")

        return X, y

    # ─────────────────────────────────────────
    # X AND Y PREPARATION
    # ─────────────────────────────────────────
    @staticmethod
    def prepare_features_list(features_path: str) -> list:
        features = pd.read_excel(features_path, index_col=None)
        selected_features = features["features"].dropna().to_list()
        features_names = features["names"].dropna().to_list()
        return selected_features, features_names

    @staticmethod
    def prepare_data2(
        data_path: str,
        model_name: str,
        condition_to_predict: str,
        features: list = None,
    ):
        """
        This function prepares the data to be suitable for the model (i.e., features, & label)

        Parameters
        ----------
        data_path : str
            Path to the CSV file containing all the data (i.e., features, with PRE and POST variables).
        features : list
            List of features to select.
        condition_to_predict : str
            Sets the condition to predict (i.e., 'VIT' or '6MWT' or 'GPS'), i.e., the label.
        model_name : str
            String to specify the model to use (i.e., 'svc', 'svr').

        Returns
        -------
        X : DataFrame
            Features matrix.
        y : Series
            Labels vector.
        """

        all_data = Process.load_csv(data_path)

        data, y = Process.prepare_dataframe(all_data, condition_to_predict, model_name)

        if features:
            X = data[features]
        else:
            X = data.loc[:, ~data.columns.str.endswith("_MCID")]

        return X, y

    @staticmethod
    # TODO: splitting in several functions for each condition
    def prepare_data(data_path, condition_to_predict, model_name, features_path=None):
        """
        This function prepares the data to be suitable for the model (i.e., features, & label)

        Parameters
        ----------
        data_path : DataFrame
            Matrix with all the data (i.e., features and labels).
        condition_to_predict : str
            Sets the condition to predict (i.e., 'VIT' or '6MWT' or 'GPS').
        model_name : str
            String to specify the model to use (i.e., 'svc', 'svr').
        features_path : str, optional
            Path to the Excel file containing the features to select and their names (i.e., Max_Knee_flx/ext = Maximum Knee Flexion/Extension), by default None
            Like this, all features can be choosen by the user when running the model.

        Returns
        -------
        X : DataFrame
            Features matrix.
        y : Series
            Labels vector.
        features_names : list
            List of the names of the selected features.

        Raises
        ------
        ValueError
            If the condition to predict is not recognized.
        """

        all_data = Process.load_csv(data_path)

        data, y = Process.prepare_dataframe(all_data, condition_to_predict, model_name)

        if features_path:
            features = pd.read_excel(features_path)
            selected_features = features["features"].dropna().to_list()
            features_names = features["names"].dropna().to_list()
        else:
            selected_features = data.columns.to_list()
            features_names = data.columns.to_list()

        X = data[selected_features]

        return X, y, features_names

    # ─────────────────────────────────────────
    # SAVING
    # ─────────────────────────────────────────
    @staticmethod
    def save_df(df, output_path, separate_legs):
        if separate_legs:
            label = "leg_separated"
        else:
            label = "leg_not_separated"

        os.makedirs(output_path, exist_ok=True)
        nb_pp = len(df) // 2
        filename = f"all_data_{nb_pp}pp_{label}.csv"
        filepath = os.path.join(output_path, filename)
        df.to_csv(filepath)
        return filepath
