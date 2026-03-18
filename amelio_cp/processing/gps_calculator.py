import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.io import loadmat
from scipy import interpolate
import sys

print(">>> Running with:", sys.executable)


# ------------------------------------    This Part loads the data, extract the features and save them in an excel file     -----------------------------------


# ----- Acess nested structure -----
def access_struct(data, structs):

    for struct in structs:
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            data = data[0, 0][struct]
        else:
            data = data[struct]

    return data


# ----- Sort the order of the files -----
def natural_sort_key(s):

    return [int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)]


def get_data_for_joint(all_data, side, joint):
    joint_with_side_name = side + joint
    if joint == "Hip" or joint == "LPelvis" or joint == "Pelvis":
        joint_kin = all_data[0, 0][joint_with_side_name][0][0]
        joint_kin = np.reshape(joint_kin, (100, 3), order="F")
    elif joint == "FootProgress":
        joint_kin = all_data[0, 0][joint_with_side_name][0][0][:, 2]
        joint_kin = np.reshape(joint_kin, (100, 1), order="F")
    else:
        joint_kin = all_data[0, 0][joint_with_side_name][0][0][:, 0]
        joint_kin = np.reshape(joint_kin, (100, 1), order="F")

    return joint_kin


# ----- Access features in the mat files -----
def gps_features(file_path: str, file_num: int, num_all_files: int, separate_legs: bool, output_path: str):
    file = loadmat(file_path)
    side_structs = ["Right", "Left"]
    joint_names = ["Pelvis", "Hip", "Knee", "Ankle", "FootProgress"]
    if separate_legs == False:
        joint_data = np.empty((100, 0))

        for side_struct in side_structs:
            structs = ["c", "results", side_struct, "angAtFullCycle"]
            all_data = access_struct(file, structs)
            side = side_struct[0]

            if side_struct == "Left":
                joint_kin = get_data_for_joint(all_data, side, joint)
                joint_data = np.concatenate((joint_data, joint_kin), axis=1)

            for joint in joint_names:
                joint_kin = get_data_for_joint(all_data, side, joint)
                joint_data = np.concatenate((joint_data, joint_kin), axis=1)

        joint_data = pd.DataFrame(joint_data)
        joint_data.columns = [
            "R_Hip flex/ext",
            "R_Hip abd/add",
            "R_Hip int/ext rotation",
            "R_Knee flx/ext",
            "R_Ankle dorsi/plantar flx",
            "R_foot progression",
            "L_Pelvis",
            "L_Pelvis",
            "L_Pelvis",
            "L_Hip flex/ext",
            "L_Hip abd/add",
            "L_Hip int/ext rotation",
            "L_Knee flx/ext",
            "L_Ankle dorsi/plantar flx",
            "L_foot progression",
        ]

        if output_path:
            if file_num < (((num_all_files) + 1) // 2):
                joint_data.to_csv(output_path + "Subject%d_PreLokomat.csv" % (file_num + 1), index=False)

            else:
                joint_data.to_csv(
                    output_path + "Subject%d_PostLokomat.csv" % (file_num + 1 - (((num_all_files) + 1) // 2)), index=False
                )

        print("The data of both sides is extracted together!")
        return joint_data

    else:
        joint_data_both_sides = pd.DataFrame()
        for side_struct in side_structs:
            joint_data = np.empty((100, 0))
            structs = ["c", "results", side_struct, "angAtFullCycle"]
            all_data = access_struct(file, structs)
            side = side_struct[0]

            for joint in joint_names:
                joint_kin = get_data_for_joint(all_data, side, joint)
                joint_data = np.concatenate((joint_data, joint_kin), axis=1)

            joint_data = pd.DataFrame(joint_data)
            dofs = [
                "Pelvis tilt",
                "Pelvis rot",
                "Pelvis obli",
                "Hip flex/ext",
                "Hip abd/add",
                "Hip int/ext rotation",
                "Knee flx/ext",
                "Ankle dorsi/plantar flx",
                "foot progression",
            ]
            joint_data.columns = [side + "_" + dof for dof in dofs]

            if output_path:
                if file_num < (((num_all_files) + 1) // 2) or file_num == 25:
                    joint_data.to_csv(output_path + "Subject%d_%s_PreLokomat.csv" % (file_num + 1, side), index=False)
                    print("Subject%d_PreLokomat.csv for %s" % (file_num + 1, side_struct))
                else:
                    joint_data.to_csv(
                        output_path + "Subject%d_%s_PostLokomat.csv" % (file_num + 1 - (((num_all_files) + 1) // 2), side),
                        index=False,
                    )
                    print("Subject%d_PreLokomat.csv for %s" % (file_num + 1, side_struct))

            joint_data_both_sides = pd.concat([joint_data_both_sides, joint_data], axis=1)

        print("Subject%d" % (file_num + 1))
        print("The data of each side is extracted separately!")
        return joint_data_both_sides


def calculate_gvs(leg, reference, separate_legs):
    all_GVS = []
    if separate_legs == True:
        n = len(reference.columns.to_list())
    elif separate_legs == False:
        n = 15
    else:
        raise ValueError("separate_legs was not True nor False")

    for i in range(n):
        differences = leg.values[:, i] - reference.values[:, i]
        leg_GVS = np.sqrt(np.mean(differences**2))
        all_GVS.append(leg_GVS)
    return all_GVS


def calculate_gps(data, reference, separate_legs: bool):

    if separate_legs == False:
        all_GVS = calculate_gvs(data, reference, separate_legs)
        GPS = np.sqrt(np.mean(all_GVS**2))
        return GPS

    else:
        right_ref = reference.loc[:, reference.columns.str.startswith("R")]
        left_ref = reference.loc[:, reference.columns.str.startswith("L")]
        right_leg = data.loc[:, data.columns.str.startswith("R")]
        right_leg.drop(['R_Pelvis tilt', 'R_Pelvis rot', 'R_Pelvis obli'], axis=1, inplace=True)
        left_leg = data.loc[:, data.columns.str.startswith("L")]
        # reference = np.concatenate((reference[6:9], (reference[9:] + reference[:6]) / 2))

        right_all_GVS = calculate_gvs(right_leg, right_ref, separate_legs)
        left_all_GVS = calculate_gvs(left_leg, left_ref, separate_legs)

        r_GPS = np.sqrt(np.mean(np.power(right_all_GVS, 2)))
        l_GPS = np.sqrt(np.mean(np.power(left_all_GVS, 2)))

        return r_GPS, l_GPS


def process_files(input_directory):
    pre_files = [f for f in os.listdir(input_directory) if f.endswith("eLokomat.mat")]
    post_files = [f for f in os.listdir(input_directory) if f.endswith("stLokomat.mat")]
    mat_files_pre_sorted = sorted(pre_files, key=natural_sort_key)
    mat_files_post_sorted = sorted(post_files, key=natural_sort_key)
    mat_files_sorted = mat_files_pre_sorted + mat_files_post_sorted

    return mat_files_sorted


# %%--------------------------------------------------------------------------
# TODO: transform this script into a class and remove if main
def main(input_directory, file_path_for_ref, output_path_for_gps_cache, output_path, separate_legs=True):
    mat_files_sorted = process_files(input_directory)

    reference = pd.read_csv(file_path_for_ref)
    # reference.drop(reference.columns[0], axis=1, inplace=True)
    # reference = reference.values
    all_GPS = []

    # ----- Read data: First read all Pre data and the all Post data -----
    print("----------------------------------")
    print("LOADING DATA")
    print("----------------------------------")

    for indx, file in enumerate(mat_files_sorted, start=0):
        file_path = os.path.join(input_directory, file)
        joint_data = gps_features(
            file_path,
            output_path=output_path_for_gps_cache,
            file_num=indx,
            num_all_files=len(list(enumerate(mat_files_sorted, start=0))),
            separate_legs=separate_legs,
        )
        gps = calculate_gps(joint_data, reference, separate_legs)
        all_GPS.append(gps)

    all_GPS = np.array(all_GPS)
    all_GPS = all_GPS.flatten()
    reshaped_all_GPS = all_GPS.reshape((-1, 2), order="F")  # separates in two col (i.e., Pre, and Post)
    GPS_output = pd.DataFrame(reshaped_all_GPS)
    GPS_output.columns = ["Pre", "Post"]

    if separate_legs == True:
        row_names = ["Right", "Left"]
        row_names = np.tile(row_names, len(reshaped_all_GPS) // 2)
        GPS_output.index = row_names
        if output_path:
            GPS_output.to_csv(f"{output_path}gps.csv")

    print("----------------------------------")
    print("ANALYSIS DONE!")
    print("----------------------------------")


if __name__ == "__main__":
    input = input("enter: ")
    input_directory = input
    file_path_for_ref = input + r"/average_result.csv"
    output_path_for_gps_cache = input + r"/gps_cache/"
    output_path = input + r"/"
    main(input_directory, file_path_for_ref, output_path_for_gps_cache, output_path, separate_legs=True)
