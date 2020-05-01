from utils import convert_vott_csv_to_yolo

from PIL import Image
from os import path, makedirs
import os
import re
import pandas as pd
import sys
import argparse


if __name__ == "__main__":
    # surpress any inhereted default values
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--VoTT_Folder",
        type=str,
        help="Absolute path to the exported files from the image tagging step with VoTT. Default is ",
    )
    parser.add_argument(
        "--YOLO_filename",
        type=str,
        default="./data/data_train.txt",
        help="Absolute path to the file where the annotations in YOLO format should be saved. Default is "
        + "./data/data_train.txt",
    )

    FLAGS = parser.parse_args()
    FLAGS.VoTT_csv = os.path.join(FLAGS.VoTT_Folder, "Annotations-export.csv")
    # Prepare the dataset for YOLO
    multi_df = pd.read_csv(FLAGS.VoTT_csv)
    labels = multi_df["label"].unique()
    labeldict = dict(zip(labels, range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
    train_path = FLAGS.VoTT_Folder
    convert_vott_csv_to_yolo(
        multi_df, labeldict, path=train_path, target_name=FLAGS.YOLO_filename
    )

    # Make classes file
    file = open("./data/data_classes.txt", "w")

    # Sort Dict by Values
    SortedLabelDict = sorted(labeldict.items(), key=lambda x: x[1])
    for elem in SortedLabelDict:
        file.write(elem[0] + "\n")
    file.close()
