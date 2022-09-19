from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def draw_bar_plot(df:pd.DataFrame):
    x = np.arange(len(df["accuracy"]))
    width = 0.35
    fig,ax = plt.subplots()
    ax.bar(x-width/2, df["accuracy"], width, label="accuracy")
    ax.bar(x+width/2, df["confidence"], width, label="confidence")
    ax.set_xlabel('group')
    ax.set_ylabel('probability')
    ax.set_title('reliability diagram')
    ax.set_xticks(x)
    # label1 = ["0-.1",".1-.2",".2-.3",".3-.4",".4-.5",".5-.6",".6-.7",".7-.8",".8-.9",".9-1"]
    # label2 = [".1-.2",".2-.3",".3-.4",".4-.5",".5-.6",".6-.7",".7-.8",".8-.9",".9-1"]
    # ax.set_xticklabels(label2)
    ax.legend()
    fig.tight_layout()
    plt.show()


def ece(df: pd.DataFrame):
    ece_ = ((df["accuracy"] - df["confidence"]).abs() * df["count"]).sum() / (df["count"].sum())
    return ece_


def create_grouped_result(result):
    # create a DataFrame that groups the result
    result["group"] = result["prob"].floordiv(0.1)
    group_count = result.groupby("group")["group"].count()
    group_accurate_count = result[result["pred"] == result["label"]].groupby("group")["group"].count()
    group_accuracy = group_accurate_count / group_count
    group_accuracy.fillna(0)
    group_confidence = result.groupby("group")["prob"].mean()
    group_result = pd.DataFrame({"count": group_count,
                                 "accuracy": group_accuracy,
                                 "confidence": group_confidence})
    return group_result


if __name__ == "__main__":
    result = pd.read_csv("experiment/result_30.csv")
    print(result)
    group_result = create_grouped_result(result)
    print(group_result)
    draw_bar_plot(group_result)
    ece_ = ece(group_result)
    print("ECE: ", ece_)
    accuracy = len(result[result["pred"] == result["label"]].index) / len(result.index)
    print("Accuracy: ", accuracy)



