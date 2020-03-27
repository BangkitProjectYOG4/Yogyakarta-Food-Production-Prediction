import matplotlib.pyplot as plt
import pandas as pd


def ploting_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error (Target)')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['mae'], label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error (Target^2)')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['mae'], label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
