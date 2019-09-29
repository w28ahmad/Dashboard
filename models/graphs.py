from usedImports import plt

# ? Create a plot of the data


def visualize(df):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df["Low"]+df["High"])/2)
    plt.xticks(range(0, df.shape[0], 500), df["Date"].loc[::500], rotation=45)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("MidPrice", fontsize=18)
    plt.show()


# ? Compare 2 graphs


def compare_graphs(dataA, rangeA, labelA, dataB, rangeB, labelB, xlabel, ylabel):
    plt.figure(figsize=(18, 9))
    plt.plot(rangeA, dataA, color='b', label=labelA)
    plt.plot(rangeB, dataB, color='orange', label=labelB)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=18)
    plt.show()
