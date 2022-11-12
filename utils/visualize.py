from matplotlib import pyplot as plt


def showMe(data, range=[-10000,10000]):
    plt.rcParams["figure.figsize"] = [17, 10]
    max_std = 0
    for d in data:
        if d.std() > max_std:
            max_std = d.std()

    for i, channel in enumerate(data):
        offset = i*max_std*2
        plt.plot(channel+offset)
        plt.show()

def showHistory(history):
    plt.rcParams["figure.figsize"] = [5, 5]
    with plt.rc_context({'figure.facecolor':'white'}):
        for key in history.history.keys():

            if "val_" not in key and "lr" != key:
                try:
                    plt.clf()
                    
                    plt.plot(history.history[key])
                    try:
                        plt.plot(history.history["val_" + key])
                    except:
                        print("Overfit Warning")
                    plt.ylabel(key)
                    plt.xlabel("epoch")
                    plt.legend(["train", "validation"], loc="upper left")
                    plt.show()
                except Exception as e:
                    print(e)
                    ...
