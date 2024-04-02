import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Consumer:
    def __init__(self):
        self.util_weight = np.random.normal(4, 1)

    def utility_func(self, q: float) -> float:
        return (self.util_weight * q) ** 0.5

def main():
    consumers = pd.DataFrame(columns=['Consumer', 'Utility_Function'])
    for i in range(100):
        consumer = Consumer()
        consumers.loc[i] = [consumer, consumer.utility_func]

    data = pd.DataFrame()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Utility')
    for i in range(0, 100, 1):
        data[f'{i/10}'] = consumers.apply(lambda x: x['Utility_Function'](i/10), axis=1)
        # ax.scatter([i/10 for x in range(len(data))], data[f'{i/10}'], label=f'{i/10}')

    # plt.show()

    data.columns = data.columns.astype(float)


    for consumer in data.iterrows():
        ax.plot(consumer[1], linestyle="dashed", linewidth=0.5, color="black")

    ax.plot(data.mean(), color="red", linewidth=4)
    ax.plot(np.linspace(0, 10, 100), (lambda x: (4*x)**0.5)(np.linspace(0, 10, 100)), color="blue", linewidth=4)

    plt.show()
    print("pass")
    pass

if __name__ == "__main__":
    main()