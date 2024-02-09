import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cmap = "YlGnBu"


class GRA:
    def __init__(self, data):
        self.data = data

    def dimensionlessProcessing(self, df_values, df_columns):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        res = scaler.fit_transform(df_values)
        return pd.DataFrame(res, columns=df_columns)

    def GRA(self, data, m=0):
        data = self.dimensionlessProcessing(data.values, data.columns)
        std = data.iloc[:, m]
        ce = data.copy()

        n = ce.shape[0]
        m = ce.shape[1]

        grap = np.zeros([n, m])
        for i in range(m):
            for j in range(n):
                grap[j, i] = abs(ce.iloc[j, i] - std[j])

        mmax = np.amax(grap)
        mmin = np.amin(grap)

        gra_cole = 0.5

        grap = pd.DataFrame(grap).applymap(lambda x: (mmin + gra_cole) / (x + gra_cole * mmax))

        RT = grap.mean(axis=0)
        return pd.Series(RT)

    def run_GRA(self, data):
        list_columns = np.arange(data.shape[1])
        df_local = pd.DataFrame(columns=list_columns)
        for i in np.arange(data.shape[1]):
            df_local.iloc[:, i] = self.GRA(data, m=i)
        return df_local

    def show_GRAHeatMap(self, data):
        # colormap = plt.cm.RdBu
        plt.figure(figsize=(7, 6))
        plt.title('Person Correlation of Features', y=1.05, size=18)
        sns.heatmap(data.astype(float), linewidths=0.1, square=True, cmap=cmap, linecolor='white', annot=False)
        # plt.savefig('outputs/GRAHeatmap.png')
        plt.show()

    def run(self):
        data_gra = self.run_GRA(self.data)
        self.show_GRAHeatMap(data_gra)
        return data_gra
