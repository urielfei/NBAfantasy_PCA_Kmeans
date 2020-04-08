import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering

class Nba:
    def __init__(self,my_df):
        self.raw_df = my_df
        self.df = None
        self.df_scaled = None
        self.pca_coef = None
        self.pca_score = None
        self.names = None
        self.df_groups = None
        self.centers = None
        self.groups = None

    def get_data(self,mode='per_game'):
        features = ['PLAYER_NAME','GP', 'FG_PCT', 'FT_PCT', 'FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        features_dict = {'PLAYER_NAME': 'Player', 'GP': 'GAMES', 'FG_PCT': 'FG%', 'FT_PCT': 'FT%',
                         'FG3M': '3PTS', 'PTS': 'PTS', 'REB': 'REB', 'AST': "AST",
                         'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV'}
        df = self.raw_df[features].rename(features_dict, axis='columns')
        df = df[df['GAMES'] >= 20]

        names = df["Player"]
        x = df.set_index('Player')

        if mode == 'per_game':
            x['3PTS'] = x['3PTS'] / x['GAMES']
            x['PTS'] = x['PTS'] / x['GAMES']
            x['REB'] = x['REB'] / x['GAMES']
            x['STL'] = x['STL'] / x['GAMES']
            x['BLK'] = x['BLK'] / x['GAMES']
            x['TOV'] = x['TOV'] / x['GAMES']

        x['TOV'] = x['TOV'] * -1
        df = x.drop('GAMES', axis=1)
        sc = StandardScaler()
        sc.fit(df)
        x_scaled = sc.transform(df)
        x = pd.DataFrame(data=x_scaled, columns=features[2:11], index=names)
        df['z'] = x.sum(axis=1)
        df = df.sort_values(by='z', ascending=False)
        self.df = df[0:160]
        self.names = self.df.index
        return self.df

    def get_data_scaled(self,mode='per_game'):
        features_dict = {'PLAYER_NAME': 'Player','GP':'GAMES' ,'FG_PCT': 'FG%', 'FT_PCT': 'FT%',
                         'FG3M': '3PTS', 'PTS': 'PTS', 'REB': 'REB', 'AST': "AST",
                         'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV'}
        features = ['PLAYER_NAME','GP', 'FG_PCT', 'FT_PCT', 'FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        df = self.raw_df[features].rename(features_dict, axis='columns')
        df = df[df['GAMES'] >= 20]

        names = df["Player"]
        x = df.set_index('Player')

        if mode == 'per_game':
            x['3PTS'] = x['3PTS'] / x['GAMES']
            x['PTS'] = x['PTS'] / x['GAMES']
            x['REB'] = x['REB'] / x['GAMES']
            x['STL'] = x['STL'] / x['GAMES']
            x['BLK'] = x['BLK'] / x['GAMES']
            x['TOV'] = x['TOV'] / x['GAMES']

        x['TOV'] = x['TOV'] * -1
        x = x.drop('GAMES',axis=1)
        sc = StandardScaler()
        sc.fit(x)
        x_scaled = sc.transform(x)
        x = pd.DataFrame(data=x_scaled, columns=x.columns, index=names)
        x['z'] = x.sum(axis=1)
        x = x.sort_values(by='z', ascending=False)
        self.df_scaled = x[0:160]
        return self.df_scaled

    def get_data_Kmeans(self,clusters):
        raw_df = self.get_data_scaled()
        raw_df = raw_df.reset_index()
        X = raw_df.drop('z',axis=1)
        X = X.drop('Player', axis=1)

        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        df_y_kmeans = pd.DataFrame(data=y_kmeans, columns=['Group'])
        df_groups = pd.concat([raw_df, df_y_kmeans], axis=1).set_index('Player')
        self.df_groups = df_groups
        self.centers = kmeans.cluster_centers_
        self.groups = y_kmeans
        return self.df_groups

    def pca(self):
        x = self.get_data_scaled()
        x = x.loc[:, x.columns != 'z']

        pca = PCA(n_components=2)
        pca_x = pca.fit_transform(x)

        self.pca_coef = np.transpose(pca.components_[0:2, :])
        self.pca_score = pca_x[:, 0:2]

    def biplot(self,plot_name=None):
        font_title = {'family': 'serif',
                      'color': 'darkred',
                      'weight': 'normal',
                      'size': 16,}
        self.pca()

        x = self.df_scaled.loc[:, self.df_scaled.columns != 'z']
        x_biplot = x.reset_index()
        x = x_biplot.loc[:, x_biplot.columns != 'Player']

        xs = self.pca_score[:, 0]
        ys = self.pca_score[:, 1]
        scale_x = 1.0 / (xs.max() - xs.min())
        scale_y = 1.0 / (ys.max() - ys.min())


        fig, ax = plt.subplots()
        ax.scatter(xs* scale_x, ys * scale_y, s=8, c='darkblue')

        col = list(x.columns)
        n = self.pca_coef.shape[0]
        for i in range(n):
            ax.arrow(0, 0, self.pca_coef[i, 0] * 0.8, self.pca_coef[i, 1] * 0.8, color='crimson', alpha=2)
            ax.text(self.pca_coef[i, 0] * 0.87, self.pca_coef[i, 1] * 0.87, col[i], color='black',
                    ha='center', va='center',weight='bold')

        k = self.df_scaled.reset_index()
        names = k['Player'].str.split(" ", expand=True)
        names_index = names[0].str[0] + '.' + names[1]
        n = pd.DataFrame(data={"xs": xs, "ys": ys}, index=names_index)
        right = n.sort_values(by=["xs", "ys"], ascending=False)[0:2]
        up = n.sort_values(by=["ys", "xs"], ascending=False)[2:4]
        left = n.sort_values(by=["xs", "ys"], ascending=True)[0:1]

        for i, txt in enumerate(right.index):
            ax.annotate(txt, (right['xs'][i] * scale_x, right['ys'][i] * scale_y), ha='center')

        for i, txt in enumerate(left.index):
            ax.annotate(txt, (left['xs'][i] * scale_x, left['ys'][i] * scale_y))

        for i, txt in enumerate(up.index):
            ax.annotate(txt, (up['xs'][i] * scale_x, up['ys'][i] * scale_y))

        ax.set_xlabel("PC{}".format(1))
        ax.set_ylabel("PC{}".format(2))

        ax.grid(True)
        ax.set_xlim([-0.70, 0.65])
        ax.set_ylim([-0.55, 0.58])

        if plot_name == None:
            plt.show()
        else:
           p_name = plot_name + '.png'
           plt.savefig(p_name)
           print("Plot saved!")

    def choose_K_means(self, k):
        raw_df = self.get_data_scaled().reset_index()
        X = raw_df.drop('z', axis=1)
        X = X.drop('Player', axis=1)
        Sum_of_squared_distances = []
        K = range(1, k)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def choose_K_means_sil(self, k):
        raw_df = self.get_data_scaled().reset_index()
        X = raw_df.drop('z', axis=1)
        X = X.drop('Player', axis=1)
        sil = []
        for k in range(2, k + 1):
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            sil.append(silhouette_score(X, labels, metric='euclidean'))
        plt.plot(range(2, k + 1),sil)
        plt.show()

    def spectral_Kmeans(self,clusters):
        raw_df = self.get_data_scaled().reset_index()
        X = raw_df.drop('z',axis=1)
        X = X.drop('Player', axis=1).to_numpy()

        model = SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors',
                                   assign_labels='kmeans')
        labels = model.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        plt.show()

    def Kmeans_results(self,clusters,data='reg'):
        if data == 'reg':
            data_k_means = self.get_data_Kmeans(clusters)
            data_k_means = data_k_means[['z','Group']]
            groups = data_k_means.groupby(['Group'])

            for key, group in groups:
                print(key)
                print(groups.get_group(key).head())
        print(data_k_means.groupby(['Group']).mean())

    def Kmeans_plot(self):
        self.pca()
        data_k_means = self.get_data_Kmeans(5)
        xs = self.pca_score[:, 0]
        ys = self.pca_score[:, 1]
        df = pd.DataFrame(dict(x=xs, y=ys, label=self.groups))
        groups = df.groupby(df.label)

        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', label=name)

        ax.legend()
        centers = obj.centers
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

        # plt.savefig('aa.png')

        plt.show()




'''Read data'''

raw = leaguedashplayerstats.LeagueDashPlayerStats(season='2019-20').get_data_frames()[0]
obj = Nba(raw)

'''Data prep'''
#data = obj.get_data()
#print(data.columns,data.head())

#data_pca = obj.get_data_scaled()
#print(data_pca.columns,data_pca.head(),data_pca.shape)

'''Kmeans'''
#data_k_means = obj.get_data_Kmeans(5)
#print(data_k_means.columns,data_k_means.shape,data_k_means.head())

'''PCA'''
#obj.pca()
#coef = obj.pca_coef
#score = obj.pca_score
#print(coef,score)

'''Biplot'''
#obj.biplot()

'''Evaluate K clusters to use'''
#obj.choose_K_means(10)
#obj.choose_K_means_sil(10)

'''spectral kmeans'''
#obj.spectral_Kmeans(2)

''' K means Groups head data'''
#obj.Kmeans_results(5)

'''K means Plot'''
obj.Kmeans_plot()
