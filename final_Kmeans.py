import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.cluster import KMeans

class Nba:
    def __init__(self,my_df):
        self.raw_df = my_df
        self.df_pca = None
        self.pca_coef = None
        self.pca_score = None
        self.names = None
        self.df_groups = None
        self.centers = None
        self.groups = None

    def create_data(self):
        features_dict = {'PLAYER_NAME': 'Player', 'GP': 'GAMES', 'FG_PCT': 'FG%', 'FT_PCT': 'FT%',
                         'FG3M': '3PTS', 'PTS': 'PTS', 'REB': 'REB', 'AST': "AST",
                         'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV'}
        features = ['PLAYER_NAME', 'GP', 'FG_PCT', 'BLK', 'REB', 'STL', 'PTS', 'TOV', 'AST', 'FT_PCT', 'FG3M']
        df = self.raw_df[features].rename(features_dict, axis='columns')
        names = df["Player"]
        x = df.set_index('Player')
        x['3PTS'] = x['3PTS'] / x['GAMES']
        x['PTS'] = x['PTS'] / x['GAMES']
        x['REB'] = x['REB'] / x['GAMES']
        x['STL'] = x['STL'] / x['GAMES']
        x['BLK'] = x['BLK'] / x['GAMES']
        x['TOV'] = x['TOV'] / x['GAMES'] * -1
        df = x.drop('GAMES', axis=1)
        sc = StandardScaler()
        sc.fit(df)
        x_scaled = sc.transform(df)
        x = pd.DataFrame(data=x_scaled, columns=features[2:11], index=names)
        df['z'] = x.sum(axis=1)
        df = df.sort_values(by='z', ascending=False)
        self.df = df[0:160]
        self.names = self.df.index

    def get_data(self):
        self.create_data()
        return self.df

    def create_data_pca(self):
        features_dict = {'PLAYER_NAME': 'Player','GP':'GAMES' ,'FG_PCT': 'FG%', 'FT_PCT': 'FT%',
                         'FG3M': '3PTS', 'PTS': 'PTS', 'REB': 'REB', 'AST': "AST",
                         'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV'}
        features = ['PLAYER_NAME','GP', 'FG_PCT', 'FT_PCT', 'FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        df = self.raw_df[features].rename(features_dict, axis='columns')
        names = df["Player"]
        x = df.set_index('Player')
        x['3PTS'] = x['3PTS']/x['GAMES']
        x['PTS'] = x['PTS']/x['GAMES']
        x['REB'] = x['REB']/x['GAMES']
        x['STL'] = x['STL']/x['GAMES']
        x['BLK'] = x['BLK']/x['GAMES']
        x['TOV'] = x['TOV']/x['GAMES'] * -1
        x = x.drop('GAMES',axis=1)
        sc = StandardScaler()
        sc.fit(x)
        x_scaled = sc.transform(x)
        x = pd.DataFrame(data=x_scaled, columns=x.columns, index=names)
        x['z'] = x.sum(axis=1)
        x = x.sort_values(by='z', ascending=False)
        self.df_pca = x[0:160]

    def get_data_pca(self):
        self.create_data_pca()
        return self.df_pca

    def create_data_Kmeans(self,clusters):
        raw_df = self.df_pca.reset_index()
        X = raw_df.drop('z',axis=1)
        X = X.drop('Player', axis=1)

        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        df_y_kmeans = pd.DataFrame(data=y_kmeans, columns=['Group'])
        df_groups = pd.concat([raw_df, df_y_kmeans], axis=1).set_index(self.names)
        self.df_groups = df_groups
        self.centers = kmeans.cluster_centers_
        self.groups = y_kmeans

    def get_data_Kmeans(self):
        self.create_data_Kmeans()
        return self.df_groups

    def pca(self):
        x = self.df_pca
        x = x.loc[:, x.columns != 'z']

        pca = PCA(n_components=2)
        pca_x = pca.fit_transform(x)

        self.pca_coef = np.transpose(pca.components_[0:2, :])
        self.pca_score = pca_x[:, 0:2]

    def biplot(self,plot_name):
        font_title = {'family': 'serif',
                      'color': 'darkred',
                      'weight': 'normal',
                      'size': 16,
                      }

        x = self.df_pca.loc[:, self.df_pca.columns != 'z']
        x_biplot = x.reset_index()
        x = x_biplot.loc[:, x_biplot.columns != 'Player']
        #x = x.set_index('Pos')

        xs = self.pca_score[:, 0]
        ys = self.pca_score[:, 1]
        scale_x = 1.0 / (xs.max() - xs.min())
        scale_y = 1.0 / (ys.max() - ys.min())

        #w = pd.DataFrame(data={"xs": xs, "ys": ys}, index=x.index)
        #groups = w.groupby(w.index)
        #c = groups.get_group('Center')
        #g = groups.get_group('Guard')
        #f = groups.get_group('Forward')

        fig, ax = plt.subplots()
        ax.scatter(xs* scale_x, ys * scale_y, s=8, c='darkblue')
        #ax.scatter(xs* scale_x, ys * scale_y, s=8, c='b', label='Forwards')
        #ax.scatter(xs* scale_x, ys * scale_y, s=8, c='cornflowerblue', label='Guards')

        col = list(x.columns)
        n = self.pca_coef.shape[0]
        for i in range(n):
            ax.arrow(0, 0, self.pca_coef[i, 0] * 0.8, self.pca_coef[i, 1] * 0.8, color='crimson', alpha=2)
            ax.text(self.pca_coef[i, 0] * 0.87, self.pca_coef[i, 1] * 0.87, col[i], color='black',
                    ha='center', va='center',weight='bold')

        k = self.df_pca.reset_index()
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
        ax.set_title(plot_name+' Season'+"\n", va='bottom', fontdict=font_title)
        #ax.legend(loc="best")
        ax.grid(True)
        ax.set_xlim([-0.70, 0.65])
        ax.set_ylim([-0.55, 0.58])
        p_name = plot_name + '.png'
        plt.savefig(p_name)
        print("Plot saved!")


raw = leaguedashplayerstats.LeagueDashPlayerStats(season='2019-20').get_data_frames()[0]
obj = Nba(raw)

obj.create_data()
obj.create_data_pca()
obj.create_data_Kmeans(5)
obj.pca()

xs = obj.pca_score[:, 0]
ys = obj.pca_score[:, 1]
df = pd.DataFrame(dict(x=xs, y=ys, label=obj.groups))
groups = df.groupby(df.label)

#scale_x = 1.0 / (xs.max() - xs.min())
#scale_y = 1.0 / (ys.max() - ys.min())

fig, ax = plt.subplots()
#ax.scatter(xs, ys, s=8, c='darkblue')
for name, group in groups:
    ax.plot(group.x, group.y
            , marker='o'
            , linestyle=''
            # , ms=12
            , label=name
            )
ax.legend()
centers = obj.centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])

plt.savefig('aa.png')

