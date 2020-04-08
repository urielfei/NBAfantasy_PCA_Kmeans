import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from final_Kmeans import Nba



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
