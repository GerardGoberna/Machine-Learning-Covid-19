#Matriu de correlacions
correlacio = dataset.corr()

sns.heatmap(correlacio, annot = False)
plt.show()

#Ordenem els valors segons la seva correlació
corr_parelles = correlacio.unstack()
print(corr_parelles)


#Escollim els valors més alts de correlació

corr_negatives = corr_parelles[corr_parelles < 0]
print(corr_negatives)

corr_forta = corr_parelles[abs(corr_parelles) > 0.5]
print(corr_forta)

#Fem el mateix però només amb la variable presència de Covid

correlacions_covid = corr_parelles["COVID-19"]
ordenats = correlacions_covid.sort_values(kind="quicksort")
print(ordenats)

cor_negativa = ordenats[ordenats < 0]
print(cor_negativa)

cor_alta = ordenats[abs(ordenats) > 0.3]
print(cor_alta)