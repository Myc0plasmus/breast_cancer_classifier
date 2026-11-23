
Imię|Nazwisko|nr. indeksu
-|-|-
Natan|Jabłoński|155621

# Analiza eksploracyjna

## Analiza korelacji
Poniższa sekcja przedstawia analizę korelacji na rekordach z surowych danych ze zbioru treningowego i walidacyjnego.

<!-- ### Kod źródłowy -->
<!-- Poniższa analiza została wykonana za pomocą poniższego kodu źródłowego -->
<!-- ```python -->
<!-- {{breast_cancer_classifier.analysis.correlation_analysis}} -->
<!-- ``` -->

### Macierz korelacji

![](artifacts/figures/feature_correlation_matrix.png)

### Wykres pudełkowy wartości korelacji


![](artifacts/figures/correlation_box_plot.png)

### Wykres pudełkowy wartości korelacji do targetu

![](artifacts/figures/target_correlation_box_plot.png)

### Największe wartości korelacji

{{table artifacts/tables/top_target_corr_values.csv}}

### Statystyki wartości korelacji do targetu

{{table artifacts/tables/target_correlation_statistics.csv}}


<!-- ## UMAP -->
<!-- ![[../artifacts/figures/cor_inter_params.png]] -->








