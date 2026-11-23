
Imię|Nazwisko|nr. indeksu
-|-|-
Natan|Jabłoński|155621

# Kod źródłowy:
Kod źródłowy jest dostępny na [githubie](https://github.com/Myc0plasmus/breast_cancer_classifier).

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

Korelacja cech do targetu jest głównie ujemna, aczkolwiek dla paru cech jest wysoka. Mimo to przez fakt, że target jest binarny to lepiej uważać z poleganiem na korelacji do targetu pod kątem dobierania cech.

### Wykres pudełkowy wartości korelacji do targetu

![](artifacts/figures/target_correlation_box_plot.png)

Brak dużej ilość wartości odstających

### Największe wartości korelacji do targetu

{{table artifacts/tables/top_target_corr_values.csv}}

### Statystyki wartości korelacji do targetu

{{table artifacts/tables/target_correlation_statistics.csv}}

Te wartości pokazują że można w miarę zaufać temu, że wysoka korelacja do niektórych cech.

# Wyniki modelu

## Random Forest

### Ewaluacja datasetu treningowego
{{table artifacts/tables/Train_dataset_evaluation.csv}}

### Ewaluacja datasetu ewalucyjnego
{{table artifacts/tables/Eval_dataset_evaluation.csv}}

### Ewaluacja datasetu testowego
{{table artifacts/tables/Test_dataset_evaluation.csv}}

### Wyniki ewaluacji-krzyżowej
Do ewaluacji wykorzystanie stratified K-fold z k=20:

{{table artifacts/tables/my_random_forest_cross_eval_metrics.csv}}

### Krzywa uczenia

![](artifacts/figures/my_random_forest_learning_curve.png)




<!-- ## UMAP -->
<!-- ![[../artifacts/figures/cor_inter_params.png]] -->








