# INTRODUCCION

Este análisis tiene como objetivo examinar la relación entre ciertas condiciones médicas, como la presión arterial y los niveles de glucosa en sangre, y la diabetes. También se busca construir un modelo que pueda predecir si una persona tiene diabetes basándose en esta información, utilizando diferentes métodos y comparándolos entre sí. Es importante tener en cuenta que el conjunto de datos utilizado en este estudio se compone exclusivamente de mujeres mayores de 21 años, por lo que los resultados pueden variar significativamente en personas que no cumplan estas características.

# Exploracion y Limpieza de los datos

#### importando librerias a utilizar


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv('/content/sample_data/diabetes.csv')
df.head()
```





  <div id="df-7c80723e-d4b8-4f5f-99c3-93e86cc9d9f0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7c80723e-d4b8-4f5f-99c3-93e86cc9d9f0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7c80723e-d4b8-4f5f-99c3-93e86cc9d9f0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7c80723e-d4b8-4f5f-99c3-93e86cc9d9f0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Primero, vamos a analizar la distribución de los datos.


```python
df.describe()
```





  <div id="df-359fd3ea-1600-43f6-b635-4dde5afdb8eb">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-359fd3ea-1600-43f6-b635-4dde5afdb8eb')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-359fd3ea-1600-43f6-b635-4dde5afdb8eb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-359fd3ea-1600-43f6-b635-4dde5afdb8eb');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Hemos observado que todas las columnas contienen valores numéricos y que estos valores varían ampliamente entre las diferentes columnas. Para garantizar un rendimiento óptimo de nuestro modelo, es necesario normalizar los datos.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB



```python
df.isin([0]).sum()
```




    Pregnancies                 111
    Glucose                       5
    BloodPressure                35
    SkinThickness               227
    Insulin                     374
    BMI                          11
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                     500
    dtype: int64



No se han encontrado columnas con datos faltantes, sin embargo, hemos notado la presencia de numerosos valores de cero en las columnas correspondientes a "insulina" y "grosor de la piel". Dado que el valor cero no es válido para representar la información que intentan describir, podemos interpretar que las mediciones en estos casos no fueron realizadas o se cargaron incorrectamente.

Además, las columnas "glucosa", "IMC" (Índice de Masa Corporal) y "presión sanguínea" también contienen valores inválidos, aunque en menor cantidad.


```python
corr_matrix = np.corrcoef(df.values.T)
fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=df.columns,
        y=df.columns,
        colorscale='dense'))
fig.update_layout(
    title = 'mapa de calor',
    title_x = 0.5,
    width = 1300,
    height = 1000
)
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="a41b33c6-5f60-4c1c-8f9a-da3153423a1d" class="plotly-graph-div" style="height:1000px; width:1300px;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a41b33c6-5f60-4c1c-8f9a-da3153423a1d")) {                    Plotly.newPlot(                        "a41b33c6-5f60-4c1c-8f9a-da3153423a1d",                        [{"colorscale":[[0.0,"rgb(230, 240, 240)"],[0.09090909090909091,"rgb(191, 221, 229)"],[0.18181818181818182,"rgb(156, 201, 226)"],[0.2727272727272727,"rgb(129, 180, 227)"],[0.36363636363636365,"rgb(115, 154, 228)"],[0.45454545454545453,"rgb(117, 127, 221)"],[0.5454545454545454,"rgb(120, 100, 202)"],[0.6363636363636364,"rgb(119, 74, 175)"],[0.7272727272727273,"rgb(113, 50, 141)"],[0.8181818181818182,"rgb(100, 31, 104)"],[0.9090909090909091,"rgb(80, 20, 66)"],[1.0,"rgb(54, 14, 36)"]],"x":["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"],"y":["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"],"z":[[0.9999999999999999,0.12945867149927245,0.14128197740713988,-0.08167177444900722,-0.07353461435162825,0.017683090727830596,-0.03352267296261302,0.5443412284023394,0.22189815303398613],[0.12945867149927245,0.9999999999999999,0.15258958656866448,0.057327890738177074,0.33135710992020934,0.22107106945898294,0.13733729982837065,0.2635143198243338,0.46658139830687295],[0.14128197740713988,0.15258958656866448,1.0,0.20737053840307082,0.08893337837319316,0.2818052888499106,0.041264947930098564,0.23952794642136363,0.0650683595503327],[-0.08167177444900721,0.05732789073817707,0.20737053840307082,1.0,0.4367825701200126,0.39257320415903835,0.18392757295416323,-0.11397026236774165,0.0747522319183194],[-0.07353461435162825,0.33135710992020934,0.08893337837319315,0.4367825701200126,1.0,0.19785905649310134,0.1850709291680992,-0.04216295473537674,0.13054795488404794],[0.017683090727830596,0.22107106945898294,0.2818052888499106,0.3925732041590384,0.19785905649310132,1.0,0.1406469525451051,0.03624187009229417,0.29269466264444494],[-0.03352267296261302,0.13733729982837065,0.04126494793009856,0.18392757295416323,0.18507092916809922,0.14064695254510512,1.0,0.033561312434805514,0.1738440656529596],[0.5443412284023393,0.2635143198243338,0.23952794642136366,-0.11397026236774165,-0.04216295473537674,0.03624187009229416,0.03356131243480552,1.0,0.23835598302719754],[0.2218981530339861,0.46658139830687295,0.0650683595503327,0.0747522319183194,0.1305479548840479,0.2926946626444449,0.1738440656529596,0.23835598302719752,1.0]],"type":"heatmap"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"mapa de calor","x":0.5},"width":1300,"height":1000},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a41b33c6-5f60-4c1c-8f9a-da3153423a1d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


Hemos observado que tanto la columna de "insulina" como la de "grosor de la piel" presentan una correlación muy baja con la presencia de diabetes. Además, debido a la gran cantidad de valores inválidos en estas columnas, hemos decidido no utilizarlas en nuestra predicción.

En cambio, para las columnas de "glucosa", "IMC" (BMI) y "presión sanguínea", hemos notado una correlación más significativa con la presencia de diabetes. Para poder aprovechar esta información en nuestra predicción, vamos a rellenar los espacios vacíos con el promedio de cada columna. Esto nos permitirá utilizar de manera más completa y precisa los datos disponibles.


```python
df.Glucose.replace(0,int(df.Glucose.mean()),inplace=True)
df.Insulin.replace(0,df.Insulin.mean(),inplace=True)
df.BMI.replace(0,df.BMI.mean(),inplace=True)

```


```python
drop_columns = ["Outcome","Insulin","SkinThickness"]
df.isin([0]).sum()
```




    Pregnancies                 111
    Glucose                       0
    BloodPressure                35
    SkinThickness               227
    Insulin                       0
    BMI                           0
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                     500
    dtype: int64



Ahora que hemos realizado la limpieza del dataset, es el momento de analizar la distribución de los datos. Este paso nos permitirá obtener una visión más detallada de cómo se distribuyen las variables en nuestro conjunto de datos.


```python
proporcion = df.Outcome.value_counts()
proporcion.index = ['No tiene Diabetes','Tiene Diabetes']
fig = go.Figure(go.Pie(
    labels = proporcion.index,
    values = proporcion.values,
    pull=[0.05, 0],
    marker=dict(colors=['#9560d6','#4490bd'])
))

fig.update_layout(
    title = 'Distribución de casos positivos y negativos',
    title_x= 0.46,

)

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="8a3918b7-d11d-4f39-85dc-8e9a40cbb80e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("8a3918b7-d11d-4f39-85dc-8e9a40cbb80e")) {                    Plotly.newPlot(                        "8a3918b7-d11d-4f39-85dc-8e9a40cbb80e",                        [{"labels":["No tiene Diabetes","Tiene Diabetes"],"marker":{"colors":["#9560d6","#4490bd"]},"pull":[0.05,0],"values":[500,268],"type":"pie"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"Distribuci\u00f3n de casos positivos y negativos","x":0.46}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('8a3918b7-d11d-4f39-85dc-8e9a40cbb80e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


Notamos que hay una mayor cantidad de observaciones de personas sin diabetes en comparación con aquellas que sí la tienen. Esto implica que será más fácil predecir el resultado negativo (ausencia de diabetes) que el positivo (presencia de diabetes) en nuestro modelo.


```python
Cant_por_embarazo = df.groupby(['Outcome','Pregnancies']).size().reset_index(name='cantidad')
Cant_por_embarazo_pos = Cant_por_embarazo[Cant_por_embarazo['Outcome'] == 1 ]
Cant_por_embarazo_neg = Cant_por_embarazo[Cant_por_embarazo['Outcome'] == 0 ]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        y=Cant_por_embarazo_neg['cantidad'],
        x=Cant_por_embarazo_neg['Pregnancies'],
        orientation='v',
        name='Caso Negativo',
        marker=dict(color='#9560d6')
    )
)

fig.add_trace(
    go.Bar(
        y=Cant_por_embarazo_pos['cantidad'],
        x=Cant_por_embarazo_pos['Pregnancies'],
        orientation='v',
        name='Caso Positivo',
        marker=dict(color='#4490bd')
    )
)


fig.update_layout(
    title = 'Distribución de casos según cantidad de embarazos',
    title_x=0.5,
    barmode='group',
    xaxis_title='Cantidad de embarazos',
)

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="b3693602-d6d7-4ef9-ab54-b4dc7e6f87aa" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b3693602-d6d7-4ef9-ab54-b4dc7e6f87aa")) {                    Plotly.newPlot(                        "b3693602-d6d7-4ef9-ab54-b4dc7e6f87aa",                        [{"marker":{"color":"#9560d6"},"name":"Caso Negativo","orientation":"v","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],"y":[73,106,84,48,45,36,34,20,16,10,14,4,5,5],"type":"bar"},{"marker":{"color":"#4490bd"},"name":"Caso Positivo","orientation":"v","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17],"y":[38,29,19,27,23,21,16,25,22,18,10,7,4,5,2,1,1],"type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"Distribuci\u00f3n de casos seg\u00fan cantidad de embarazos","x":0.5},"barmode":"group","xaxis":{"title":{"text":"Cantidad de embarazos"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b3693602-d6d7-4ef9-ab54-b4dc7e6f87aa');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


Al examinar el número de embarazos, se observa que la proporción de casos positivos de diabetes es baja en aquellos con un bajo número de embarazos. Sin embargo, a partir de tres embarazos, la distribución comienza a ser más equitativa, y en la mayoría de los casos con más de siete hijos, la proporción de casos positivos es mayor.


```python
fig = go.Figure()
for label in df['Outcome'].unique():
    if label == 0:
        name = 'Caso Negativo'
        color = '#9560d6'
    else:
        name = 'Caso Positivo'
        color = '#4490bd'


    fig.add_trace(go.Histogram(x=df[df['Outcome'] == label]['Glucose'],
                                xbins=dict(start=30, end=200, size=60),
                                name=name , marker=dict(color=color)))

fig.update_layout(title='Distribución de personas con diabetes según el nivel de glucosa',
                  xaxis_title='Glucosa',
                  yaxis_title='Cantidad',
                  xaxis=dict(ticktext=['Nivel de Glucosa Bajo:30-90 mg/d', 'Nivel de Glucosa medio:90-150mg/dL', 'Nivel de Glucosa alto:150-210mg/dL'],
                             tickvals=[60, 120, 180])
                  )

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="74b0700e-dbf5-4e76-a49b-05d8d7a83738" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("74b0700e-dbf5-4e76-a49b-05d8d7a83738")) {                    Plotly.newPlot(                        "74b0700e-dbf5-4e76-a49b-05d8d7a83738",                        [{"marker":{"color":"#4490bd"},"name":"Caso Positivo","x":[148,183,137,78,197,125,168,189,166,100,118,107,115,196,119,143,125,147,158,102,90,111,171,180,103,176,187,133,114,109,100,126,131,137,136,134,122,163,95,171,155,160,146,124,162,113,88,117,105,173,122,170,108,156,188,152,163,131,104,102,134,179,129,130,194,181,128,109,139,159,135,158,107,109,148,196,162,184,140,112,151,109,85,112,177,158,162,142,134,171,181,179,164,139,119,184,92,113,155,141,123,138,146,101,106,146,161,108,119,107,128,128,146,100,167,144,115,161,128,124,155,109,182,194,112,124,152,122,102,180,115,152,178,120,165,125,129,196,189,146,124,133,173,140,156,116,105,144,166,158,131,193,95,136,168,115,197,172,138,173,144,129,151,184,181,95,141,189,108,117,180,104,134,175,148,120,105,158,145,135,125,195,120,180,84,163,145,130,132,129,100,128,90,186,187,125,198,121,118,197,151,124,143,176,111,132,188,173,150,183,181,174,168,138,112,119,114,104,97,147,167,179,136,155,80,199,167,145,115,145,111,195,156,121,162,125,144,158,129,142,169,125,168,115,164,93,129,187,173,97,149,130,120,174,102,120,140,147,187,162,136,181,154,128,123,190,170,126],"xbins":{"end":200,"size":60,"start":30},"type":"histogram"},{"marker":{"color":"#9560d6"},"name":"Caso Negativo","x":[85,89,116,115,110,139,103,126,99,97,145,117,109,88,92,122,103,138,180,133,106,159,146,71,105,103,101,88,150,73,100,146,105,84,44,141,99,109,95,146,139,129,79,120,62,95,112,113,74,83,101,110,106,100,107,80,123,81,142,144,92,71,93,151,125,81,85,126,96,144,83,89,76,78,97,99,111,107,132,120,118,84,96,125,100,93,129,105,128,106,108,154,102,57,106,147,90,136,114,153,99,109,88,151,102,114,100,148,120,110,111,87,79,75,85,143,87,119,120,73,141,111,123,85,105,113,138,108,99,103,111,96,81,147,179,125,119,142,100,87,101,197,117,79,122,74,104,91,91,146,122,165,124,111,106,129,90,86,111,114,193,191,95,142,96,128,102,108,122,71,106,100,104,114,108,129,133,136,155,96,108,78,161,151,126,112,77,150,120,137,80,106,113,112,99,115,129,112,157,179,105,118,87,106,95,165,117,130,95,120,122,95,126,139,116,99,92,137,61,90,90,88,158,103,147,99,101,81,118,84,105,122,98,87,93,107,105,109,90,125,119,100,100,131,116,127,96,82,137,72,123,101,102,112,143,143,97,83,119,94,102,115,94,135,99,89,80,139,90,140,147,97,107,83,117,100,95,120,82,91,119,100,135,86,134,120,71,74,88,115,124,74,97,154,144,137,119,136,114,137,114,126,132,123,85,84,139,173,99,194,83,89,99,80,166,110,81,154,117,84,94,96,75,130,84,120,139,91,91,99,125,76,129,68,124,114,125,87,97,116,117,111,122,107,86,91,77,105,57,127,84,88,131,164,189,116,84,114,88,84,124,97,110,103,85,87,99,91,95,99,92,154,78,130,111,98,143,119,108,133,109,121,100,93,103,73,112,82,123,67,89,109,108,96,124,124,92,152,111,106,105,106,117,68,112,92,183,94,108,90,125,132,128,94,102,111,128,92,104,94,100,102,128,90,103,157,107,91,117,123,120,106,101,120,127,162,112,98,154,165,99,68,123,91,93,101,56,95,136,129,130,107,140,107,121,90,99,127,118,122,129,110,80,127,158,126,134,102,94,108,83,114,117,111,112,116,141,175,92,106,105,95,126,65,99,102,109,153,100,81,121,108,137,106,88,89,101,122,121,93],"xbins":{"end":200,"size":60,"start":30},"type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"title":{"text":"Glucosa"},"ticktext":["Nivel de Glucosa Bajo:30-90 mg/d","Nivel de Glucosa medio:90-150mg/dL","Nivel de Glucosa alto:150-210mg/dL"],"tickvals":[60,120,180]},"title":{"text":"Distribuci\u00f3n de personas con diabetes seg\u00fan el nivel de glucosa"},"yaxis":{"title":{"text":"Cantidad"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('74b0700e-dbf5-4e76-a49b-05d8d7a83738');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


Podemos observar una fuerte relación entre el nivel de glucosa y la proporción de casos positivos de diabetes. A niveles bajos de glucosa, entre 30-90, los casos positivos representan apenas un 5% de las participantes. Sin embargo, a niveles más altos de glucosa, entre 150-210, la proporción de casos positivos aumenta significativamente, llegando a representar un 74% de los casos. Esto indica que el nivel de glucosa es un factor importante en la predicción de la presencia de diabetes.


```python
fig = go.Figure()
for label in df['Outcome'].unique():
    if label == 0:
        name = 'Caso Negativo'
        color = '#9560d6'
    else:
        name = 'Caso Positivo'
        color = '#4490bd'


    fig.add_trace(go.Histogram(x=df[df['Outcome'] == label]['BloodPressure'],
                                xbins=dict(start=20, end=150, size=20),
                                name=name , marker=dict(color=color)))

fig.update_layout(title='Distribución de personas con diabetes según la presion sanguinea',
                  xaxis_title='Presion Sanguinea',
                  yaxis_title='Cantidad')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="1f409440-a3bb-4521-8193-1a73aab0a5a8" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("1f409440-a3bb-4521-8193-1a73aab0a5a8")) {                    Plotly.newPlot(                        "1f409440-a3bb-4521-8193-1a73aab0a5a8",                        [{"marker":{"color":"#4490bd"},"name":"Caso Positivo","x":[72,64,40,50,70,96,74,60,72,0,84,74,70,90,80,94,70,76,76,76,68,72,110,66,66,90,68,72,66,88,66,90,0,108,70,72,90,72,85,72,62,54,92,74,76,76,30,88,84,70,56,64,66,86,78,88,72,88,74,82,70,72,110,82,68,68,98,76,80,66,0,84,62,64,60,76,104,84,65,82,70,62,74,66,60,90,52,86,80,72,84,90,84,54,50,85,62,64,76,0,62,0,0,86,60,70,86,80,86,62,78,48,70,78,0,58,98,68,68,68,74,80,74,78,74,70,90,64,86,0,76,78,84,80,88,50,0,76,64,78,72,102,82,82,75,74,100,82,76,78,66,70,64,84,64,72,74,68,60,84,82,64,78,78,64,82,0,104,70,62,78,64,70,62,84,0,80,70,0,68,70,70,68,90,72,70,88,70,0,92,74,72,85,90,76,76,66,66,80,70,90,76,66,86,84,80,82,74,78,0,78,58,88,74,82,0,64,72,76,80,74,50,84,52,82,76,106,80,60,82,70,70,86,52,76,80,82,114,68,90,74,78,88,0,78,64,62,50,78,76,68,78,86,88,74,80,94,94,70,62,70,88,78,88,72,92,74,60],"xbins":{"end":150,"size":20,"start":20},"type":"histogram"},{"marker":{"color":"#9560d6"},"name":"Caso Negativo","x":[66,66,74,0,92,80,30,88,84,66,82,92,75,58,92,78,60,76,64,84,92,64,56,70,0,80,50,66,66,50,88,82,64,0,62,58,74,92,66,85,64,86,75,48,78,72,66,44,0,78,65,74,72,68,68,55,80,78,82,72,62,48,50,60,96,72,65,56,122,58,58,76,62,48,60,76,64,74,80,70,58,74,68,60,70,60,80,72,78,82,52,62,75,80,64,78,70,74,65,82,52,56,74,90,74,80,64,66,68,66,90,0,60,64,78,78,80,64,74,60,74,62,70,55,58,80,82,68,70,72,72,64,60,85,95,68,0,60,66,78,76,70,80,80,68,68,76,64,70,76,68,90,70,86,52,84,80,68,56,68,50,68,70,80,74,64,52,62,78,78,70,70,64,74,62,76,88,74,84,56,72,88,50,62,84,72,82,76,76,68,66,70,50,68,80,66,60,75,72,70,70,72,58,80,60,76,0,70,74,68,86,72,88,46,0,62,80,84,82,62,78,74,70,108,74,54,64,86,64,64,58,52,82,60,100,72,68,60,62,70,54,68,66,64,72,58,56,70,61,78,48,62,90,72,84,74,68,68,88,68,64,64,0,94,0,74,74,75,68,85,75,70,88,66,64,72,80,74,64,68,0,54,54,68,74,72,62,70,78,98,56,52,64,78,82,70,66,90,64,84,76,74,86,88,58,82,62,78,72,80,65,90,68,0,74,68,72,74,90,72,64,78,82,60,50,78,62,68,62,54,86,60,90,70,80,0,58,60,64,74,66,65,60,76,66,0,56,90,60,80,90,78,68,82,110,70,68,88,62,64,70,70,76,68,74,68,60,80,54,72,62,72,70,96,58,60,86,44,44,68,60,78,76,56,66,0,78,52,72,76,24,38,88,0,74,60,62,82,62,54,80,72,96,62,86,76,94,70,64,88,68,78,80,65,78,60,82,62,74,76,74,86,70,0,72,74,60,54,60,74,54,70,58,80,106,84,80,58,78,68,58,106,100,82,60,58,56,64,82,74,64,50,74,80,70,60,0,88,70,76,0,76,80,46,64,78,58,74,72,60,86,66,86,94,78,78,84,88,52,56,75,60,86,72,60,44,58,88,84,74,78,62,90,76,58,62,76,70,72,70],"xbins":{"end":150,"size":20,"start":20},"type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"Distribuci\u00f3n de personas con diabetes seg\u00fan la presion sanguinea"},"xaxis":{"title":{"text":"Presion Sanguinea"}},"yaxis":{"title":{"text":"Cantidad"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('1f409440-a3bb-4521-8193-1a73aab0a5a8');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


Cuando la presión sanguínea se encuentra en niveles normales, es decir, por debajo de 80, se observa una proporción relativamente baja de casos positivos de diabetes, aproximadamente un 30%. Sin embargo, es importante tener en cuenta que esta proporción también está influenciada por la distribución de los datos. A medida que la presión sanguínea aumenta, la proporción de casos positivos tiende a incrementarse, llegando a alcanzar el 50% en aquellas participantes con los niveles más elevados de presión sanguínea. Esta tendencia sugiere una posible asociación entre la presión sanguínea elevada y la presencia de diabetes en nuestra muestra de datos.


```python
fig = go.Figure()
for label in df['Outcome'].unique():
    if label == 0:
        name = 'Caso Negativo'
        color = '#9560d6'
    else:
        name = 'Caso Positivo'
        color = '#4490bd'


    fig.add_trace(go.Histogram(x=df[df['Outcome'] == label]['Age'],
                                xbins=dict(start=20, end=90, size=10),
                                name=name , marker=dict(color=color)))

fig.update_layout(title='Distribución de personas con diabetes según la Edad',
                  xaxis_title='Edad',
                  yaxis_title='Cantidad')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="4491a145-b2fa-4301-8ecf-2001169ec108" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4491a145-b2fa-4301-8ecf-2001169ec108")) {                    Plotly.newPlot(                        "4491a145-b2fa-4301-8ecf-2001169ec108",                        [{"marker":{"color":"#4490bd"},"name":"Caso Positivo","x":[50,32,33,26,53,54,34,59,51,32,31,31,32,41,29,51,41,43,28,46,27,56,54,25,31,58,41,39,42,38,28,42,26,37,43,60,31,33,24,24,46,39,61,38,25,23,26,40,62,33,33,30,42,42,43,36,47,32,41,36,29,36,26,37,41,60,33,31,25,36,40,29,23,26,29,57,52,41,24,36,38,25,32,41,21,66,24,22,46,26,51,23,32,22,33,49,44,21,51,27,35,25,28,38,29,28,47,52,29,25,31,24,29,46,30,25,28,47,25,30,27,43,29,59,25,36,43,30,23,41,44,33,41,37,49,28,44,29,29,67,29,45,25,58,32,35,45,58,27,31,22,25,31,35,41,46,39,28,21,22,37,28,36,31,38,43,29,41,33,30,25,22,23,38,51,38,29,35,31,24,45,55,41,35,46,28,53,45,23,32,43,27,56,37,53,54,28,33,21,62,21,52,41,52,45,44,22,38,54,36,22,36,40,50,50,24,34,38,32,50,33,22,42,25,27,22,43,40,40,70,40,31,53,25,26,27,46,44,43,43,31,49,52,30,45,23,38,34,31,52,42,34,22,24,42,48,45,27,36,50,22,26,45,37,52,66,43,47],"xbins":{"end":90,"size":10,"start":20},"type":"histogram"},{"marker":{"color":"#9560d6"},"name":"Caso Negativo","x":[31,21,30,29,30,57,33,27,50,22,57,38,60,22,28,45,33,35,26,37,48,40,29,22,24,22,26,30,42,21,31,44,22,21,36,24,32,54,25,27,26,23,22,22,41,27,24,22,22,36,22,27,45,26,24,21,34,42,21,40,24,22,23,22,21,24,27,21,27,37,25,23,25,25,22,21,24,23,69,30,23,39,26,31,21,22,29,28,55,38,22,23,21,41,34,65,22,24,37,23,21,23,22,36,45,27,21,22,34,29,29,25,23,33,42,47,32,23,21,27,40,21,40,42,21,21,28,32,27,55,27,21,25,24,60,32,37,61,26,22,26,31,24,22,29,23,27,21,22,29,41,23,34,23,42,27,24,25,30,25,24,34,24,63,43,24,21,21,40,21,52,25,23,57,22,39,37,51,34,26,33,21,65,28,24,58,35,37,29,21,41,22,25,26,30,28,31,21,24,37,37,46,25,44,22,26,44,22,36,22,33,57,49,22,23,26,29,30,46,24,21,48,63,65,30,30,21,22,21,21,25,28,22,22,35,24,22,21,25,25,24,28,42,21,37,25,39,25,55,38,26,25,28,25,22,21,22,27,26,21,21,21,25,26,23,38,22,29,36,41,28,21,31,22,24,28,26,26,23,25,72,24,62,24,81,48,26,39,37,34,21,22,25,27,28,22,22,50,24,59,31,39,63,29,28,23,21,58,28,67,24,42,33,22,66,30,25,39,21,28,41,40,38,21,21,64,21,58,22,24,51,41,60,25,26,26,24,21,21,24,22,31,22,24,29,31,24,46,67,23,25,29,28,50,37,21,25,66,23,28,37,30,58,42,35,24,32,27,22,21,46,37,39,21,22,22,23,25,35,36,27,62,42,22,29,25,24,25,34,46,21,26,24,28,30,21,25,27,23,24,26,27,30,23,28,28,45,21,21,29,21,21,45,21,24,23,22,31,48,23,28,27,24,31,27,30,23,23,27,28,27,22,22,41,51,54,24,43,45,49,21,47,22,68,25,23,22,22,69,25,22,29,23,34,23,25,22,28,26,26,41,27,28,22,24,40,21,32,56,24,34,21,42,45,38,25,22,22,22,22,53,28,21,42,21,26,22,39,46,32,28,25,39,26,22,33,63,27,30,23],"xbins":{"end":90,"size":10,"start":20},"type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"Distribuci\u00f3n de personas con diabetes seg\u00fan la Edad"},"xaxis":{"title":{"text":"Edad"}},"yaxis":{"title":{"text":"Cantidad"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4491a145-b2fa-4301-8ecf-2001169ec108');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


Se observa que en el rango de edad comprendido entre los 20 y 30 años, la proporción de casos positivos de diabetes es relativamente baja, representando apenas un 20% de los casos. Sin embargo, a medida que la edad aumenta, también lo hace el porcentaje de casos positivos. En el grupo de edad de 50 a 60 años, la proporción de casos positivos alcanza un 60%. Este incremento sugiere una asociación entre la edad y la probabilidad de tener diabetes, ya que el riesgo de desarrollarla tiende a aumentar con el envejecimiento.


```python
fig = go.Figure()
for label in df['Outcome'].unique():
    if label == 0:
        name = 'Caso Negativo'
        color = '#9560d6'
    else:
        name = 'Caso Positivo'
        color = '#4490bd'


    fig.add_trace(go.Histogram(x=df[df['Outcome'] == label]['DiabetesPedigreeFunction'],
                                xbins=dict(start=0, end=1, size=0.1),
                                name=name , marker=dict(color=color)))

fig.update_layout(title='Distribución de personas con diabetes según la Función Pedigrí de la Diabetes,',
                  xaxis_title='Función Pedigrí de la Diabetes',
                  yaxis_title='Cantidad')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="52ef8dc5-0ea2-4cc3-b535-fb6f98fda8c5" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("52ef8dc5-0ea2-4cc3-b535-fb6f98fda8c5")) {                    Plotly.newPlot(                        "52ef8dc5-0ea2-4cc3-b535-fb6f98fda8c5",                        [{"marker":{"color":"#4490bd"},"name":"Caso Positivo","x":[0.627,0.672,2.288,0.248,0.158,0.232,0.537,0.398,0.587,0.484,0.551,0.254,0.529,0.451,0.263,0.254,0.205,0.257,0.851,0.665,0.503,1.39,0.721,1.893,0.344,0.467,0.254,0.27,0.258,0.855,0.867,0.583,0.27,0.227,0.153,0.277,0.325,1.222,0.247,0.199,0.543,0.588,0.539,0.22,0.759,0.278,0.496,0.403,0.741,0.361,1.114,0.356,0.272,1.189,0.137,0.337,0.817,0.743,0.722,0.18,0.542,0.719,0.319,0.956,0.745,0.615,1.321,0.64,0.361,0.383,0.578,0.395,0.678,0.905,0.15,0.605,0.151,0.355,0.431,0.26,0.742,0.514,1.224,0.261,1.072,0.805,0.652,0.645,0.238,0.479,0.586,0.686,0.831,0.402,1.318,1.213,0.926,0.543,1.353,0.761,0.226,0.933,0.24,1.136,0.296,0.334,0.165,0.259,0.808,0.757,1.224,0.613,0.337,0.412,0.839,0.422,0.209,0.326,1.391,0.875,0.433,1.127,0.345,0.129,0.197,0.254,0.731,0.692,0.127,0.282,0.343,0.893,0.331,0.346,0.302,0.962,0.569,0.875,0.583,0.52,0.368,0.234,2.137,0.528,0.238,0.66,0.239,0.452,0.34,0.803,0.196,0.241,0.161,0.286,0.135,0.376,1.191,0.702,0.534,0.258,0.554,0.219,0.516,0.264,0.328,0.233,0.205,0.435,0.955,0.38,2.42,0.51,0.542,0.212,1.001,0.183,0.711,0.344,0.63,0.365,1.144,0.163,0.727,0.314,0.297,0.268,0.771,0.652,0.302,0.968,0.661,0.549,0.825,0.423,1.034,0.121,0.502,0.203,0.693,0.575,0.371,0.687,0.129,1.154,0.925,0.402,0.682,0.088,0.692,0.212,1.258,0.593,0.787,0.557,1.282,0.141,0.732,0.465,0.871,0.178,0.447,0.455,0.26,0.24,1.292,1.394,0.165,0.637,0.245,0.235,0.141,0.328,0.23,0.127,0.364,0.536,0.335,0.257,0.439,0.128,0.268,0.565,0.905,0.261,0.148,0.674,0.441,0.826,0.97,0.378,0.349,0.323,0.259,0.646,0.293,0.785,0.734,0.358,0.408,0.178,1.182,0.222,0.443,1.057,0.258,0.278,0.403,0.349],"xbins":{"end":1,"size":0.1,"start":0},"type":"histogram"},{"marker":{"color":"#9560d6"},"name":"Caso Negativo","x":[0.351,0.167,0.201,0.134,0.191,1.441,0.183,0.704,0.388,0.487,0.245,0.337,0.546,0.267,0.188,0.512,0.966,0.42,0.271,0.696,0.235,0.294,0.564,0.586,0.305,0.491,0.526,0.342,0.718,0.248,0.962,1.781,0.173,0.304,0.587,0.699,0.203,0.845,0.334,0.189,0.411,0.231,0.396,0.14,0.391,0.37,0.307,0.14,0.102,0.767,0.237,0.698,0.178,0.324,0.165,0.258,0.443,0.261,0.761,0.255,0.13,0.323,0.356,0.179,0.262,0.283,0.93,0.801,0.207,0.287,0.336,0.192,0.391,0.654,0.443,0.223,0.26,0.404,0.186,0.452,0.261,0.457,0.647,0.088,0.597,0.532,0.703,0.159,0.268,0.286,0.318,0.237,0.572,0.096,1.4,0.218,0.085,0.399,0.432,0.687,0.637,0.833,0.229,0.294,0.204,0.167,0.368,0.256,0.709,0.471,0.495,0.773,0.678,0.37,0.382,0.19,0.084,0.725,0.299,0.268,0.244,0.142,0.374,0.136,0.187,0.874,0.236,0.787,0.235,0.324,0.407,0.289,0.29,0.375,0.164,0.464,0.209,0.687,0.666,0.101,0.198,2.329,0.089,0.583,0.394,0.293,0.582,0.192,0.446,0.329,0.258,0.427,0.282,0.143,0.38,0.284,0.249,0.238,0.557,0.092,0.655,0.299,0.612,0.2,0.997,1.101,0.078,0.128,0.254,0.422,0.251,0.677,0.454,0.744,0.881,0.28,0.262,0.647,0.619,0.34,0.263,0.434,0.254,0.692,0.52,0.84,0.156,0.207,0.215,0.143,0.313,0.605,0.626,0.315,0.284,0.15,0.527,0.148,0.123,0.2,0.122,1.476,0.166,0.137,0.26,0.259,0.932,0.472,0.673,0.389,0.29,0.485,0.349,0.654,0.187,0.279,0.237,0.252,0.243,0.58,0.559,0.378,0.207,0.305,0.385,0.499,0.252,0.306,1.731,0.545,0.225,0.816,0.299,0.509,1.021,0.821,0.236,0.947,1.268,0.221,0.205,0.949,0.444,0.389,0.463,1.6,0.944,0.389,0.151,0.28,0.52,0.336,0.674,0.528,1.076,0.256,1.095,0.624,0.507,0.561,0.496,0.421,0.256,0.284,0.108,0.551,0.527,0.167,1.138,0.244,0.434,0.147,0.727,0.497,0.23,0.658,0.33,0.285,0.415,0.381,0.832,0.498,0.687,0.364,0.46,0.733,0.416,0.705,0.258,1.022,0.452,0.269,0.6,0.571,0.607,0.17,0.259,0.21,0.126,0.231,0.466,0.162,0.419,0.197,0.306,0.233,0.536,1.159,0.294,0.551,0.629,0.292,0.145,0.174,0.304,0.292,0.547,0.839,0.313,0.267,0.738,0.238,0.263,0.692,0.968,0.409,0.207,0.2,0.525,0.154,0.304,0.18,0.582,0.187,0.305,0.189,0.151,0.444,0.299,0.107,0.493,0.66,0.717,0.686,0.917,0.501,1.251,0.197,0.735,0.804,0.159,0.365,0.16,0.341,0.68,0.204,0.591,0.247,0.422,0.471,0.161,0.218,0.237,0.126,0.3,0.401,0.497,0.601,0.748,0.412,0.085,0.338,0.27,0.268,0.43,0.198,0.892,0.28,0.813,0.245,0.206,0.259,0.19,0.417,0.249,0.342,0.175,1.699,0.733,0.194,0.559,0.407,0.4,0.19,0.1,0.514,0.482,0.27,0.138,0.292,0.878,0.207,0.157,0.257,0.246,1.698,1.461,0.347,0.158,0.362,0.206,0.393,0.144,0.148,0.238,0.343,0.115,0.167,0.153,0.649,0.149,0.695,0.303,0.61,0.73,0.134,0.133,0.234,0.466,0.269,0.455,0.142,0.155,1.162,0.19,0.182,0.217,0.43,0.164,0.631,0.551,0.285,0.88,0.587,0.263,0.614,0.332,0.366,0.64,0.591,0.314,0.181,0.828,0.856,0.886,0.191,0.253,0.598,0.904,0.483,0.304,0.118,0.177,0.176,0.295,0.439,0.352,0.121,0.595,0.415,0.317,0.289,0.251,0.265,0.236,0.496,0.433,0.326,0.141,0.426,0.56,0.284,0.515,0.6,0.453,0.4,0.219,1.174,0.488,1.096,0.261,0.223,0.391,0.197,0.766,0.142,0.171,0.34,0.245,0.315],"xbins":{"end":1,"size":0.1,"start":0},"type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"Distribuci\u00f3n de personas con diabetes seg\u00fan la Funci\u00f3n Pedigr\u00ed de la Diabetes,"},"xaxis":{"title":{"text":"Funci\u00f3n Pedigr\u00ed de la Diabetes"}},"yaxis":{"title":{"text":"Cantidad"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('52ef8dc5-0ea2-4cc3-b535-fb6f98fda8c5');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


La función pedigrí de la diabetes nos permite estimar el riesgo de desarrollar diabetes en función del historial familiar, vemos que los datos que nos brinda en general coinciden con la información tomada, pero no son definitorias, ya que tenemos un 50% de casos con diabetes incluso cuando la función nos otorga los valores más altos.

En resumen, al examinar todas las variables, se observa que todas ellas están relacionadas de alguna manera con la diabetes. Algunas variables, como los niveles de glucosa en sangre, son especialmente precisas para determinar la presencia de diabetes. Sin embargo, todas las variables aportan información útil en conjunto. Aunque ninguna variable por sí sola ofrece una precisión total, utilizaremos todas estas variables en nuestra predicción para aprovechar al máximo la información disponible y mejorar la precisión de nuestro modelo.

# Creando los modelos predictivos

## Dividiendo el dataset


```python
data = df.drop(columns=drop_columns)
scaler=StandardScaler()
scaler.fit(data)
x=scaler.fit_transform(data)
y = df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15)
```

Para evitar el riesgo de sobreajuste, divido mi conjunto de datos en un 85% para entrenamiento y un 15% para evaluación. Esta división me permite evaluar la capacidad de los modelos para generalizar en datos desconocidos durante la prueba, en lugar de simplemente memorizar los datos de entrenamiento.

Además, como mencioné anteriormente, es necesario estandarizar los datos para mejorar el rendimiento del predictor. Por lo tanto, realicé la estandarización de los datos. Al estandarizar los datos, se logra una mejor performance del predictor al eliminar las diferencias de escala entre las variables y facilitar el entrenamiento del modelo. Esto asegura que las diferencias en las unidades o rangos de los datos no afecten negativamente la precisión del modelo.

## Comparando los modelos


```python
# logistic regresion
lm = LogisticRegression()
lm.fit(x_train,y_train)
lm_y_predictions=lm.predict(x_test)
lm_result = classification_report(y_test, lm_y_predictions, output_dict=True)


# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_predictions = rfc.predict(x_test)
rfc_result = classification_report(y_test, rfc_y_predictions, output_dict=True)

# Support Vector Machines
svm = SVC(probability=True)
svm.fit(x_train, y_train)
svm_y_predictions=svm.predict(x_test)
svm_result = classification_report(y_test, svm_y_predictions, output_dict=True)


# KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_y_predictions = knn.predict(x_test)
knn_result = classification_report(y_test, knn_y_predictions, output_dict=True)

result = pd.DataFrame(columns=['type','accuracy','precision_neg','precision_pos'])
new_rows = []

new_row_lm = {'type': 'logistic regresion', 'accuracy': lm_result['accuracy'], 'precision_neg': lm_result['0']['precision'], 'precision_pos': lm_result['1']['precision']}
new_rows.append(new_row_lm)

new_row_rfc = {'type': 'Random Forest Classifier', 'accuracy': rfc_result['accuracy'], 'precision_neg': rfc_result['0']['precision'], 'precision_pos': rfc_result['1']['precision']}
new_rows.append(new_row_rfc)

new_row_svm = {'type': 'Support Vector Machines', 'accuracy': svm_result['accuracy'], 'precision_neg': svm_result['0']['precision'], 'precision_pos': svm_result['1']['precision']}
new_rows.append(new_row_svm)

new_row_knn = {'type': 'KNeighborsClassifier', 'accuracy': knn_result['accuracy'], 'precision_neg': knn_result['0']['precision'], 'precision_pos': knn_result['1']['precision']}
new_rows.append(new_row_knn)

result = result.append(new_rows, ignore_index=True)

result

```





  <div id="df-c8f4de96-8790-4f52-99e7-625cf3fa9894">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>accuracy</th>
      <th>precision_neg</th>
      <th>precision_pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>logistic regresion</td>
      <td>0.775862</td>
      <td>0.810127</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest Classifier</td>
      <td>0.758621</td>
      <td>0.830986</td>
      <td>0.644444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Support Vector Machines</td>
      <td>0.775862</td>
      <td>0.795181</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.750000</td>
      <td>0.802632</td>
      <td>0.650000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c8f4de96-8790-4f52-99e7-625cf3fa9894')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c8f4de96-8790-4f52-99e7-625cf3fa9894 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c8f4de96-8790-4f52-99e7-625cf3fa9894');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




En general, se pudo observar que todos los modelos utilizados presentaron tasas de éxito similares en la predicción de los resultados, con una precisión aproximada del 78%.

Dado el desempeño ligeramente superior del modelo de Support Vector Machines, se tomará la decisión de guardarlo para su uso posterior. Al guardar el modelo, se evita la necesidad de volver a entrenarlo cada vez que se desee utilizar, lo que ahorra tiempo y recursos. Esta práctica es especialmente beneficiosa cuando se trabaja con grandes conjuntos de datos o se requiere un proceso de predicción rápido y eficiente.


```python
import joblib
from google.colab import files
from sklearn.calibration import CalibratedClassifierCV

calibrated_svc = CalibratedClassifierCV(svm, cv='prefit')
calibrated_svc.fit(x_train, y_train)


# Save the model for future use
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(calibrated_svc, 'calibrated_svc.pkl')

# Download the saved models
files.download('svm_model.pkl')
files.download('calibrated_svc.pkl')

```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


## Conclusión

Esto señala la existencia de varias afecciones médicas que pueden servir como indicadores para diagnosticar la posible presencia de diabetes, lo que a su vez permite tomar medidas preventivas para reducir las posibilidades de desarrollar esta enfermedad. Estos hallazgos resaltan la importancia de detectar tempranamente y monitorear las afecciones médicas relacionadas con la diabetes para una intervención oportuna y la adopción de medidas preventivas adecuadas.

Aunque la tasa de éxito no es baja, considerando el tamaño de nuestro conjunto de datos, podríamos mejorarla si contáramos con un mayor número de pacientes en los cuales basar nuestros modelos. Es importante tener en cuenta que no se puede garantizar el éxito del modelo para personas que no cumplan con las características de los pacientes incluidos en el conjunto de datos utilizado.


```python
!jupyter nbconvert --to html analysis_final.ipynb
```

    [NbConvertApp] Converting notebook analysis_final.ipynb to html
    /usr/local/lib/python3.10/dist-packages/nbconvert/filters/widgetsdatatypefilter.py:71: UserWarning: Your element with mimetype(s) dict_keys(['application/vnd.plotly.v1+json']) is not able to be represented.
      warn(
    [NbConvertApp] Writing 649100 bytes to analysis_final.html

