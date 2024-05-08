**This part mainly looks at the balance of the data set and the effect of up and down sampling on the predictive performance of the model**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import LabelBinarizer

```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
data = pd.read_csv('/content/drive/MyDrive/fetal_health.csv')
data.head()
```





  <div id="df-d008c6b9-2dd8-4144-b1ed-b3624b315158" class="colab-df-container">
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
      <th>baseline value</th>
      <th>accelerations</th>
      <th>fetal_movement</th>
      <th>uterine_contractions</th>
      <th>light_decelerations</th>
      <th>severe_decelerations</th>
      <th>prolongued_decelerations</th>
      <th>abnormal_short_term_variability</th>
      <th>mean_value_of_short_term_variability</th>
      <th>percentage_of_time_with_abnormal_long_term_variability</th>
      <th>...</th>
      <th>histogram_min</th>
      <th>histogram_max</th>
      <th>histogram_number_of_peaks</th>
      <th>histogram_number_of_zeroes</th>
      <th>histogram_mode</th>
      <th>histogram_mean</th>
      <th>histogram_median</th>
      <th>histogram_variance</th>
      <th>histogram_tendency</th>
      <th>fetal_health</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>73.0</td>
      <td>0.5</td>
      <td>43.0</td>
      <td>...</td>
      <td>62.0</td>
      <td>126.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>137.0</td>
      <td>121.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132.0</td>
      <td>0.006</td>
      <td>0.0</td>
      <td>0.006</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>...</td>
      <td>68.0</td>
      <td>198.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>141.0</td>
      <td>136.0</td>
      <td>140.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>133.0</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.008</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>...</td>
      <td>68.0</td>
      <td>198.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>141.0</td>
      <td>135.0</td>
      <td>138.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>134.0</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.008</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>170.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>134.0</td>
      <td>137.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132.0</td>
      <td>0.007</td>
      <td>0.0</td>
      <td>0.008</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>170.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>136.0</td>
      <td>138.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d008c6b9-2dd8-4144-b1ed-b3624b315158')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-d008c6b9-2dd8-4144-b1ed-b3624b315158 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d008c6b9-2dd8-4144-b1ed-b3624b315158');
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


<div id="df-376cd6d8-196a-43c9-9a6f-78ab4a9e0d34">
  <button class="colab-df-quickchart" onclick="quickchart('df-376cd6d8-196a-43c9-9a6f-78ab4a9e0d34')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-376cd6d8-196a-43c9-9a6f-78ab4a9e0d34 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Check the distribution of the target variable 'sepsis'
fetal_distribution = data['fetal_health'].value_counts(normalize=True)
fetal_distribution
```




    fetal_health
    1.0    0.778457
    2.0    0.138758
    3.0    0.082785
    Name: proportion, dtype: float64




```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Trying to reload the dataset and apply SMOTE
try:

    # Separating features and target
    X = data.drop('fetal_health', axis=1)
    y = data['fetal_health']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Applying SMOTE for upsampling
    smote = SMOTE(random_state=42)
    X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)

    # Checking the distribution after SMOTE
    upsample_distribution = y_train_upsampled.value_counts(normalize=True)
    upsample_result = upsample_distribution
except Exception as e:
    upsample_result = str(e)

upsample_result

# Check the new class distribution
print(y_train_upsampled.value_counts())
```

    fetal_health
    3.0    1323
    1.0    1323
    2.0    1323
    Name: count, dtype: int64
    


```python
upsampled_data = pd.concat([X_train_upsampled, y_train_upsampled], axis=1)
upsampled_data
```





  <div id="df-85737139-0a3a-488b-a92e-f5a0f3b5288f" class="colab-df-container">
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
      <th>baseline value</th>
      <th>accelerations</th>
      <th>fetal_movement</th>
      <th>uterine_contractions</th>
      <th>light_decelerations</th>
      <th>severe_decelerations</th>
      <th>prolongued_decelerations</th>
      <th>abnormal_short_term_variability</th>
      <th>mean_value_of_short_term_variability</th>
      <th>percentage_of_time_with_abnormal_long_term_variability</th>
      <th>...</th>
      <th>histogram_min</th>
      <th>histogram_max</th>
      <th>histogram_number_of_peaks</th>
      <th>histogram_number_of_zeroes</th>
      <th>histogram_mode</th>
      <th>histogram_mean</th>
      <th>histogram_median</th>
      <th>histogram_variance</th>
      <th>histogram_tendency</th>
      <th>fetal_health</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>133.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012000</td>
      <td>0.001000</td>
      <td>0.000000</td>
      <td>0.002000</td>
      <td>60.000000</td>
      <td>3.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>58.000000</td>
      <td>155.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>125.000000</td>
      <td>96.000000</td>
      <td>105.000000</td>
      <td>79.000000</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>146.000000</td>
      <td>0.006000</td>
      <td>0.000000</td>
      <td>0.003000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>126.000000</td>
      <td>175.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>150.000000</td>
      <td>152.000000</td>
      <td>153.000000</td>
      <td>5.000000</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>129.000000</td>
      <td>0.000000</td>
      <td>0.001000</td>
      <td>0.007000</td>
      <td>0.006000</td>
      <td>0.000000</td>
      <td>0.002000</td>
      <td>67.000000</td>
      <td>3.200000</td>
      <td>0.0</td>
      <td>...</td>
      <td>66.000000</td>
      <td>146.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>105.000000</td>
      <td>80.000000</td>
      <td>107.000000</td>
      <td>9.000000</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>134.000000</td>
      <td>0.008000</td>
      <td>0.001000</td>
      <td>0.010000</td>
      <td>0.006000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>61.000000</td>
      <td>1.100000</td>
      <td>0.0</td>
      <td>...</td>
      <td>80.000000</td>
      <td>189.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>156.000000</td>
      <td>144.000000</td>
      <td>151.000000</td>
      <td>61.000000</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>125.000000</td>
      <td>0.000000</td>
      <td>0.005000</td>
      <td>0.001000</td>
      <td>0.001000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>70.000000</td>
      <td>0.400000</td>
      <td>29.0</td>
      <td>...</td>
      <td>52.000000</td>
      <td>133.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>125.000000</td>
      <td>123.000000</td>
      <td>125.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3964</th>
      <td>129.000000</td>
      <td>0.000000</td>
      <td>0.001000</td>
      <td>0.005932</td>
      <td>0.008000</td>
      <td>0.000000</td>
      <td>0.002000</td>
      <td>65.000000</td>
      <td>2.806767</td>
      <td>0.0</td>
      <td>...</td>
      <td>50.000000</td>
      <td>151.000000</td>
      <td>6.932332</td>
      <td>2.000000</td>
      <td>105.000000</td>
      <td>85.864663</td>
      <td>111.932332</td>
      <td>12.932332</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3965</th>
      <td>131.168660</td>
      <td>0.001000</td>
      <td>0.000708</td>
      <td>0.012292</td>
      <td>0.009416</td>
      <td>0.000000</td>
      <td>0.001584</td>
      <td>52.358558</td>
      <td>3.397876</td>
      <td>0.0</td>
      <td>...</td>
      <td>50.000000</td>
      <td>207.786186</td>
      <td>8.123505</td>
      <td>0.876495</td>
      <td>63.213814</td>
      <td>94.258968</td>
      <td>110.954846</td>
      <td>239.774228</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3966</th>
      <td>110.000000</td>
      <td>0.002913</td>
      <td>0.001631</td>
      <td>0.005195</td>
      <td>0.008087</td>
      <td>0.000000</td>
      <td>0.002000</td>
      <td>68.000000</td>
      <td>3.108708</td>
      <td>0.0</td>
      <td>...</td>
      <td>61.630615</td>
      <td>189.738771</td>
      <td>6.456462</td>
      <td>0.543538</td>
      <td>91.000000</td>
      <td>82.108156</td>
      <td>94.456462</td>
      <td>39.912924</td>
      <td>-1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3967</th>
      <td>131.245176</td>
      <td>0.000409</td>
      <td>0.307785</td>
      <td>0.002918</td>
      <td>0.003000</td>
      <td>0.000000</td>
      <td>0.002082</td>
      <td>36.288310</td>
      <td>2.142792</td>
      <td>0.0</td>
      <td>...</td>
      <td>55.389331</td>
      <td>182.163451</td>
      <td>9.509648</td>
      <td>0.918275</td>
      <td>76.144155</td>
      <td>100.144155</td>
      <td>102.653803</td>
      <td>148.000000</td>
      <td>-1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3968</th>
      <td>124.791896</td>
      <td>0.000000</td>
      <td>0.015626</td>
      <td>0.003542</td>
      <td>0.007208</td>
      <td>0.000458</td>
      <td>0.002167</td>
      <td>50.207320</td>
      <td>2.370850</td>
      <td>0.0</td>
      <td>...</td>
      <td>54.458301</td>
      <td>166.208889</td>
      <td>6.166797</td>
      <td>1.541699</td>
      <td>71.333595</td>
      <td>86.416994</td>
      <td>82.666405</td>
      <td>99.292288</td>
      <td>-1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>3969 rows × 22 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-85737139-0a3a-488b-a92e-f5a0f3b5288f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-85737139-0a3a-488b-a92e-f5a0f3b5288f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-85737139-0a3a-488b-a92e-f5a0f3b5288f');
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


<div id="df-c8d52ec3-793a-47e3-a3ed-42d2c171af26">
  <button class="colab-df-quickchart" onclick="quickchart('df-c8d52ec3-793a-47e3-a3ed-42d2c171af26')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c8d52ec3-793a-47e3-a3ed-42d2c171af26 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_f697ebdd-3a27-4936-89da-fb90a68529cf">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('upsampled_data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_f697ebdd-3a27-4936-89da-fb90a68529cf button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('upsampled_data');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro'),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo', average='macro')
    }
    return scores

```

### original data GBoost matrix


```python
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
scores_without_smote = evaluate_model(model, X_test, y_test)
print("Performance without SMOTE:", scores_without_smote)

```

    Performance without SMOTE: {'Accuracy': 0.9272300469483568, 'Recall': 0.8550477940818966, 'Precision': 0.880127723063821, 'F1 Score': 0.8637701463903719, 'Kappa': 0.7941740309533829, 'AUC': 0.9545725273977265}
    

### with smote GB(upsampling)


```python
smote_pipeline = make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier(random_state=42))
smote_pipeline.fit(X_train, y_train)
scores_with_smote = evaluate_model(smote_pipeline, X_test, y_test)
print("Performance with SMOTE:", scores_with_smote)

```

    Performance with SMOTE: {'Accuracy': 0.9295774647887324, 'Recall': 0.8839260188453572, 'Precision': 0.8805591521603606, 'F1 Score': 0.8822236586942469, 'Kappa': 0.8087370358730301, 'AUC': 0.9646117156276436}
    


```python
print("Performance Comparison:\n")
print("Metrics Without SMOTE:")
for key, value in scores_without_smote.items():
    print(f"{key}: {value:.4f}")

print("\nMetrics With SMOTE:")
for key, value in scores_with_smote.items():
    print(f"{key}: {value:.4f}")

```

    Performance Comparison:
    
    Metrics Without SMOTE:
    Accuracy: 0.9272
    Recall: 0.8550
    Precision: 0.8801
    F1 Score: 0.8638
    Kappa: 0.7942
    AUC: 0.9546
    
    Metrics With SMOTE:
    Accuracy: 0.9296
    Recall: 0.8839
    Precision: 0.8806
    F1 Score: 0.8822
    Kappa: 0.8087
    AUC: 0.9646
    

### Random Forest


```python
def train_evaluate_model(X_train, y_train, X_test, y_test):
    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)

    # AUC
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    y_pred_bin = lb.transform(y_pred)
    auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro', multi_class='ovr')

    return {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Kappa': kappa,
        'AUC': auc
    }

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

```


```python
# Evaluate without SMOTE
results_without_smote = train_evaluate_model(X_train, y_train, X_test, y_test)

# Apply SMOTE
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# Evaluate with SMOTE
results_with_smote = train_evaluate_model(X_train_smote, y_train_smote, X_test, y_test)

# Print results
print("Original dataset:")
for key, value in results_without_smote.items():
    print(f"{key}: {value:.4f}")

print("\nResults with SMOTE:")
for key, value in results_with_smote.items():
    print(f"{key}: {value:.4f}")

print("\nResults with Downsampling:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

```

    Original dataset:
    Accuracy: 0.9249
    Recall: 0.8285
    Precision: 0.8770
    F1 Score: 0.8499
    Kappa: 0.7845
    AUC: 0.9787
    
    Results with SMOTE:
    Accuracy: 0.9296
    Recall: 0.8793
    Precision: 0.8718
    F1 Score: 0.8755
    Kappa: 0.8088
    AUC: 0.9817
    
    Results with Downsampling:
    Accuracy: 0.8850
    Recall: 0.8827
    Precision: 0.7904
    F1 Score: 0.8283
    Kappa: 0.7192
    AUC: 0.9623
    

Model Results with Downsampling:

Accuracy: 0.8850

Recall: 0.8827

Precision: 0.7904

F1 Score: 0.8283

Kappa: 0.7192

AUC: 0.9623

## down sampling


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score, classification_report
from imblearn.under_sampling import RandomUnderSampler

```


```python
def apply_downsampling(X_train, y_train):
    # Define the downsampling method
    rus = RandomUnderSampler(random_state=42)

    # Apply downsampling
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    return X_train_res, y_train_res

# Apply downsampling
X_train_res, y_train_res = apply_downsampling(X_train, y_train)

```


```python
def train_evaluate_model(X_train, y_train, X_test, y_test):
    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='macro')

    return {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Kappa': kappa,
        'AUC': auc
    }

```


```python
# Evaluate the model
results = train_evaluate_model(X_train_res, y_train_res, X_test, y_test)

# Print results
print("Model Results with Downsampling:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

```

    Model Results with Downsampling:
    Accuracy: 0.8850
    Recall: 0.8827
    Precision: 0.7904
    F1 Score: 0.8283
    Kappa: 0.7192
    AUC: 0.9623
    

### Original data with Logistic Regression, Decision Tree, KNN and SVC.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
scores_lr = evaluate_model(lr_model, X_test, y_test)
print("Performance without SMOTE (Logistic Regression):", scores_lr)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
scores_dt = evaluate_model(dt_model, X_test, y_test)
print("Performance without SMOTE (Decision Tree):", scores_dt)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
scores_knn = evaluate_model(knn_model, X_test, y_test)
print("Performance without SMOTE (K-Nearest Neighbors):", scores_knn)

# Support Vector Classifier
svc_model = SVC(probability=True, random_state=42)
svc_model.fit(X_train, y_train)
scores_svc = evaluate_model(svc_model, X_test, y_test)
print("Performance without SMOTE (SVC):", scores_svc)

```

    /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Performance without SMOTE (Logistic Regression): {'Accuracy': 0.8474178403755869, 'Recall': 0.6675870552427629, 'Precision': 0.7419419125093927, 'F1 Score': 0.6966582001095672, 'Kappa': 0.5332726537216829, 'AUC': 0.8692073378258799}
    Performance without SMOTE (Decision Tree): {'Accuracy': 0.9014084507042254, 'Recall': 0.8091528340966773, 'Precision': 0.8247315362699977, 'F1 Score': 0.8159515832839358, 'Kappa': 0.7244501940491591, 'AUC': 0.856864625572508}
    Performance without SMOTE (K-Nearest Neighbors): {'Accuracy': 0.8943661971830986, 'Recall': 0.749604714256542, 'Precision': 0.8299732341519085, 'F1 Score': 0.7844259290883834, 'Kappa': 0.6920927094877849, 'AUC': 0.9042033752443187}
    Performance without SMOTE (SVC): {'Accuracy': 0.8544600938967136, 'Recall': 0.63883475791787, 'Precision': 0.7578114417914167, 'F1 Score': 0.6837606837606837, 'Kappa': 0.5554208958238651, 'AUC': 0.8897697326837616}
    
