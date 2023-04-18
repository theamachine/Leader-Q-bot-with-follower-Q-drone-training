# Leader-Q-bot-with-follower-Q-drone-training
Qbot-Qdrone leader/follower model using 4 different unsupervised machine learning models:
1. SVM (binary)
2. KNN (unsupervised implementation)
3. PCA
4. DBSCAN  

The following dependencies should be installed:
!pip install pycaret[full]
!pip install markupsafe==2.0.1
!pip uninstall jinja2 --yes
!pip install jinja2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
import pandas as pd
from pycaret.anomaly import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.cluster import  DBSCAN
sns.reset_defaults()
