**Title: Movie parental guide predictive profiling from raw movie script using AI for automated viewability classification**

**Introduction:** Entertainment industry is one of the fastest growing industries in the current age of globalization where content is no longer restricted to regional or language boundaries. Though there are many positive sides to the growth there few negative implications as well of which one of the major concerns is the influence of Movies on audience, especially children. Improper regulation/monitor/classification of content can have very critical impact on society, children in specific. Government regulatory bodies like _Classification & Ratings Administration (CARA)_[1] by _Motion Picture Association of America (MPAA)_[2], in USA, regulates the content by provided appropriate classification of Parental Guide for Movies aired in USA. The rating provided by these agencies are generally based on potential violence, drug usage, and nudity etc., where a committee of 2-3 people manually watch a movie and provide ratings based on their observation. Though this approach of manual rating a movie may seem accurate it severely suffers from standardization as it can sometime be opinion driven (those there are certain guidelines for rating generation). In order to address this problem of slow, manual and subjective approach we are proposing a AI driven solution where a deep learning model can be trained to predict intensity of critical factors like violence, foul language, drug/alcohol usage etc., from movie script, which can later be used to provide overall rating for movie view-ability for children. In the current solution proposed we will be rating a movie on the following dimensions sex, violence, profanity, drugs and intense

**File Names for EDA & Model Development:** Movie Script Extraction.ipynb, Script Pre-processing.ipynb, Parental Guide - EDA\_Intense\_light model.ipynb, Parental Guide - EDA\_Profanity.ipynb, Parental Guide - EDA\_Sex\_Light Model.ipynb, Parental Guide - EDA\_Violence\_For Size Reduction.ipynb

**Folder:** Code

**Description:** _Movie Script Extraction.ipynb_ contains the code for extracting and processing scripts from IMSDB while rest of the files have code for EDA & Model Development

## Exploratory Data Analysis

Programming Language: Python

IDE: Jupyter Notebook

Packages: Pandas, Numpy, NLTK, Beautifulsoup, Wordcloud

### Script Extraction from Web (Code Snip):

![](RackMultipart20230427-1-uzlkk0_html_54daf6d063afd676.png)

### Script Pre-Processing (Sample Code Snip):

![](RackMultipart20230427-1-uzlkk0_html_4736fbdcaead9fc7.png)

### Data Exploration (Sample Code Snip – For word cloud):

![](RackMultipart20230427-1-uzlkk0_html_1be397fd590a0652.png)

## Model Development

Programming Language: Python

IDE: Jupyter Notebook

Packages: Tensorflow, Keras


## Web Application Development

**File Names:** sriptRatingPrediction.py and appeal.css

**Folder:** webapp/code

**Description:** sriptRatingPrediction.py is the main file for running streamlit app while appeal.css is for custom styling

**Programming Language** : Python, HTML, CSS

**IDE:** Visual Studio Code

**Packages:** Streamlit
