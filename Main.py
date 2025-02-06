#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
file_paths=['0000.parquet','0001.parquet','0002.parquet','0003.parquet','0004.parquet','0005.parquet']
df=pd.read_parquet(file_paths)


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


import matplotlib.pyplot as plt
print(df.isnull().sum())
print(df.duplicated().sum())


# In[ ]:


# In[8]:


import isodate
def duration_to_seconds(duration):
    if pd.isnull(duration):
        return np.nan
    try:
        parsed_duration = isodate.parse_duration(duration)
        return parsed_duration.total_seconds()
    except Exception as e:
        print(f"Error parsing duration: {duration} - {e}")
        return np.nan
# Apply the function to the 'duration' column
df['duration_seconds'] = df['duration'].apply(duration_to_seconds)


# In[9]:


df=df.head(10000)


# In[10]:


# Descriptive statistics for video duration
print(df['duration_seconds'].describe())
import matplotlib.pyplot as plt
df['duration_seconds'].hist(bins=10)
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Video Duration Distribution')
plt.show()


# Most videos are short: The majority of videos fall within the 0-20 second range, indicating a strong skew towards shorter durations.
# Few long videos: A small number of videos extend beyond 50 seconds, with very few reaching over 100 seconds.
# Right-skewed distribution: The distribution is heavily right-skewed, meaning there are more short videos than long ones.

# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# Boxplot for duration_seconds
plt.figure(figsize=(10, 6))
plt.boxplot(df['duration_seconds'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot of Video Durations (seconds)")
plt.xlabel("Duration (seconds)")
plt.show()

# Log-scaled histogram
plt.figure(figsize=(10, 6))
plt.hist(df['duration_seconds'], bins=100, log=True, color='skyblue', edgecolor='black')
plt.title("Log-Scaled Histogram of Video Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency (log scale)")
plt.show()

# Cumulative Distribution Function (CDF)
sorted_durations = np.sort(df['duration_seconds'])
cdf = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)

plt.figure(figsize=(10, 6))
plt.plot(sorted_durations, cdf, color="blue")
plt.title("CDF of Video Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Cumulative Probability")
plt.grid()
plt.show()


# Very few long videos: The tail of the distribution beyond 100 seconds is nearly flat, meaning that long videos are rare.
# Right-skewed distribution: Similar to the histogram, this confirms that the dataset is dominated by shorter videos.

# In[12]:


filtered_df = df[df['duration_seconds'] > 0]
print(filtered_df.shape)  # Verify the number of rows after removal
print(filtered_df['duration_seconds'].describe())  # Summary statistics after removal


# In[13]:


df=filtered_df[filtered_df['duration_seconds'] < 175]
df.describe()


# In[14]:


df.head(5)


# In[15]:


# Group by 'page_dir' and calculate statistics
duration_by_page = df.groupby('page_dir')['duration_seconds'].describe()
print(duration_by_page)


# In[28]:


unique_page_dir_count = df['page_dir'].nunique()
print(f"Number of unique page directories: {unique_page_dir_count}")


# ## Stratified Sampling

# In[23]:


sampled_df = df.groupby('page_dir', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=42))  # Adjust frac for sample size
print(sampled_df.shape)


# In[18]:


sampled_df.head(5)


# # Checking Words with highest frequency

# In[36]:


import pandas as pd
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
sampled_df["cleaned_name"] = sampled_df["name"].astype(str).apply(preprocess)
# Word Frequency Analysis
word_freq = Counter(" ".join(sampled_df["cleaned_name"]).split())
common_words = word_freq.most_common(20)  # Top 20 words
print("Top 20 Most Frequent Words in Descriptions:")
print(common_words)
# TF-IDF Analysis
vectorizer = TfidfVectorizer(max_features=20)  # Top 20 features
tfidf_matrix = vectorizer.fit_transform(sampled_df["cleaned_name"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print("\nTF-IDF Scores for Top Words:")
print(tfidf_df.mean().sort_values(ascending=False))  # Average importance of each word


# In[29]:


page_dir_counts = sampled_df['page_dir'].nunique()
print("\nPage Directory Counts:\n", page_dir_counts)


# In[31]:


page_dir_counts = sampled_df['page_dir'].value_counts()
most_rows_page_dir = page_dir_counts.idxmax()
most_rows_count = page_dir_counts.max()
print(f"The page directory with the most rows is: {most_rows_page_dir} with {most_rows_count} rows.")


# In[37]:


sampled_df.columns


# In[ ]:


vectorizer = TfidfVectorizer(stop_words='english', max_features=10)  # Limit to 10 most important words
X = vectorizer.fit_transform(sampled_df["cleaned_name"])
top_words = vectorizer.get_feature_names_out()
word_scores = X.sum(axis=0).A1 
word_score_dict = dict(zip(top_words, word_scores))
sorted_word_score = sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)
most_relevant_word = sorted_word_score[0][0]
print(f"Most relevant one-word project directory name: {most_relevant_word}")


# In[39]:


def most_common_word(group):
    text = " ".join(group["cleaned_name"])
    word_freq = Counter(text.split())
    return word_freq.most_common(1)[0][0] 
common_words_per_page_dir = sampled_df.groupby('page_dir').apply(most_common_word)
print("\nMost Common Word in Each Project Directory:")
print(common_words_per_page_dir)


# In[ ]:




