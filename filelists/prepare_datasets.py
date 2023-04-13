import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 10
df = pd.read_csv('Vax_Culture.csv')
if 'tweet_text' not in df.columns:
    raise Exception("Column 'tweet_text' not found!")

tweet_train, tweet_test = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df["communicated_message"])
tweet_train, tweet_val = train_test_split(tweet_train, test_size=0.25, random_state=random_state, stratify=tweet_train["communicated_message"])

tweet_train.to_pickle('./df_train')
tweet_val.to_pickle('./df_val')
tweet_test.to_pickle('./df_test')

