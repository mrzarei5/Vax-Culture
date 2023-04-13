# Vac-Culture Dataset
Vax-Culture consists of 6373 manually annotated vaccine-related English tweets. The human-provided annotations include vaccine-hesitancy stance, indication of any misinformation in tweet, the entities criticized and supported in each tweet and the communicated message of each tweet. Please refer to ['Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter'](address will be added) for additional information regarding the collection and annotation procedures of Vax-Culture.

## Directory Structure

### Vax_Culture.csv 

Vax_Culture.csv includes the tweets ids and the annotations related to each tweet. This csv file containts the following columns:

- tweet_id: Unique indentifier of each tweet that can be used to fetch the original tweet from Twitter.

- meaning: Intended meaning of the tweet

- comminicated_message: Message communicated in the tweet. The value can be 'Anti-vaccine', 'Pro-vaccine' or 'Unsure about the vaccine'.

- inaccurate_or_misleadning: Specifies whether any information in the tweet seems misleading/inaccurate or not. The value is eather 0 or 1.

- criticism_politicians: Whether the tweet criticizes politicians or not. The value is eather 0 or 1

- criticism_pharmaceutical_companies: Whether the tweet criticizes pharmaceutical companies or not. The value is eather 0 or 1.
 
- criticism_public_health_officials: Whether the tweet criticizes public health officials or not. The value is eather 0 or 1.
 
- criticism_anti-vaxxers: Whether the tweet critisizes anti-vaxxers or not. The value is eather 0 or 1.
 
- criticism_vaccine_mandates: Whether the tweet criticizes vaccine mandates or not. The value is eather 0 or 1.
 
- criticism_vaccine_safety: Whether the tweet criticizes vaccine safety or not. The value is eather 0 or 1.
 
- criticism_conservative_media: Whether the tweet criticizes conservative media or not. The value is eather 0 or 1.
 
- criticism_mainstream_media: Whether the tweet criticizes mainstream media or not. The value is eather 0 or 1.
 
- criticism_public_health_policy: Whether the tweet criticizes public health policy or not. The value is eather 0 or 1.
 
- criticism_democrats_or_liberals: Whether the tweet criticizes democrats/liberals or not. The value is eather 0 or 1.
 
- criticism_government: Whether the tweet criticizes government or not. The value is eather 0 or 1.
 
- criticism_vaccine_effectiveness: Whether the tweet criticizes vaccine effectiveness or not. The value is eather 0 or 1.
 
- criticism_other: Any other entities that have been criticized in the tweet and have been identified by the annotators. This field is in the form of free text. 
 
- support_science: Whether the tweet supports/promotes science or not. The value is eather 0 or 1.
 
- support_choice_freedom: Whether the tweet supports/promotes freedom of choice or not. The value is eather 0 or 1.
 
- support_natural_health: Whether the tweet supports/promotes natural health or not. The value is eather 0 or 1.
 
- support_vaccines: Whether the tweet supports/promotes vaccines or not. The value is eather 0 or 1.
 
- support_small_business: Whether the tweet supports/promotes small business or not. The value is eather 0 or 1.
 
- support_alternative_remedies: Whether the tweet supports/promotes alternative remedies or not. The value is eather 0 or 1.

- support_relaxed_approach: Whether the tweet supports/promotes a more relaxed approach or not. The value is eather 0 or 1.
 
- support_more_information: Whether the tweet supports/promotes waiting for more information or not. The value is eather 0 or 1.
 
- support_public_health_interventions: Whether the tweet supports/promotes public health interventions or not. The value is eather 0 or 1.
 
- support_global_response: Whether the tweet supports/promotes global response or not. The value is eather 0 or 1.
 
- support_religious_beliefs: Whether the tweet supports/promotes religious beliefs or not. The value is eather 0 or 1.
 
- support_other: Any other entities that have been supported/promoted in the tweet and have been identified by the annotators. This field is in the form of free text. 


### TweetsIDs
The TweetsIDs folder comprises four text files, each containing identifiers for a different set of tweets. The file "ids.txt" includes identifiers for all tweets in the same order as "Vax_Culture.csv". The files "ids_train.txt", "ids_val.txt", and "ids_test.txt" include identifiers for the separate subsets of tweets that were used in the paper "Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter".
