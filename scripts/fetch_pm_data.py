import pandas as pd
from psaw import PushshiftAPI

api = PushshiftAPI()

gen = api.search_submissions(subreddit='papermario')

df = pd.DataFrame([sub.d_ for sub in gen])

df.to_csv('packages/regression_model/regression_model/datasets/pm_data.csv') # mode='a', header=not os.path.exists('pm_data.csv')