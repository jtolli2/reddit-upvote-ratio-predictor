## Reddit Upvote Ratio Predictor TODO

Current goal is pipeline implementation all the way up to CI/CD integration and then improve upon that

### Regression Model

- [x] Implement model pipeline using XGBRegressor to fit data make predictions
  - [ ] Review currently used features for data leakage
  - [ ] Improvine model prediction accuracy
  - [ ] Setup data validation
  - [ ] Expand predictions beyond Paper Mario subreddit
- [x] Create dataset loading scripts
  - [ ] Implement incremental loading of data
  - [ ] Include more subreddits than just r/PaperMario
- [x] Configure Tox to run train/predict jobs
  - [ ] Further tweaking needed to properly run each environment

### Model Serving/Deployment

- [x] Setup FastAPI and Uvicorn to serve prediction requests
- [ ] Complete input schema for Reddit data
  - [ ] Add example request
- [ ] Setup error handling for requests

### CI/CD

- [ ] Integrate model build/deployment with CircleCI

### Github

- [x] Create TODO
- [ ] Create README
