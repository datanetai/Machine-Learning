import wandb
run = wandb.init()
project_path ='ameerhamza0220/credit-default-risk/'
cat_models = []
lgb_models = []
for i in range(0,5):
    artifact = run.use_artifact(project_path+'catboost_model_{0}:v0'.format(i))
    cat_models.append(artifact.download())
    if i==0:
        artifact = run.use_artifact(project_path+'lgb_clf{0}:v1'.format(i))
        lgb_models.append(artifact.download())
        continue
    artifact = run.use_artifact(project_path+'lgb_clf{0}:v0'.format(i))
    lgb_models.append(artifact.download())

