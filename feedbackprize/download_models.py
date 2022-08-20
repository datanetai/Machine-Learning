# download models for inference form wandb
import wandb
run = wandb.init()
project_path ='ameerhamza0220/credit-default-risk/'
model_name = None
artifact = run.use_artifact(project_path,model_name)
model_dir = artifact.download()
print("model saved to {}".format(model_dir))
