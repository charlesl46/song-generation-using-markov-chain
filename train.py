import pandas as pd
import markovify
import json
from rich.console import Console

C = Console()

with C.status("Loading data"):
    df = pd.read_csv("spotify_millsongdata.csv")
    #print(f"shape of data = {df.shape}")
    lyrics = " ".join(df["text"].values)
with C.status("Building model"):
    M = markovify.Text(lyrics,state_size=10,retain_original=False)
with C.status("Saving model"):
    model_json = M.to_json()
    json.dump(model_json,open("model.json","w"))