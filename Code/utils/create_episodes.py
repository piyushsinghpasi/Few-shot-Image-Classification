import numpy as np
import pandas as pd
import random
random.seed(10)
np.random.seed(10)

def make_episodes(ann, N, K):
    d = {"img_name":[str(x) for x in range(len(ann))], "class":ann}
    df = pd.DataFrame(d)
    df = df.sample(frac=1).reset_index(drop=True)

    ids = df["class"].unique()
    random.shuffle(ids)
    df = df.set_index("class").loc[ids].reset_index()


    all_classes = df["class"].unique()

    def shuffleClasses(arr, seed):
        random.seed(seed)
        random.shuffle(arr)


    min_class_chunk = 1e100
    chunks = {}
    for c in all_classes:
        df_c = df[df['class'] == c]
        chunks[c] = [" ".join(df_c[i:i+K]['img_name'].tolist()) for i in range(0,df_c.shape[0],K)]
        min_class_chunk = min(min_class_chunk, len(chunks[c])-1)

    

    def get_episode(all_classes):
        all_episodes = pd.DataFrame()
        for idx in range(0,min_class_chunk):
            shuffleClasses(all_classes, idx)
            s = [all_classes[i:i+N] for i in range(0,len(all_classes), N)]
            if len(s[-1]) < N:
                s = s[:-1]

            for episode in s:
                support_set = "@".join([chunks[c][idx] for c in episode])
                support_label = "@".join([str(c) for c in episode])
                query_set = [(chunks[c][-1],c) for c in episode]
                random.shuffle(query_set)
                query = "@".join([q for q,_ in query_set])
                query_label = "@".join([str(c) for _,c in query_set])
                
                ep = {
                    "support_set": support_set,
                    "support_label": support_label,
                    "query_set": query,
                    "query_label":query_label
                }
        
                df_ep = pd.DataFrame([ep])
                all_episodes = pd.concat([all_episodes, df_ep], ignore_index=True)
        return all_episodes

    train_split = int(len(all_classes)* 0.7)
    train_class = all_classes[0: train_split]
  
    val_split = int(len(all_classes)*0.1)
    val_class = all_classes[train_split: train_split + val_split]

    test_class = all_classes[train_split + val_split :]

    return get_episode(train_class), get_episode(test_class), get_episode(val_class)


