import os

import numpy as np
import torch
from pandas import DataFrame
from plotly import express as px
from plotly import subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from torch.nn import Linear
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

DIM = 8
CLUSTERS = 4
POINTS = 25000
SCALE = 6
DEVICE = "cpu"
ITERS = 1000

# Generate clusters
mix = Categorical(torch.ones([CLUSTERS]))
centers = torch.rand([CLUSTERS, DIM]) * SCALE
stddev = torch.rand([CLUSTERS, DIM])
comp = Independent(Normal(centers, stddev), 1)
gmm = MixtureSameFamily(mix, comp)
samples = gmm.sample([POINTS])


# Kmeans clusters
km = KMeans(CLUSTERS, max_iter=5000)
labels = km.fit_predict(X=samples)

# Training

model = Linear(DIM, CLUSTERS).to(DEVICE)
samples = samples.to(DEVICE)
labels = torch.tensor(labels, device=DEVICE).long()
optimizer = Adam(model.parameters())

for _ in tqdm(range(ITERS)):
    loss = F.cross_entropy(model(samples), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting
pca = PCA()
all_points = torch.cat([samples.cpu(), centers.cpu()], 0).cpu().numpy()

labels = labels.cpu().numpy()
predicted = model(samples).argmax(-1).cpu().numpy()
labels = np.array(list(labels) + [CLUSTERS] * len(centers))
predicted = np.array(list(predicted) + [CLUSTERS] * len(centers))

df = DataFrame(all_points)
df["labels"] = labels
df["predict"] = predicted

print(df)
axises = pca.fit_transform(all_points)
print(axises.shape)

df["first"] = axises[:, 0]
df["second"] = axises[:, 1]

os.makedirs("img", exist_ok=True)


fig = px.scatter_matrix(df, dimensions=range(DIM), color="labels")
fig.show()

fig = px.scatter_matrix(df, dimensions=range(DIM), color="predict")
fig.show()


# px.savefig(f"img/{DIM}-{CLUSTERS}.png")
