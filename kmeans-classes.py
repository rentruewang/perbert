import os
from torch import optim
from tqdm import tqdm
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from torch.nn import Linear, functional as F
from torch.optim import Adam

DIM = 2
CLUSTERS = 4
POINTS = 25000
SCALE = 6
DEVICE = "cuda"
ITERS = 10000

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
all_points = torch.cat([samples.cpu(), centers.cpu()], 0).cpu().numpy()

labels = labels.cpu().numpy()
predicted = model(samples).argmax(-1).cpu().numpy()

df = DataFrame(
    {
        "x": all_points[:, 0],
        "y": all_points[:, 1],
        "labels": list(labels) + [CLUSTERS] * len(centers),
        "predict": list(predicted) + [CLUSTERS] * len(centers),
    }
)
print(df)

os.makedirs("img", exist_ok=True)
sns.set_palette("Paired")

sns.scatterplot(data=df, x="x", y="y", hue="labels")
plt.savefig("img/kmeans.png")
plt.clf()

sns.scatterplot(data=df, x="x", y="y", hue="predict")
plt.savefig("img/predict.png")
plt.clf()
