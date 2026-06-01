import matplotlib.pyplot as plt
import numpy as np

from density_equalizing_transformations import *

np.random.seed(0)
points = np.random.normal(loc=(0.5, 0.5), scale=0.1, size=(1000, 2))
points = np.clip(points, 0.0, 1.0)
points = points.astype(np.float32)

algorithms = [
    ("Original", lambda _: None),
    ("Integral Images", density_equalizing_transformation_integral_images),
    ("Sector-based", density_equalizing_transformation_sector_based),
    ("Multiresolution", density_equalizing_transformation_multiresolution),
]

figure, axes = plt.subplots(nrows=1, ncols=len(algorithms), figsize=(len(algorithms) * 4, 4))

for i, (title, transformation) in enumerate(algorithms):
    points_transformed = points.copy()
    transformation(points_transformed)

    axis = axes[i]
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title)

    axis.scatter(points_transformed[:, 0], points_transformed[:, 1], s=20, color="lightgray", edgecolors="black", linewidths=0.2)

figure.savefig("example.png", dpi=300, bbox_inches="tight")
plt.show()